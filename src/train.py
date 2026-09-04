"""Train a freight-rate model and produce all assessment deliverables.

Usage:
    python src/train.py

Outputs:
    validation_predictions.csv
    data/december_chart_inputs.csv (filled in, in place is avoided -> written to
        december_chart_inputs_filled.csv)
    reports/metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

from features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_COLS,
    CityEffectEncoder,
    add_date_features,
    impute_numeric,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

RANDOM_STATE = 42


def load_train() -> pd.DataFrame:
    df = pd.read_csv(DATA / "train_test.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_medians(df: pd.DataFrame) -> dict:
    """Median for every numeric feature column, computed on the training
    set only. Used both to fill NaNs and to backfill columns that are
    entirely absent from a given inference file (e.g. December scenario)."""
    return {col: float(df[col].median()) for col in NUMERIC_COLS}


def fit_city_encoders(df: pd.DataFrame):
    """Fit a baseline log-linear model (distance + equipment) then target-encode
    the residuals per pickup/delivery city. Returns encoders + baseline model
    so the same residualisation can be replicated at inference time if needed."""
    baseline_X = pd.get_dummies(df[["equipment"]], prefix="equip")
    baseline_X["log_distance"] = np.log(df["distance"])
    baseline_y = np.log(df["posted_rate"])
    baseline_model = LinearRegression().fit(baseline_X, baseline_y)
    residual = baseline_y - baseline_model.predict(baseline_X)

    pickup_encoder = CityEffectEncoder().fit(df["pickup"], df["pickup_lat"], df["pickup_lon"], residual)
    delivery_encoder = CityEffectEncoder().fit(df["delivery"], df["delivery_lat"], df["delivery_lon"], residual)
    return pickup_encoder, delivery_encoder


def build_city_coords(train_df: pd.DataFrame) -> dict:
    """Map city name -> (lat, lon), built from whichever role (pickup or
    delivery) each city appears in within the training data. Needed because
    the December scenario file gives only city names, not coordinates."""
    coords = {}
    for _, row in train_df[["pickup", "pickup_lat", "pickup_lon"]].drop_duplicates("pickup").iterrows():
        coords[row["pickup"]] = (row["pickup_lat"], row["pickup_lon"])
    for _, row in train_df[["delivery", "delivery_lat", "delivery_lon"]].drop_duplicates("delivery").iterrows():
        coords.setdefault(row["delivery"], (row["delivery_lat"], row["delivery_lon"]))
    return coords


def fill_missing_coords(df: pd.DataFrame, city_coords: dict) -> pd.DataFrame:
    df = df.copy()
    if "pickup_lat" not in df.columns:
        df["pickup_lat"] = df["pickup"].map(lambda c: city_coords.get(c, (np.nan, np.nan))[0])
        df["pickup_lon"] = df["pickup"].map(lambda c: city_coords.get(c, (np.nan, np.nan))[1])
    if "delivery_lat" not in df.columns:
        df["delivery_lat"] = df["delivery"].map(lambda c: city_coords.get(c, (np.nan, np.nan))[0])
        df["delivery_lon"] = df["delivery"].map(lambda c: city_coords.get(c, (np.nan, np.nan))[1])
    return df


def build_features(df: pd.DataFrame, medians: dict, pickup_enc: CityEffectEncoder,
                    delivery_enc: CityEffectEncoder, city_coords: dict) -> pd.DataFrame:
    df = fill_missing_coords(df, city_coords)
    df = add_date_features(df)
    df = impute_numeric(df, medians)
    df = df.copy()
    df["pickup_effect"] = pickup_enc.transform(df["pickup"], df["pickup_lat"], df["pickup_lon"])
    df["delivery_effect"] = delivery_enc.transform(df["delivery"], df["delivery_lat"], df["delivery_lon"])
    df["equipment"] = df["equipment"].astype("category")
    return df


def make_lgb_dataset(df: pd.DataFrame, target: np.ndarray | None = None):
    X = df[MODEL_FEATURES].copy()
    if target is not None:
        return lgb.Dataset(X, label=target, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False)
    return X


LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=40,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    seed=RANDOM_STATE,
)


def time_based_holdout_eval(train_df: pd.DataFrame, medians: dict) -> dict:
    """Mimic the real task (forecasting future, unseen months) by holding out
    the last two months of history rather than a random split."""
    cutoff = train_df["date"].max() - pd.DateOffset(months=2)
    fit_part = train_df[train_df["date"] <= cutoff].reset_index(drop=True)
    holdout_part = train_df[train_df["date"] > cutoff].reset_index(drop=True)

    pickup_enc, delivery_enc = fit_city_encoders(fit_part)
    city_coords = build_city_coords(train_df)
    fit_feat = build_features(fit_part, medians, pickup_enc, delivery_enc, city_coords)
    holdout_feat = build_features(holdout_part, medians, pickup_enc, delivery_enc, city_coords)

    y_fit = np.log(fit_part["posted_rate"].values)
    y_hold = holdout_part["posted_rate"].values

    train_set = make_lgb_dataset(fit_feat, y_fit)
    model = lgb.train(LGB_PARAMS, train_set, num_boost_round=600)

    pred_log = model.predict(make_lgb_dataset(holdout_feat))
    pred = np.exp(pred_log)

    rmse = float(np.sqrt(mean_squared_error(y_hold, pred)))
    mape = float(mean_absolute_percentage_error(y_hold, pred))
    baseline_pred = np.full_like(y_hold, fit_part["posted_rate"].mean())
    baseline_rmse = float(np.sqrt(mean_squared_error(y_hold, baseline_pred)))

    return {
        "holdout_start": str(holdout_part["date"].min().date()),
        "holdout_end": str(holdout_part["date"].max().date()),
        "n_holdout_rows": int(len(holdout_part)),
        "rmse": rmse,
        "mape": mape,
        "naive_mean_baseline_rmse": baseline_rmse,
        "improvement_vs_baseline_pct": float(100 * (1 - rmse / baseline_rmse)),
    }


def main():
    train_df = load_train()
    medians = compute_medians(train_df)

    print("Running time-based holdout evaluation (last 2 months of train data)...")
    metrics = time_based_holdout_eval(train_df, medians)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nRefitting on full train_test.csv for final predictions...")
    pickup_enc, delivery_enc = fit_city_encoders(train_df)
    city_coords = build_city_coords(train_df)
    full_feat = build_features(train_df, medians, pickup_enc, delivery_enc, city_coords)
    y_full = np.log(train_df["posted_rate"].values)
    full_set = make_lgb_dataset(full_feat, y_full)
    final_model = lgb.train(LGB_PARAMS, full_set, num_boost_round=600)

    feat_importance = pd.Series(
        final_model.feature_importance(importance_type="gain"), index=MODEL_FEATURES
    ).sort_values(ascending=False)
    print("\nFeature importance (gain):")
    print(feat_importance)

    # ---- validation.csv -> validation_predictions.csv ----
    val_df = pd.read_csv(DATA / "validation.csv")
    val_df["date"] = pd.to_datetime(val_df["date"])
    val_feat = build_features(val_df, medians, pickup_enc, delivery_enc, city_coords)
    val_pred = np.exp(final_model.predict(make_lgb_dataset(val_feat)))

    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    pred_map = dict(zip(val_df["load_id"], val_pred))
    template["predicted_rate"] = template["load_id"].map(pred_map).round(2)
    assert template["predicted_rate"].isna().sum() == 0, "missing predictions for some load_id"
    out_path = ROOT / "validation_predictions.csv"
    template.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(template)} rows)")

    # ---- december_chart_inputs.csv (fill predicted_rate) ----
    dec_df = pd.read_csv(DATA / "december_chart_inputs.csv")
    dec_df["date"] = pd.to_datetime(dec_df["date"])
    dec_feat = build_features(dec_df, medians, pickup_enc, delivery_enc, city_coords)
    dec_pred = np.exp(final_model.predict(make_lgb_dataset(dec_feat)))
    dec_out = pd.read_csv(DATA / "december_chart_inputs.csv")
    dec_out["predicted_rate"] = np.round(dec_pred, 2)
    dec_out_path = ROOT / "december_chart_inputs.csv"
    dec_out.to_csv(dec_out_path, index=False)
    print(f"Wrote {dec_out_path} ({len(dec_out)} rows)")

    metrics["december_predicted_rate_range"] = [float(dec_pred.min()), float(dec_pred.max())]
    metrics["feature_importance_gain"] = feat_importance.to_dict()
    with open(REPORTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nSaved metrics to {REPORTS / 'metrics.json'}")


if __name__ == "__main__":
    main()
