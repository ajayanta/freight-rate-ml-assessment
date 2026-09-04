"""Feature engineering for the freight rate model.

Key design decisions (see report for full reasoning):
  * distance/equipment/weight/market_index/quote_signal are used directly.
  * date is decomposed into month + day-of-week + cyclical day-of-year,
    since exploratory analysis showed a mild seasonal pattern in rate/mile.
  * pickup/delivery cities carry a real, lane-specific rate premium beyond
    what distance + equipment explain (within-lane std of rate/mile is
    roughly half the overall std). We capture this with a smoothed target
    encoding on regression residuals.
  * validation.csv contains 8 cities never seen in train_test.csv. A plain
    target encoding would silently fall back to "no premium" (0) for those,
    which throws away good information given we still have their lat/lon.
    Instead we estimate a new city's premium as the distance-weighted
    average premium of its nearest known neighbours (by haversine distance).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_MI = 3958.8


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(a))


class CityEffectEncoder:
    """Smoothed target encoding of a city column, with KNN fallback for
    cities never observed during `fit`."""

    def __init__(self, smoothing: float = 20.0, n_neighbors: int = 3):
        self.smoothing = smoothing
        self.n_neighbors = n_neighbors
        self.city_effect_: dict[str, float] = {}
        self.city_coords_: dict[str, tuple[float, float]] = {}
        self.global_mean_ = 0.0

    def fit(self, cities: pd.Series, lats: pd.Series, lons: pd.Series, residual: pd.Series):
        frame = pd.DataFrame({"city": cities.values, "lat": lats.values,
                               "lon": lons.values, "residual": residual.values})
        self.global_mean_ = float(frame["residual"].mean())
        grouped = frame.groupby("city")["residual"].agg(["mean", "count"])
        n = grouped["count"]
        smoothed = (n * grouped["mean"] + self.smoothing * self.global_mean_) / (n + self.smoothing)
        self.city_effect_ = smoothed.to_dict()
        coords = frame.groupby("city")[["lat", "lon"]].first()
        self.city_coords_ = {c: (row.lat, row.lon) for c, row in coords.iterrows()}
        return self

    def _knn_fallback(self, lat: float, lon: float) -> float:
        if not self.city_coords_:
            return self.global_mean_
        known = list(self.city_coords_.items())
        dists = np.array([haversine_miles(lat, lon, c[1][0], c[1][1]) for c in known])
        idx = np.argsort(dists)[: self.n_neighbors]
        nearest_dists = dists[idx]
        nearest_effects = np.array([self.city_effect_[known[i][0]] for i in idx])
        if np.any(nearest_dists == 0):
            return float(nearest_effects[nearest_dists == 0].mean())
        weights = 1.0 / nearest_dists
        return float(np.average(nearest_effects, weights=weights))

    def transform(self, cities: pd.Series, lats: pd.Series, lons: pd.Series) -> np.ndarray:
        out = np.empty(len(cities), dtype=float)
        for i, (city, lat, lon) in enumerate(zip(cities.values, lats.values, lons.values)):
            if city in self.city_effect_:
                out[i] = self.city_effect_[city]
            else:
                out[i] = self._knn_fallback(lat, lon)
        return out


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["date"])
    df["month"] = dt.dt.month
    df["day_of_week"] = dt.dt.dayofweek
    doy = dt.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def impute_numeric(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """Fill missing values in existing columns, and add entirely missing
    columns (e.g. market_index/quote_signal are absent from the December
    scenario file, since it's a hypothetical future lane with no live
    market signal) using training-set medians as a neutral default."""
    df = df.copy()
    for col, val in medians.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = df[col].fillna(val)
    return df


NUMERIC_COLS = ["distance", "weight", "market_index", "quote_signal"]
MODEL_FEATURES = [
    "distance", "weight", "market_index", "quote_signal",
    "month", "day_of_week", "doy_sin", "doy_cos",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "pickup_effect", "delivery_effect", "equipment",
]
CATEGORICAL_FEATURES = ["equipment"]
