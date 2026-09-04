# Freight Rate Prediction Challenge

Solution to the Spotter Machine Learning Engineer take-home assessment.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Run

```bash
cd src
python train.py
```

This will:
1. Load `data/train_test.csv`.
2. Run a time-based holdout evaluation (last 2 months of history) and print
   RMSE / MAPE vs. a naive mean baseline.
3. Refit on the full training set.
4. Predict `data/validation.csv` and write `validation_predictions.csv`
   (repo root) in the exact `load_id,predicted_rate` format required.
5. Fill in `data/december_chart_inputs.csv`'s `predicted_rate` column and
   write the result to `december_chart_inputs.csv` (repo root).
6. Save evaluation metrics + feature importances to `reports/metrics.json`.

## Verify against the provided scorer

```bash
python -m pip install -r requirements.txt   # scorer's own light requirements
python score.py --predictions validation_predictions.csv \
                 --december-predictions december_chart_inputs.csv
```

This validates both output files and produces
`scorer_results/candidate_december.png`.

## Repository layout

```
data/                          # provided input files (unchanged)
src/
  features.py                  # feature engineering + city-effect encoder
  train.py                     # training, evaluation, and prediction pipeline
reports/
  metrics.json                 # holdout metrics + feature importances
validation_predictions.csv     # final deliverable
december_chart_inputs.csv      # filled-in December scenario file
score.py                       # provided scorer (unmodified)
requirements.txt
```

## Approach summary

See the accompanying report (`Freight_Rate_Assessment_Report.docx`/PDF) for
full detail on data exploration, data-quality handling (missing values,
8 cities unseen in validation, and the December file's missing columns),
the time-based train/validation split, model choice (LightGBM on
log-rate), and results (RMSE ≈ $641, MAPE ≈ 6.0% on a Sep–Oct 2025
holdout, a 58% improvement over a naive mean baseline).
