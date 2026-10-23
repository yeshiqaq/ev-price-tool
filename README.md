# PriceSpark: What Drives EV Resale Value?

**EBA5012 Predictive Analytics Lifecycle Practice Module · Group 21 (Team PriceSpark)**

A fair-price and price-cut review tool for U.S. used-EV dealers, built on 15,155 Cars.com battery-electric listings (April 2023) and 9,440 NHTSA owner complaints received before 1 May 2023.

| Question | Result (test set, 3,031 listings) |
|---|---|
| Q1 Fair price | XGBoost: MAPE 5.94%, 83.4% within ±10% (benchmark: 9.59%, 61.4%) |
| Q2 Price drivers | Same trim: −8.4% per year, −3.0% per 10,000 miles, −4.5% for an accident |
| Q3 Complaints | Small but consistent gain: +0.05 pts MAPE (XGBoost), better in 5 of 5 folds |
| Q4 Price cuts | Decision tree review score: AUC 0.630, recall 73.8% (80% target not met), precision 70.4% |

The price-cut output is an **uncalibrated review score** trained on recorded historical cuts, not a probability of a future cut. All prices reflect the April 2023 listing market (asking prices, not sale prices).

## Team

| Member | Role |
|---|---|
| M1 Jin Zihao | Data cleaning, exploration, condition features |
| M2 Zhang Ruijing | Complaint matching, price-cut model |
| M3 Jin ZhiShuo | MySQL warehouse, SQL, AWS deployment |
| M4 Choo Kai Chuen Caleb | Complaint text analytics, findings, slides and report |
| M5 Byun Hyunbin | Price models, shared preprocessing (`src/prep.py`), MLflow |

## Repository structure

```
notebooks/   01_cleaning (M1), 02_complaints (M2), 04_eda (M1), 06_price_cut (M2)
src/         prep.py + condition_features.py (M5, shared - do not overwrite)
             train_price_models.py, finalize_model.py, make_fair_prices.py,
             condition_test.py, make_charts.py (M5)
             price_cut.py, train_price_cut.py, prepare_complaints.py,
             join_reliability.py, download_nhtsa.py (M2)
             app.py, server.py (M3, FastAPI service)
scripts/m1/  M1 cleaning and EDA pipeline (uses its own copy of the condition rules)
scripts/m2/  M2 hand-over verification
sql/         01_create_tables ... 05_validation (M3), star_schema.mwb, EER diagram
models/      price_model_v3.joblib (registered ev_price_model v2 @champion)
             price_cut_model.joblib + metadata (registered ev_price_cut_model v1 @champion)
             dim_reliability.csv (complaint features by model line and year)
results/     m1 (data quality, EDA figures), m2 (Q4 metrics, ROC, tree),
             m3 (warehouse validation, business queries), m5 (model comparison, errors, charts)
data/        small reference files: data dictionary, train/test/fold IDs, model-name lookups
docs/        evidence (deployment, warehouse, MLflow screenshots) and member hand-over notes
```

## Data (not included)

Raw data is not stored in this repository:

- **Cars.com listings:** Novikov, A. (2023), *Used Cars Dataset*, Kaggle (762,091 U.S. listings, April 2023).
- **NHTSA complaints:** downloadable complaint files (2010–2024); `src/download_nhtsa.py` fetches them.

The cleaned listings and complaint text are kept in the team's shared folder.

## How to reproduce

1. **Clean listings (M1):** `scripts/m1/run_m1.py` → 15,155 BEV listings, data dictionary, quality log.
2. **Complaints (M2/M4):** `src/download_nhtsa.py`, `src/prepare_complaints.py`, `notebooks/02_complaints.ipynb` → 9,440 EV complaints and model-line matches; text features → `models/dim_reliability.csv`.
3. **Warehouse (M3):** run `sql/01_create_tables.sql` → `02_load.sql` → `03_business_questions.sql` → `04_extract.sql` (view `v_model_dataset`) → `05_validation.sql`.
4. **Price models (M5):** `src/train_price_models.py`, then `src/finalize_model.py` → `models/price_model_v3.joblib`; `src/make_fair_prices.py` → out-of-fold fair prices for Q4.
5. **Price-cut model (M2):** `src/train_price_cut.py` or `notebooks/06_price_cut.ipynb` → `models/price_cut_model.joblib`.

Every model uses the same train/test split and five folds (`data/train_ids_v1.csv`, `data/test_ids_v1.csv`, `data/fold_ids_v1.csv`, seed 42). Choices were made on cross-validation; the test set was scored once.

Environments: `requirements-train.txt` (training), `requirements-serve.txt` (API, Python 3.13), `requirements-text-optional.txt` (sentiment models).

## Deployment

Real-time inference on AWS (Singapore region), deployed 19 October 2026 by M3:

- FastAPI in a Docker container (Python 3.13) → Amazon ECR → AWS Lambda, exposed through a **Lambda Function URL**
- Swagger UI: `https://ncnbucti2my7ygmwzpdf5vq3b40etqnp.lambda-url.ap-southeast-1.on.aws/docs`
- `POST /api/v1/used-ev/price` returns the fair price, the price-cut review score and a review flag
- CloudWatch logs every invocation; input-drift monitoring is planned

Example: a 2020 Tesla Model 3 Long Range, 35,000 miles, asking 42,000 US dollars → fair price 38,814; review score 0.620 (threshold 0.482) → flagged for review.

## Status (20 October 2026)

- [x] Warehouse, SQL and validation
- [x] Price models trained, registered in MLflow (`ev_price_model` v2 @champion)
- [x] Price-cut model re-run on final fair prices, registered (`ev_price_cut_model` v1 @champion)
- [x] Dual-model API live on AWS Lambda
- [x] M3: push the deployed `app.py`, `price_cut.py` integration and Python 3.13 Dockerfiles; call `prep.add_lookups()` so blank ratings do not inflate prices
- [x] M5: push `src/prep.py` v6 (adds `add_lookups`) and `tests/test_parity.py`; parity test against the live URL should print `PARITY OK`
- [x] M4: add the complaint text-analytics notebook and results
- [ ] Input-drift monitoring
