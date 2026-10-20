# M5: price models (EV fair-price project)

Reproduces every M5 number in the slides and report with three commands.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # Python 3.13, pinned versions
```
Place M3's SQL extract at `data/model/model_dataset.csv`. The split and fold files are already in `data/model/` (`train_ids_v1.csv`, `test_ids_v1.csv`, `fold_ids_v1.csv`).

## Run (from the repo root)
```bash
python -m src.train_price_models      # tune 8 model families + benchmark, Q2, complaint ablation, MLflow (~25 min on 4 cores)
python -m src.condition_test          # none vs M1's vs M5's condition rules, same folds (MAPE + log-MAE)
python -m src.finalize_model --condition none   # final XGBoost (choice made on CV), test once, register @champion
python -m src.make_fair_prices --model models/price_model_v3.joblib --version v3   # fair prices for M2
python -m src.make_charts             # slide charts -> results/charts/
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db   # browse runs and the registry
```
Add `--quick` to the first command for a short smoke test (2 draws per search).

## Files
| Path | What it is |
|---|---|
| `src/prep.py` | Shared features and preprocessing (v5). Only M5 edits it. The saved model needs it at `src/prep.py`. |
| `src/condition_features.py` | M1's condition rules, copied unchanged. `prep.py` imports it, so the Docker image needs it too. |
| `src/condition_test.py` | With vs without condition features (M1's protocol) |
| `src/finalize_model.py` | Final model after the condition test: CV, one test score, error tables, MLflow version |
| `src/train_price_models.py` | Training, tuning, test scoring, error analysis, Q2 OLS, with/without tests, MLflow registration |
| `src/make_fair_prices.py` | `oof_fair_price_v3.csv`, `fair_price_nested_v3.csv` for M2's price-cut model |
| `src/make_charts.py` | PNG charts from `results/*.csv` |
| `models/price_model_v3.joblib` | **Deployed model** (XGBoost, no condition features): raw listing fields + complaint fields in, US$ out (no `np.exp` in the app). v2 = same model with condition features, kept for comparison. |
| `results/` | All tables (`*_v1.csv`), `summary_v1.json` (winner and tuned settings), charts, MLflow screenshots |

## Rules the code follows
- `random_state = 42` everywhere. Every model is cross-validated on the same folds (`fold_ids_v1`).
- Listing 13975 (M1's confirmed error) is removed from training only.
- Price, price drop, initial price, price-cut flag and outlier flags are never inputs (`prep.NEVER_INPUTS`).
- The winner is picked on CV MAPE. The test set is scored once and never used for choices.

## Suggested `.gitignore`
```
.venv/
mlflow/
__pycache__/
```
The MLflow database stores absolute artifact paths, so each machine rebuilds it by running the training script rather than copying it.
