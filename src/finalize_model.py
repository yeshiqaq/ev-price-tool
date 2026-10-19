"""Fit, test and register the final price model after the condition test (owner: M5).

    python -m src.finalize_model --condition none      # none | m1 | m5

Uses the winning family and tuned settings from results/summary_v1.json (chosen on CV MAPE
in train_price_models) and the condition setting chosen on CV in condition_test.
The test set is scored once, here, after both choices were made.
Writes models/price_model_v3.joblib, results/*_v2.csv and registers a new
`ev_price_model` version with the @champion alias.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit, cross_val_predict
from xgboost import XGBRegressor

from src import prep
from src.train_price_models import (BAND_LABELS, BANDS, benchmark_oof, benchmark_predict,
                                    fold_mapes, load, metrics)


def build(params, condition):
    reg = XGBRegressor(tree_method="hist", n_jobs=-1, random_state=prep.RANDOM_STATE, **params)
    if condition == "none":
        return prep.make_model(reg, use_condition=False, scale=False)
    return prep.make_model(reg, rules=condition, scale=False)


def by_group(frame, pred, col):
    g = frame.assign(pred=pred)
    return g.groupby(col, observed=True).apply(
        lambda s: pd.Series({"n": len(s), **metrics(s["price"], s["pred"])}), include_groups=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/model/model_dataset.csv")
    ap.add_argument("--ids", default="data/model")
    ap.add_argument("--condition", default="none", choices=["none", "m1", "m5"])
    ap.add_argument("--summary", default="results/summary_v1.json")
    ap.add_argument("--mlflow-uri", default="sqlite:///mlflow/mlflow.db")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)

    summ = json.loads(Path(args.summary).read_text())
    assert summ["winner"] == "XGBoost", "finalize_model is written for the XGBoost winner"
    train, test = load(args.data, Path(args.ids))
    X, y, fold = prep.raw_frame(train), train["price"], train["fold"].to_numpy()
    Xt, yt = prep.raw_frame(test), test["price"]

    model = build(summ["winner_params"], args.condition)
    oof = cross_val_predict(model, X, y, cv=PredefinedSplit(fold))
    model.fit(X, y)
    pred = model.predict(Xt)
    b_oof, b_test = benchmark_oof(train), benchmark_predict(train, test)

    rows = []
    for name, cvp, tp in [(f"XGBoost final (condition: {args.condition})", oof, pred),
                          ("Benchmark (median by model line + year)", b_oof, b_test)]:
        rows.append({"model": name, "cv_fold_mape_std": fold_mapes(y, cvp, fold).std(),
                     **{f"cv_{k}": v for k, v in metrics(y, cvp).items()},
                     **{f"test_{k}": v for k, v in metrics(yt, tp).items()}})
    final = pd.DataFrame(rows)
    final.to_csv(out / "final_model_v2.csv", index=False)
    print(final.round(3).T)

    tr_band = pd.cut(train.price, BANDS, labels=BAND_LABELS)
    te_band = pd.cut(test.price, BANDS, labels=BAND_LABELS)
    pd.concat({"cv_winner": by_group(train.assign(band=tr_band), oof, "band"),
               "cv_benchmark": by_group(train.assign(band=tr_band), b_oof, "band"),
               "test_winner": by_group(test.assign(band=te_band), pred, "band")},
              names=["set"]).reset_index().to_csv(out / "error_by_price_band_v2.csv", index=False)
    by_group(test, pred, "make").sort_values("n", ascending=False).to_csv(out / "error_by_brand_test_v2.csv")
    by_group(test.assign(age_band=prep.build_features(test)["age_band"]), pred, "age_band").to_csv(
        out / "error_by_age_band_test_v2.csv")
    worst = test.assign(predicted=pred, ape_pct=100 * np.abs(pred - test.price) / test.price)
    worst.nlargest(20, "ape_pct")[["listing_id", "make", "model_line", "trim_name", "year", "mileage",
                                   "accidents_or_damage", "price", "predicted", "ape_pct"]].to_csv(
        out / "largest_errors_test_v2.csv", index=False)
    pd.DataFrame({"listing_id": test.listing_id, "price": yt, "predicted": pred}).to_csv(
        out / "test_predictions_winner_v2.csv", index=False)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/price_model_v3.joblib")
    reg = register(args.mlflow_uri, final.iloc[0], summ["winner_params"], args.condition, model, X,
                   float(final.iloc[1]["cv_MAPE_pct"]))
    (out / "summary_final_v2.json").write_text(json.dumps(
        {"model_file": "models/price_model_v3.joblib", "family": "XGBoost", "condition": args.condition,
         "params": summ["winner_params"], "prep_version": prep.VERSION, "mlflow": reg}, indent=2, default=str))
    print(json.dumps(reg, indent=2))


def register(uri, row, params, condition, model, X, bench_mape):
    import mlflow
    from mlflow.models import infer_signature
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("ev_price_models")
    with mlflow.start_run(run_name=f"XGBoost final (condition: {condition})") as run:
        mlflow.log_params({"family": "XGBoost", "condition_features": condition,
                           "data": "model_dataset.csv (M3 extract)", "split": "train_ids_v1/test_ids_v1",
                           "folds": "fold_ids_v1", "prep_version": prep.VERSION,
                           "random_state": prep.RANDOM_STATE, "target": "log(price), scored in US$", **params})
        mlflow.log_metrics({k: float(v) for k, v in row.items() if k != "model" and pd.notna(v)})
        info = mlflow.sklearn.log_model(model, name="model", input_example=X.head(3), code_paths=["src"],
                                        signature=infer_signature(X.head(50), model.predict(X.head(50))))
        beats = str(row["cv_MAPE_pct"] < bench_mape).lower()
        mlflow.set_tags({"winner": "true", "beats_benchmark": beats})
    mv = mlflow.register_model(info.model_uri, "ev_price_model")
    client = mlflow.MlflowClient()
    client.set_registered_model_alias("ev_price_model", "champion", mv.version)
    client.set_model_version_tag("ev_price_model", mv.version, "beats_benchmark", beats)
    client.set_model_version_tag("ev_price_model", mv.version, "condition_features", condition)
    client.update_model_version("ev_price_model", mv.version, description=(
        f"XGBoost, tuned on 5-fold CV MAPE; condition features: {condition} (chosen on CV); "
        "raw fields in, US$ out."))
    return {"registered_model": "ev_price_model", "version": mv.version, "alias": "champion",
            "run_id": run.info.run_id, "benchmark_cv_mape": bench_mape}


if __name__ == "__main__":
    main()
