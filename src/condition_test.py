"""With vs without condition features, following M1's 16 Oct protocol (owner: M5).

    python -m src.condition_test --data data/model/model_dataset.csv

Same split, same folds (fold_ids_v1), same model and the same tuned settings for every
variant; only the condition block changes. Confirmed error 13975 is out of training.
Reports MAPE (the lecturer's selection metric) and log-MAE (M1's), plus dollar MAE / RMSE / R²
from exp(predicted log price). The test set is not touched here.

Variants: none | M5's rules | M1's rules (all 7) | M1's rules minus one group at a time.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import PredefinedSplit, cross_val_predict
from xgboost import XGBRegressor

from src import prep
from src.train_price_models import load

GROUPS = {"flags": ["low_mileage_for_age_flag", "near_new_flag", "clean_history_flag"],
          "condition_score": ["condition_score"],
          "bands": ["age_band", "mileage_band"],
          "battery_wear_proxy": ["battery_wear_proxy"]}


def scores(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAPE_pct": 100 * np.mean(np.abs(p - y) / y),
            "logMAE": np.mean(np.abs(np.log(p) - np.log(y))),
            "MAE_usd": np.mean(np.abs(p - y)),
            "RMSE_usd": np.sqrt(np.mean((p - y) ** 2)),
            "R2": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/model/model_dataset.csv")
    ap.add_argument("--ids", default="data/model")
    ap.add_argument("--params", default="results/summary_v1.json")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    train, _ = load(args.data, Path(args.ids))
    X, y, fold = prep.raw_frame(train), train["price"], train["fold"].to_numpy()
    cv = PredefinedSplit(fold)
    tuned = json.loads(Path(args.params).read_text())["all_params"]
    models = {"Ridge": (lambda: Ridge(**tuned["Ridge"]), True),
              "XGBoost": (lambda: XGBRegressor(tree_method="hist", n_jobs=-1, random_state=prep.RANDOM_STATE,
                                               **tuned["XGBoost"]), False)}
    variants = {"none": dict(use_condition=False),
                "M5 rules": dict(rules="m5"),
                "M1 rules (all 7)": dict(rules="m1")}
    for g, cols in GROUPS.items():
        variants[f"M1 rules minus {g}"] = dict(rules="m1", drop=tuple(cols))

    rows, per_fold = [], {}
    for mname, (make_reg, scale) in models.items():
        for vname, kw in variants.items():
            p = cross_val_predict(prep.make_model(make_reg(), scale=scale, **kw), X, y, cv=cv)
            ape = pd.Series(np.abs(p - y) / y)
            lae = pd.Series(np.abs(np.log(p) - np.log(y)))
            per_fold[(mname, vname)] = pd.DataFrame({"MAPE": ape.groupby(fold).mean() * 100,
                                                     "logMAE": lae.groupby(fold).mean()})
            rows.append({"model": mname, "variant": vname, **scores(y, p)})
            print(f"{mname:8s} {vname:34s} MAPE {rows[-1]['MAPE_pct']:.3f}%  logMAE {rows[-1]['logMAE']:.4f}")

    res = pd.DataFrame(rows)
    for i, r in res.iterrows():
        # gain > 0 means the condition block (or, for "minus" rows, the dropped group) helps
        if r.variant.startswith("M1 rules minus"):
            ref = "M1 rules (all 7)"
            d = per_fold[(r.model, r.variant)] - per_fold[(r.model, ref)]
        else:
            ref = "none"
            d = per_fold[(r.model, ref)] - per_fold[(r.model, r.variant)]
        res.loc[i, "compared_with"] = ref if r.variant != "none" else ""
        if r.variant != "none":
            res.loc[i, "MAPE_gain_pts"] = d["MAPE"].mean()
            res.loc[i, "MAPE_gain_fold_sd"] = d["MAPE"].std()
            res.loc[i, "folds_helped_MAPE"] = int((d["MAPE"] > 0).sum())
            res.loc[i, "logMAE_gain"] = d["logMAE"].mean()
            res.loc[i, "folds_helped_logMAE"] = int((d["logMAE"] > 0).sum())
        res.loc[i, "fold_sd_MAPE"] = per_fold[(r.model, r.variant)]["MAPE"].std()
    res.to_csv(Path(args.out) / "condition_test_v1.csv", index=False)
    pd.concat(per_fold, names=["model", "variant", "fold"]).to_csv(Path(args.out) / "condition_test_folds_v1.csv")


if __name__ == "__main__":
    main()
