"""Train, tune, compare and register the EV price models (owner: M5).

Run from the repo root:
    python -m src.train_price_models --data data/model/model_dataset.csv

What it does, in order:
  1. Loads M3's SQL extract, removes M1's confirmed error (id 13975) from training,
     and uses the shared split (train_ids_v1 / test_ids_v1) and folds (fold_ids_v1).
  2. Tunes every model family on 5-fold CV MAPE (RandomizedSearchCV, same folds for all).
     Every model trains on log(price) but is scored in US$ (see prep.make_model).
  3. Picks the winner on CV MAPE only, then scores the test set (all families are
     scored on test for the table, but the test set plays no part in the choice).
  4. Error by price band, brand and age band; 20 largest test errors.
  5. With vs without M1's condition features and M4's complaint features (paired folds).
  6. Q2 % effects from an unscaled OLS on log(price) (statsmodels, robust SEs).
  7. Logs every family to MLflow and registers the winner as `ev_price_model`.

Outputs go to results/ (csv/json), models/price_model_v2.joblib and mlflow/.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import loguniform, randint, uniform
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src import prep

RS = prep.RANDOM_STATE
CONFIRMED_ERRORS = [13975]          # M1 outlier_ids.csv: Toyota bZ4X at $333,333
BANDS = [0, 20000, 35000, 50000, 75000, np.inf]
BAND_LABELS = ["<$20k", "$20-35k", "$35-50k", "$50-75k", "$75k+"]
SCORING = "neg_mean_absolute_percentage_error"


# ---------------------------------------------------------------- data
def load(data_path, ids_dir):
    df = pd.read_csv(data_path, na_values=["NULL"]).rename(columns={"listing_key": "listing_id"})
    train_ids = pd.read_csv(ids_dir / "train_ids_v1.csv")["listing_id"]
    test_ids = pd.read_csv(ids_dir / "test_ids_v1.csv")["listing_id"]
    folds = pd.read_csv(ids_dir / "fold_ids_v1.csv")
    train = (df[df.listing_id.isin(train_ids) & ~df.listing_id.isin(CONFIRMED_ERRORS)]
             .merge(folds, on="listing_id", how="inner").reset_index(drop=True))
    test = df[df.listing_id.isin(test_ids)].reset_index(drop=True)
    assert len(train) == len(folds), "fold file and training set disagree"
    return train, test


# ---------------------------------------------------------------- metrics
def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ape = np.abs(p - y) / y
    return {
        "MAPE_pct": 100 * ape.mean(),
        "within_10pct_share": 100 * (ape <= 0.10).mean(),
        "MAE_usd": np.abs(p - y).mean(),
        "RMSE_usd": np.sqrt(((p - y) ** 2).mean()),
        "R2": 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
        "over_predicted_share": 100 * (p > y).mean(),
    }


def fold_mapes(y, p, fold):
    ape = np.abs(np.asarray(p) - np.asarray(y)) / np.asarray(y)
    return pd.Series(ape).groupby(np.asarray(fold)).mean() * 100


# ---------------------------------------------------------------- benchmark
def benchmark_predict(fit_df, pred_df):
    """Median price per model_line + year from fit_df; falls back to model_line, then overall."""
    by_my = fit_df.groupby(["model_line", "year"])["price"].median().rename("m1")
    by_m = fit_df.groupby("model_line")["price"].median().rename("m2")
    out = pred_df[["model_line", "year"]].merge(by_my, on=["model_line", "year"], how="left")
    out = out.merge(by_m, on="model_line", how="left")
    return out["m1"].fillna(out["m2"]).fillna(fit_df["price"].median()).to_numpy()


def benchmark_oof(train):
    p = np.empty(len(train))
    for k in sorted(train.fold.unique()):
        m = train.fold == k
        p[m] = benchmark_predict(train[~m], train[m])
    return p


# ---------------------------------------------------------------- model families
def families():
    """name -> (model, search space, n_iter). n_iter=None means no hyperparameters to tune."""
    p = "regressor__model__"
    mlp = TransformedTargetRegressor(       # standardise log(price) for the network
        regressor=MLPRegressor(early_stopping=True, max_iter=500, random_state=RS),
        transformer=StandardScaler())
    pca_ridge = make_pipeline(TruncatedSVD(random_state=RS), Ridge())
    return {
        "Linear regression": (prep.make_model(LinearRegression()), {}, None),
        "Ridge": (prep.make_model(Ridge()), {p + "alpha": loguniform(1e-3, 1e2)}, 25),
        "LASSO": (prep.make_model(Lasso(max_iter=20000)), {p + "alpha": loguniform(1e-6, 1e-2)}, 15),
        "PCA + Ridge": (prep.make_model(pca_ridge),
                        {p + "truncatedsvd__n_components": randint(20, 200),
                         p + "ridge__alpha": loguniform(1e-3, 1e2)}, 12),
        "Decision tree": (prep.make_model(DecisionTreeRegressor(random_state=RS), scale=False),
                          {p + "max_depth": randint(4, 30), p + "min_samples_leaf": randint(1, 40)}, 25),
        "Random forest": (prep.make_model(RandomForestRegressor(n_jobs=-1, random_state=RS), scale=False),
                          {p + "n_estimators": randint(200, 500), p + "max_features": [0.2, 0.33, 0.5],
                           p + "min_samples_leaf": randint(1, 5), p + "max_depth": [None, 20, 30]}, 8),
        "XGBoost": (prep.make_model(XGBRegressor(tree_method="hist", n_jobs=-1, random_state=RS), scale=False),
                    {p + "n_estimators": randint(300, 1500), p + "learning_rate": loguniform(0.02, 0.15),
                     p + "max_depth": randint(3, 9), p + "subsample": uniform(0.6, 0.4),
                     p + "colsample_bytree": uniform(0.4, 0.6), p + "min_child_weight": randint(1, 10),
                     p + "reg_lambda": loguniform(0.1, 10)}, 25),
        "Neural network (MLP)": (prep.make_model(mlp),
                                 {p + "regressor__hidden_layer_sizes": [(64,), (128,), (128, 64), (64, 32)],
                                  p + "regressor__alpha": loguniform(1e-5, 1e-1),
                                  p + "regressor__learning_rate_init": loguniform(5e-4, 5e-3)}, 10),
    }


def tune(name, model, space, n_iter, X, y, cv, quick=False):
    if quick and n_iter:                     # smoke test: 2 draws, small forests
        n_iter = 2
        space = {k: ([30] if k.endswith("n_estimators") else v) for k, v in space.items()}
    if n_iter is None:
        return model, {}
    search = RandomizedSearchCV(model, space, n_iter=n_iter, scoring=SCORING, cv=cv,
                                random_state=RS, n_jobs=1, refit=False)
    search.fit(X, y)
    best = search.best_params_
    model.set_params(**best)
    return model, {k.replace("regressor__model__", ""): v for k, v in best.items()}


# ---------------------------------------------------------------- Q2 effects
def q2_effects(train):
    """Unscaled OLS on log(price). % effect = exp(b) - 1, HC1 robust 95% CI (HC3 breaks on trims with one listing).
    Spec A (within trim): depreciation per year and per 10k miles, and the accident effect,
        comparing cars of the same trim.
    Spec B (brand): brand premiums vs Nissan, holding age, mileage and history fixed.
        Brand effects include model mix (a Porsche is a bigger, faster car), so they are
        'what the badge+range costs', not a pure badge effect."""
    d = prep.build_features(train)
    d["log_price"] = np.log(train["price"].to_numpy())
    d["mileage_10k"] = d["mileage"] / 10000
    for c in ["accidents_or_damage", "one_owner", "personal_use_only"]:
        d[c + "_missing"] = d[c].isna().astype(int)
        d[c] = d[c].fillna(0)
    d = d.dropna(subset=["mileage"])
    hist = ("accidents_or_damage + one_owner + personal_use_only + accidents_or_damage_missing"
            " + one_owner_missing + personal_use_only_missing")
    rows = []
    a = smf.ols(f"log_price ~ age + mileage_10k + {hist} + C(trim_name)", data=d).fit(cov_type="HC1")
    b = smf.ols(f"log_price ~ age + mileage_10k + {hist} + C(make, Treatment('Nissan'))",
                data=d).fit(cov_type="HC1")
    labels = {"age": "Each extra year of age", "mileage_10k": "Each extra 10,000 miles",
              "accidents_or_damage": "Accident or damage reported", "one_owner": "One owner",
              "personal_use_only": "Personal use only"}
    for spec, fit in [("A: within trim", a), ("B: brand", b)]:
        ci = fit.conf_int()
        for term in fit.params.index:
            if term in labels and spec.startswith("A"):
                label = labels[term]
            elif term.startswith("C(make") and spec.startswith("B"):
                label = "Brand: " + term.split("[T.")[1].rstrip("]") + " vs Nissan"
            else:
                continue
            rows.append({"spec": spec, "effect": label,
                         "pct_effect": 100 * (np.exp(fit.params[term]) - 1),
                         "ci_low_pct": 100 * (np.exp(ci.loc[term, 0]) - 1),
                         "ci_high_pct": 100 * (np.exp(ci.loc[term, 1]) - 1),
                         "p_value": fit.pvalues[term]})
    out = pd.DataFrame(rows)
    n_make = d["make"].value_counts()
    out["n_listings"] = out["effect"].map(
        lambda e: n_make.get(e.split(": ")[1].split(" vs")[0], np.nan) if e.startswith("Brand") else len(d))
    fit_info = {"spec_A_r2": a.rsquared, "spec_B_r2": b.rsquared, "n": int(a.nobs)}
    return out, fit_info


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/model/model_dataset.csv")
    ap.add_argument("--ids", default="data/model")
    ap.add_argument("--out", default="results")
    ap.add_argument("--mlflow-uri", default="sqlite:///mlflow/mlflow.db")
    ap.add_argument("--quick", action="store_true", help="smoke test with 2 draws per search")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    train, test = load(args.data, Path(args.ids))
    X, y, fold = prep.raw_frame(train), train["price"], train["fold"].to_numpy()
    Xt, yt = prep.raw_frame(test), test["price"]
    cv = PredefinedSplit(fold)
    print(f"train {len(train)}  test {len(test)}")

    # 1. benchmark + every family
    rows, oof, test_pred, fitted, params = [], {}, {}, {}, {}
    oof["Benchmark (median by model line + year)"] = benchmark_oof(train)
    test_pred["Benchmark (median by model line + year)"] = benchmark_predict(train, test)
    for name, (model, space, n_iter) in families().items():
        t0 = time.time()
        model, best = tune(name, model, space, n_iter, X, y, cv, args.quick)
        oof[name] = cross_val_predict(model, X, y, cv=cv)
        model.fit(X, y)
        test_pred[name] = model.predict(Xt)
        fitted[name], params[name] = model, best
        print(f"{name:28s} CV MAPE {metrics(y, oof[name])['MAPE_pct']:.2f}%  ({time.time()-t0:.0f}s)")

    for name in oof:
        cvm, tm = metrics(y, oof[name]), metrics(yt, test_pred[name])
        fm = fold_mapes(y, oof[name], fold)
        rows.append({"model": name, "cv_fold_mape_std": fm.std(),
                     **{f"cv_{k}": v for k, v in cvm.items()}, **{f"test_{k}": v for k, v in tm.items()}})
    table = pd.DataFrame(rows).sort_values("cv_MAPE_pct").reset_index(drop=True)
    table.to_csv(out / "model_comparison_v1.csv", index=False)
    winner = table.loc[~table.model.str.startswith("Benchmark"), "model"].iloc[0]
    print("winner:", winner)

    # 2. error by price band / brand / age band (winner: CV on train, and test)
    def by_group(frame, pred, col):
        g = frame.assign(pred=pred)
        return g.groupby(col, observed=True).apply(
            lambda s: pd.Series({"n": len(s), **metrics(s["price"], s["pred"])}), include_groups=False)
    bench = "Benchmark (median by model line + year)"
    tr_band = pd.cut(train.price, BANDS, labels=BAND_LABELS)
    te_band = pd.cut(test.price, BANDS, labels=BAND_LABELS)
    band = pd.concat({
        "cv_winner": by_group(train.assign(band=tr_band), oof[winner], "band"),
        "cv_benchmark": by_group(train.assign(band=tr_band), oof[bench], "band"),
        "test_winner": by_group(test.assign(band=te_band), test_pred[winner], "band"),
    }, names=["set"]).reset_index()
    band.to_csv(out / "error_by_price_band_v1.csv", index=False)
    feats_t = prep.build_features(test)
    by_group(test, test_pred[winner], "make").sort_values("n", ascending=False).to_csv(
        out / "error_by_brand_test_v1.csv")
    by_group(test.assign(age_band=feats_t["age_band"]), test_pred[winner], "age_band").to_csv(
        out / "error_by_age_band_test_v1.csv")
    worst = test.assign(predicted=test_pred[winner],
                        ape_pct=100 * np.abs(test_pred[winner] - test.price) / test.price)
    worst.nlargest(20, "ape_pct")[["listing_id", "make", "model_line", "trim_name", "year", "mileage",
                                   "accidents_or_damage", "price", "predicted", "ape_pct"]].to_csv(
        out / "largest_errors_test_v1.csv", index=False)
    pd.DataFrame({"listing_id": test.listing_id, "price": yt, "predicted": test_pred[winner]}).to_csv(
        out / "test_predictions_winner_v1.csv", index=False)

    # 3. with vs without condition / complaint features (tuned hyperparameters held fixed)
    abl_rows = []
    variants = {"full": (True, True), "no condition": (True, False),
                "no complaints": (False, True), "neither": (False, False)}
    abl_models = list(dict.fromkeys(["Ridge", "XGBoost", winner]))
    fam = families()
    for name in abl_models:
        base_model = fam[name][0]
        reg = base_model.regressor.named_steps["model"]
        reg.set_params(**params[name])
        scale = name not in ("Decision tree", "Random forest", "XGBoost")
        per_fold = {}
        for v, (use_comp, use_cond) in variants.items():
            m = prep.make_model(reg, use_complaints=use_comp, use_condition=use_cond, scale=scale)
            p = cross_val_predict(m, X, y, cv=cv)
            m.fit(X, y)
            per_fold[v] = fold_mapes(y, p, fold)
            abl_rows.append({"model": name, "variant": v, "cv_MAPE_pct": metrics(y, p)["MAPE_pct"],
                             "test_MAPE_pct": metrics(yt, m.predict(Xt))["MAPE_pct"],
                             **{f"fold{k}_MAPE": val for k, val in per_fold[v].items()}})
        for v, ref in [("no condition", "condition"), ("no complaints", "complaints")]:
            d = per_fold[v] - per_fold["full"]       # >0 means the feature block helps
            abl_rows.append({"model": name, "variant": f"gain from {ref} (pts)",
                             "cv_MAPE_pct": d.mean(), "test_MAPE_pct": np.nan,
                             "folds_improved": int((d > 0).sum()), "gain_std": d.std()})
    pd.DataFrame(abl_rows).to_csv(out / "feature_ablation_v1.csv", index=False)

    # 4. Q2
    q2, q2_info = q2_effects(train)
    q2.to_csv(out / "q2_pct_effects_v1.csv", index=False)

    # 5. save + MLflow
    joblib.dump(fitted[winner], "models/price_model_v2.joblib")
    reg_info = register(args.mlflow_uri, table, params, winner, fitted[winner], X)
    summary = {"winner": winner, "winner_params": params[winner], "all_params": params,
               "q2_fit": q2_info, "mlflow": reg_info, "n_train": len(train), "n_test": len(test),
               "prep_version": prep.VERSION}
    (out / "summary_v1.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(reg_info, indent=2))


def register(uri, table, params, winner, model, X):
    import mlflow
    from mlflow.models import infer_signature
    Path("mlflow").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name("ev_price_models") is None:
        mlflow.create_experiment("ev_price_models", artifact_location=Path("mlflow/artifacts").resolve().as_uri())
    mlflow.set_experiment("ev_price_models")
    bench_mape = table.loc[table.model.str.startswith("Benchmark"), "cv_MAPE_pct"].iloc[0]
    winner_run = model_uri = None
    for _, r in table.iterrows():
        with mlflow.start_run(run_name=r["model"]) as run:
            mlflow.log_params({"family": r["model"], "data": "model_dataset.csv (M3 extract)",
                               "split": "train_ids_v1/test_ids_v1", "folds": "fold_ids_v1",
                               "prep_version": prep.VERSION, "random_state": RS,
                               "target": "log(price), scored in US$",
                               **{k: v for k, v in params.get(r["model"], {}).items()}})
            mlflow.log_metrics({k: float(v) for k, v in r.items()
                                if k != "model" and pd.notna(v)})
            if r["model"] == winner:
                sig = infer_signature(X.head(50), model.predict(X.head(50)))
                info = mlflow.sklearn.log_model(model, name="model", signature=sig,
                                                input_example=X.head(3), code_paths=["src"])
                model_uri = info.model_uri
                mlflow.set_tags({"winner": "true",
                                 "beats_benchmark": str(r["cv_MAPE_pct"] < bench_mape).lower()})
                winner_run = run.info.run_id
    mv = mlflow.register_model(model_uri, "ev_price_model")
    client = mlflow.MlflowClient()
    client.set_registered_model_alias("ev_price_model", "champion", mv.version)
    client.set_model_version_tag("ev_price_model", mv.version, "beats_benchmark",
                                 str(table.loc[table.model == winner, "cv_MAPE_pct"].iloc[0] < bench_mape).lower())
    client.update_model_version("ev_price_model", mv.version,
                                description=f"{winner}, tuned on 5-fold CV MAPE; raw fields in, US$ out.")
    return {"tracking_uri": uri, "experiment": "ev_price_models", "registered_model": "ev_price_model",
            "version": mv.version, "alias": "champion", "run_id": winner_run,
            "benchmark_cv_mape": bench_mape}


if __name__ == "__main__":
    main()
