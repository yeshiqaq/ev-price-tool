"""Fair prices for M2's price-cut model, from the registered price model (owner: M5).

Run after train_price_models:
    python -m src.make_fair_prices --data data/model/model_dataset.csv

Writes (same columns as v1, so M2 only changes the file name):
  data/model/oof_fair_price_v2.csv     training cars: out-of-fold on fold_ids_v1;
                                       test cars: the model fitted on all training cars
  data/model/fair_price_nested_v2.csv  per outer fold k: classifier_train rows priced
                                       out-of-fold within the other 4 folds; classifier_valid
                                       rows priced by a model fitted on the other 4 folds only
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import PredefinedSplit, cross_val_predict

from src import prep
from src.train_price_models import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/model/model_dataset.csv")
    ap.add_argument("--ids", default="data/model")
    ap.add_argument("--model", default="models/price_model_v2.joblib")
    ap.add_argument("--version", default="v2")
    args = ap.parse_args()
    ids = Path(args.ids)

    train, test = load(args.data, ids)
    X, y, fold = prep.raw_frame(train), train["price"], train["fold"].to_numpy()
    fitted = joblib.load(args.model)          # already fitted on all training cars
    oof = cross_val_predict(clone(fitted), X, y, cv=PredefinedSplit(fold))
    ape = np.abs(oof - y) / y
    print(f"OOF MAPE {100 * ape.mean():.2f}%")

    flat = pd.concat([
        pd.DataFrame({"listing_id": train.listing_id, "split": "train", "fold": fold,
                      "fair_price": oof.round(0), "fair_price_source": "out_of_fold",
                      "prediction_origin": "oof_train"}),
        pd.DataFrame({"listing_id": test.listing_id, "split": "test", "fold": -1,
                      "fair_price": fitted.predict(prep.raw_frame(test)).round(0),
                      "fair_price_source": "model_fit_on_all_train",
                      "prediction_origin": "train_only_test"}),
    ])
    flat.to_csv(ids / f"oof_fair_price_{args.version}.csv", index=False)

    parts = []
    for k in sorted(np.unique(fold)):
        inner, outer = fold != k, fold == k
        inner_oof = cross_val_predict(clone(fitted), X[inner], y[inner], cv=PredefinedSplit(fold[inner]))
        m = clone(fitted).fit(X[inner], y[inner])
        valid = m.predict(X[outer])
        print(f"outer fold {k}: inner MAPE {100 * np.mean(np.abs(inner_oof - y[inner]) / y[inner]):.2f}%, "
              f"valid MAPE {100 * np.mean(np.abs(valid - y[outer]) / y[outer]):.2f}%")
        parts += [pd.DataFrame({"outer_fold": k, "listing_id": train.listing_id[inner],
                                "role": "classifier_train", "fair_price": inner_oof.round(0)}),
                  pd.DataFrame({"outer_fold": k, "listing_id": train.listing_id[outer],
                                "role": "classifier_valid", "fair_price": valid.round(0)})]
    pd.concat(parts).to_csv(ids / f"fair_price_nested_{args.version}.csv", index=False)


if __name__ == "__main__":
    main()
