"""Shared data preparation for the EV fair-price project (owner: M5).

One recipe for training, the price-cut model and the API. Only M5 edits this file.

The saved model takes RAW listing fields and does everything else itself:

    raw fields --> build_features --> make_prep (impute, scale, one-hot) --> regressor
                   (derived inputs)

and it is wrapped so that it trains on log(price) but predicts and is scored in US$.
So:
  * the API passes the raw fields it collects (see RAW_INPUTS) and gets dollars back;
    there is no np.exp in the app;
  * training code passes the same raw columns, never M1's pre-computed derived columns.

Keep every custom function in this file. joblib stores a reference to
`src.prep.<name>`, so the Docker image must contain src/prep.py at that same path.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

VERSION = "v4"           # announce each new version in the group chat
SNAPSHOT_YEAR = 2023     # listings collected April 2023
RANDOM_STATE = 42
TARGET = "price"

PREMIUM = ["Tesla", "Porsche", "BMW", "Mercedes-Benz", "Audi", "Jaguar",
           "Polestar", "Lucid", "Rivian", "Genesis"]

# What the API collects (and what training passes in). Missing values are allowed
# for the ratings and history flags; the imputer fills them.
RAW_INPUTS = ["make", "model_line", "trim_name", "drivetrain", "year", "mileage",
              "accidents_or_damage", "one_owner", "personal_use_only",
              "seller_rating", "driver_rating"]

# Never model inputs: target, or derived from the target or the price cut.
NEVER_INPUTS = ["price", "log_price", "price_drop", "price_drop_raw", "price_drop_was_missing",
                "initial_price", "price_cut_flag", "iqr_outlier", "iso_outlier",
                "iso_anomaly_score", "confirmed_error_flag"]

NUM = ["age", "mileage", "miles_per_year", "accidents_or_damage", "one_owner",
       "personal_use_only", "seller_rating", "driver_rating", "premium_brand_flag"]
CONDITION = ["low_mileage_for_age_flag", "near_new_flag", "clean_history_flag",
             "condition_score", "battery_wear_proxy"]
CONDITION_CAT = ["mileage_band", "age_band"]
COMPLAINT_RAW = ["complaints_per_listing", "avg_sentiment", "battery_charging_share",
                 "braking_assist_share", "software_screen_share", "crash_fire_share"]
# complaints_per_listing is extreme for rare model-years (up to 33), so the model uses log1p of it.
COMPLAINT = ["log_complaints_per_listing", "avg_sentiment", "battery_charging_share",
             "braking_assist_share", "software_screen_share", "crash_fire_share"]
CAT = ["make", "model_line", "trim_name", "drivetrain"]

# M1's file and the SQL view use different names for the same fields.
RENAME = {"manufacturer": "make", "model": "trim_name", "model_year": "year"}


def build_features(X):
    """Derive every engineered input from the raw fields. Fixed rules only (no fitting),
    so it is safe to run on any row, at training or at inference."""
    df = pd.DataFrame(X).rename(columns=RENAME).copy()
    for col in RAW_INPUTS + COMPLAINT_RAW:
        if col not in df.columns:
            df[col] = np.nan
    for col in ["year", "mileage", "accidents_or_damage", "one_owner", "personal_use_only",
                "seller_rating", "driver_rating"] + COMPLAINT_RAW:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["log_complaints_per_listing"] = np.log1p(df["complaints_per_listing"])

    df["age"] = SNAPSHOT_YEAR - df["year"]
    df["miles_per_year"] = df["mileage"] / df["age"].clip(lower=0.5)
    df["premium_brand_flag"] = df["make"].isin(PREMIUM).astype(int)

    # Condition features (lecturer feedback 2). Unknown history counts as "not shown clean".
    acc = df["accidents_or_damage"].fillna(0)
    own = df["one_owner"].fillna(0)
    per = df["personal_use_only"].fillna(0)
    high = (df["miles_per_year"] > 15000).astype(int)
    df["low_mileage_for_age_flag"] = (df["miles_per_year"] < 5000).astype(int)
    df["near_new_flag"] = ((df["age"] <= 1) & (df["mileage"] < 1000)).astype(int)
    df["clean_history_flag"] = ((df["accidents_or_damage"] == 0) & (df["one_owner"] == 1)
                                & (df["personal_use_only"] == 1)).astype(int)
    df["condition_score"] = own + per - 2 * acc - high
    # age x miles_per_year would just equal mileage; this combines calendar ageing with use.
    df["battery_wear_proxy"] = df["age"] * df["mileage"] / 10000
    df["mileage_band"] = pd.cut(df["mileage"], [-1, 10000, 30000, 60000, 100000, np.inf],
                                labels=["<10k", "10-30k", "30-60k", "60-100k", "100k+"]).astype(str)
    df["age_band"] = pd.cut(df["age"], [-1, 1, 3, 6, np.inf],
                            labels=["0-1", "2-3", "4-6", "7+"]).astype(str)
    return df


def make_prep(use_complaints=True, use_condition=True, extra_num=(), scale=True):
    """Impute, scale and one-hot encode. Fitted inside each CV fold, so no leakage."""
    num = (NUM + (CONDITION if use_condition else []) + (COMPLAINT if use_complaints else [])
           + list(extra_num))
    cat = CAT + (CONDITION_CAT if use_condition else [])
    num_steps = [SimpleImputer(strategy="median", add_indicator=True)]
    if scale:
        num_steps.append(StandardScaler())
    return ColumnTransformer([
        ("num", make_pipeline(*num_steps), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=5)), cat),
    ])


def make_model(regressor, use_complaints=True, use_condition=True, extra_num=(), scale=True):
    """Full model: raw fields in, US$ out. Trains on log(price).
    Tune with parameter names like 'regressor__model__max_depth'."""
    pipe = Pipeline([
        ("features", FunctionTransformer(build_features)),
        ("prep", make_prep(use_complaints, use_condition, extra_num, scale)),
        ("model", regressor),
    ])
    return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp,
                                      check_inverse=False)


def raw_frame(df):
    """Select the raw input columns (plus complaint columns if present) from a listings table."""
    df = pd.DataFrame(df).rename(columns=RENAME)
    cols = [c for c in RAW_INPUTS + COMPLAINT_RAW if c in df.columns]
    return df[cols].copy()


def make_classifier(classifier, use_complaints=True, use_condition=True, extra_num=("ask_gap_pct",)):
    """Price-cut model (M2): same features and preparation, plus the gap between the
    planned asking price and M5's fair price. Raw fields + 'ask_gap_pct' in."""
    return Pipeline([
        ("features", FunctionTransformer(build_features)),
        ("prep", make_prep(use_complaints, use_condition, extra_num)),
        ("clf", classifier),
    ])


def add_reliability(df, rel):
    """Attach M4's complaint summary (dim_reliability.csv) by make + model_line + year.
    Used by training and by the API, so both look complaints up the same way.
    Model lines with no matched complaints stay blank; the imputer fills them and adds a flag."""
    df = pd.DataFrame(df).rename(columns=RENAME)
    rel = rel[["make", "model_line", "year"] + COMPLAINT_RAW]
    return df.drop(columns=[c for c in COMPLAINT_RAW if c in df.columns]).merge(
        rel, on=["make", "model_line", "year"], how="left")
