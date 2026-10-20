
"""M2 price-cut inference helper, integrated with M5 prep.py v6."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from . import prep


class PriceCutScorer:

    def __init__(self, folder):
        folder = Path(folder)

        self.models_dir = folder / "models"

        # M5 Fair Price Model
        self.price_model = joblib.load(
            self.models_dir / "price_model_v3.joblib"
        )

        # M2 Price-Cut Model
        self.cut_model = joblib.load(
            self.models_dir / "price_cut_model.joblib"
        )

        # M2 threshold and metadata
        self.metadata = json.loads(
            (
                self.models_dir / "price_cut_metadata.json"
            ).read_text(encoding="utf-8")
        )

        # Reliability data for validation
        self.reliability = pd.read_csv(
            self.models_dir / "reliability_lookup_v1.csv"
        )

        keys = ["make", "model_line", "year"]

        if self.reliability.duplicated(keys).any():
            raise ValueError(
                "Duplicate reliability keys"
            )

    def predict(self, listings):

        # ----------------------------------
        # 1. Validate inputs
        # ----------------------------------
        df = pd.DataFrame(listings).rename(
            columns=prep.RENAME
        ).copy()

        if df.columns.duplicated().any():
            raise ValueError(
                "Provide each field once; "
                "do not mix alias and canonical names"
            )

        required = [
            "make",
            "model_line",
            "year",
            "initial_price"
        ]

        missing = set(required) - set(df.columns)

        if missing:
            raise ValueError(
                "Missing required fields: "
                + str(sorted(missing))
            )

        if df[required].isna().any().any():
            raise ValueError(
                "Make/model/year/asking price "
                "must be non-missing"
            )

        asking = pd.to_numeric(
            df["initial_price"],
            errors="raise"
        ).to_numpy(dtype=float)

        if (
            not np.isfinite(asking).all()
            or (asking <= 0).any()
        ):
            raise ValueError(
                "Initial price must be finite and positive"
            )

        df["year"] = pd.to_numeric(
            df["year"],
            errors="raise"
        )

        if (
            not np.isfinite(df["year"]).all()
            or (df["year"] % 1 != 0).any()
        ):
            raise ValueError(
                "Year must be an integer"
            )

        # ----------------------------------
        # 2. Reliability match check
        # ----------------------------------
        keys = ["make", "model_line", "year"]

        known_keys = self.reliability[
            keys
        ].drop_duplicates()

        match_result = df[keys].merge(
            known_keys.assign(_matched=True),
            on=keys,
            how="left",
            sort=False,
            validate="many_to_one"
        )

        reliability_matched = (
            match_result["_matched"]
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )

        # ----------------------------------
        # 3. M5 v6 lookup preparation
        # ----------------------------------
        # Adds missing driver rating and complaint features
        enriched = prep.add_lookups(
            df,
            models_dir=self.models_dir
        )

        raw = prep.raw_frame(enriched)

        # ----------------------------------
        # 4. M5 Fair Price Prediction
        # ----------------------------------
        fair = np.asarray(
            self.price_model.predict(raw),
            dtype=float
        )

        if (
            not np.isfinite(fair).all()
            or (fair <= 0).any()
        ):
            raise ValueError(
                "Price model returned invalid fair prices"
            )

        # Keep M2's original whole-dollar rounding
        fair = np.rint(fair)

        if (fair <= 0).any():
            raise ValueError(
                "Rounded fair price must be positive"
            )

        # ----------------------------------
        # 5. Asking Price Gap
        # ----------------------------------
        x = raw.copy()

        x["ask_gap_pct"] = (
            asking - fair
        ) / fair

        # ----------------------------------
        # 6. M2 Price-Cut Review Score
        # ----------------------------------
        score = self.cut_model.predict_proba(x)[:, 1]

        threshold = float(
            self.metadata["threshold"]
        )

        review_flag = score >= threshold

        # ----------------------------------
        # 7. Return results
        # ----------------------------------
        return pd.DataFrame({
            "fair_price": fair,
            "ask_gap_pct": x["ask_gap_pct"].to_numpy(),
            "recorded_cut_score": score,
            "review_price": review_flag,
            "threshold": threshold,
            "reliability_matched": reliability_matched
        })
