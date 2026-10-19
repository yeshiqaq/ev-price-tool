"""M1-owned condition rules (copied unchanged from M1, 16 Oct; source: m1-inputs/v2_16oct/).
Do not edit here; ask M1 for a new version. Imported by src/prep.py (rules="m1")."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CONDITION_BINARY = ['low_mileage_for_age_flag', 'near_new_flag', 'clean_history_flag']
CONDITION_NUMERIC = ['condition_score', 'battery_wear_proxy']
CONDITION_CATEGORY = ['age_band', 'mileage_band']
CONDITION_FEATURES = CONDITION_BINARY + CONDITION_NUMERIC + CONDITION_CATEGORY

def add_condition_features(frame):
    out = frame.copy()
    age = pd.to_numeric(out['age'], errors='coerce')
    miles = pd.to_numeric(out['mileage'], errors='coerce')
    accident = pd.to_numeric(out['accidents_or_damage'], errors='coerce')
    valid = age.ge(0) & miles.ge(0) & age.notna() & miles.notna()
    annual = miles / age.clip(lower=0.5)
    out['low_mileage_for_age_flag'] = np.where(valid, annual.lt(7500).astype(float), np.nan)
    # Three-valued logic: known failure=0; known success=1; otherwise unknown.
    near = pd.Series(np.nan, index=out.index)
    near.loc[age.gt(1) | miles.ge(5000)] = 0.0
    near.loc[valid & age.le(1) & miles.lt(5000)] = 1.0
    out['near_new_flag'] = near
    out['clean_history_flag'] = accident.map({0.0: 1.0, 1.0: 0.0})
    # Do not reward missing history or incomplete condition information.
    out['condition_score'] = out[CONDITION_BINARY].sum(axis=1, min_count=3)
    out['age_band'] = pd.cut(age.where(age.ge(0)), [-np.inf,1,3,5,np.inf], labels=['0-1','2-3','4-5','6+']).astype(object)
    out['mileage_band'] = pd.cut(miles.where(miles.ge(0)), [0,10000,30000,60000,np.inf], right=False, labels=['0-<10k','10k-<30k','30k-<60k','60k+']).astype(object)
    # Cumulative use exposure in 10,000-mile units, NOT battery SOH or degradation %.
    out['battery_wear_proxy'] = miles.where(miles.ge(0)) / 10000.0
    return out

class ConditionFeatures(BaseEstimator, TransformerMixin):
    """Place before M5 preprocessing in an sklearn Pipeline; no fitted thresholds."""
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        return add_condition_features(X)
