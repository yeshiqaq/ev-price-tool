"""M2 inference helper. Import with the package root on Python's search path."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from . import prep

class PriceCutScorer:
    def __init__(self,folder):
        folder=Path(folder)
        self.price_model=joblib.load(folder/'models/price_model_v3.joblib')
        self.cut_model=joblib.load(folder/'models/price_cut_model.joblib')
        self.metadata=json.loads((folder/'models/price_cut_metadata.json').read_text(encoding='utf-8'))
        self.reliability=pd.read_csv(folder/'data/clean/dim_reliability.csv')
        if self.reliability.duplicated(['make','model_line','year']).any():raise ValueError('Duplicate reliability keys')

    def predict(self,listings):
        df=pd.DataFrame(listings).rename(columns=prep.RENAME).copy()
        if df.columns.duplicated().any():raise ValueError('Provide each field once; do not mix alias and canonical names')
        required=['make','model_line','year','initial_price']
        missing=set(required)-set(df)
        if missing:raise ValueError('Missing required fields: '+str(sorted(missing)))
        if df[required].isna().any().any():raise ValueError('Make/model/year/asking price must be non-missing')
        asking=pd.to_numeric(df.initial_price,errors='raise').to_numpy(dtype=float)
        if not np.isfinite(asking).all() or (asking<=0).any():raise ValueError('Initial price must be finite and positive')
        df['year']=pd.to_numeric(df.year,errors='raise')
        if not np.isfinite(df.year).all() or (df.year%1!=0).any():raise ValueError('Year must be an integer')
        lookup=self.reliability[['make','model_line','year']+prep.COMPLAINT_RAW]
        df=df.drop(columns=prep.COMPLAINT_RAW,errors='ignore').merge(lookup,on=['make','model_line','year'],how='left',validate='many_to_one',sort=False,indicator='reliability_match')
        raw=prep.raw_frame(df)
        fair=np.asarray(self.price_model.predict(raw),dtype=float)
        if not np.isfinite(fair).all() or (fair<=0).any():raise ValueError('Price model returned invalid fair prices')
        # M5's training/validation price CSVs use whole-dollar predictions.
        fair=np.rint(fair)
        if (fair<=0).any():raise ValueError('Rounded fair price must be positive')
        x=raw.copy();x['ask_gap_pct']=(asking-fair)/fair
        score=self.cut_model.predict_proba(x)[:,1]
        return pd.DataFrame({'fair_price':fair,'ask_gap_pct':x.ask_gap_pct.to_numpy(),'recorded_cut_score':score,
            'review_price':score>=self.metadata['threshold'],'threshold':self.metadata['threshold'],
            'reliability_matched':df.reliability_match.eq('both').to_numpy()})
