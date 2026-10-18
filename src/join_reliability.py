"""Validate M4's dimension and join without changing missing values or listing order."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ['make', 'model_line', 'year']
SHARES = ['battery_charging_share', 'braking_assist_share', 'software_screen_share', 'crash_fire_share', 'recall_share']
NUMERIC = ['n_complaints', 'complaints_per_listing', 'avg_sentiment'] + SHARES

def join_reliability(listings, dim):
    dim = dim.copy()
    required = KEYS + NUMERIC + ['n_listings','summary_level','complaint_status']
    if set(required) - set(dim):
        raise ValueError('Missing dimension columns: ' + str(set(required)-set(dim)))
    if dim[KEYS].isna().any().any() or dim.duplicated(KEYS).any():
        raise ValueError('Dimension keys must be complete and unique')
    for col in NUMERIC + ['n_listings']:
        dim[col] = pd.to_numeric(dim[col], errors='raise')
        if np.isinf(dim[col].dropna()).any(): raise ValueError('Infinite value: '+col)
    for col in SHARES:
        if not dim[col].dropna().between(0,1).all(): raise ValueError('Invalid share: '+col)
    if not dim.avg_sentiment.dropna().between(-1,1).all(): raise ValueError('Invalid sentiment')
    if not dim[['n_complaints','n_listings']].notna().all().all(): raise ValueError('Missing counts')
    if (dim.n_complaints.lt(0) | dim.n_listings.le(0)).any(): raise ValueError('Invalid counts')
    if ((dim.n_complaints % 1 != 0) | (dim.n_listings % 1 != 0)).any(): raise ValueError('Non-integer counts')
    if not np.allclose(dim.complaints_per_listing, dim.n_complaints/dim.n_listings,atol=0.000051):
        raise ValueError('Complaint ratio disagrees with counts')
    totals=dim.groupby(['make','model_line']).n_complaints.transform('sum')
    expected=np.select([dim.n_complaints.ge(10),dim.n_complaints.gt(0),totals.gt(0)],
        ['own_figures','model_line_fallback','no_complaints_this_year_line_fallback'],default='no_complaints_for_model_line')
    if not (dim.complaint_status==expected).all(): raise ValueError('Inconsistent complaint_status')
    level=np.where(dim.n_complaints.ge(10),'model_year',np.where(dim.n_complaints.gt(0),'model_line','none'))
    if not (dim.summary_level==level).all(): raise ValueError('Inconsistent summary_level')
    blank=dim.complaint_status.eq('no_complaints_for_model_line')
    if not dim.loc[blank,['avg_sentiment']+SHARES].isna().all().all(): raise ValueError('No-complaint blanks changed')
    if dim.loc[~blank,['avg_sentiment']+SHARES].isna().any().any(): raise ValueError('Unexpected missing features')
    data=listings.copy()
    brand='manufacturer' if 'manufacturer' in data else 'make'
    left=[brand,'model_line','year']
    if data[left].isna().any().any(): raise ValueError('Incomplete listing keys')
    collisions=(set(dim.columns)-set(KEYS)) & set(data.columns)
    if collisions: raise ValueError('Already joined or conflicting columns: '+str(collisions))
    merged=data.merge(dim,left_on=left,right_on=KEYS,how='left',validate='many_to_one',sort=False,indicator='reliability_match')
    if len(merged)!=len(data): raise ValueError('Listing row count changed')
    idcol='listing_id' if 'listing_id' in data else 'listing_key'
    if not data[idcol].is_unique or not merged[idcol].equals(data[idcol]): raise ValueError('IDs/order changed')
    matched=merged.reliability_match.eq('both')
    # Record unmatched separately; it does not mean no complaints.
    merged['reliability_match']=merged.reliability_match.astype(str)
    actual=data.groupby(left).size().rename('actual_n_listings').reset_index().rename(columns={brand:'make'})
    countcheck=dim.merge(actual,on=KEYS,how='left',validate='one_to_one')
    report={'listing_rows':len(data),'dimension_rows':len(dim),'dimension_model_lines':int(dim.model_line.nunique()),
        'matched_listings':int(matched.sum()),'unmatched_listings':int((~matched).sum()),'match_rate':float(matched.mean()),
        'duplicate_dimension_keys':0,'listing_ids_and_order_preserved':True,
        'dimension_no_complaint_rows':int(blank.sum()),'no_complaint_model_lines':dim.loc[blank,'model_line'].unique().tolist(),
        'listing_counts_disagree':int((countcheck.n_listings!=countcheck.actual_n_listings).sum()),
        'status_listing_counts':merged.complaint_status.fillna('unmatched').value_counts().to_dict(),
        'missing_feature_counts':merged[NUMERIC].isna().sum().to_dict(),
        'max_complaints_per_listing':float(dim.complaints_per_listing.max())}
    return merged,report

def main():
    p=argparse.ArgumentParser();p.add_argument('--listings',type=Path,required=True);p.add_argument('--reliability',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();merged,report=join_reliability(pd.read_csv(a.listings),pd.read_csv(a.reliability))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.report.parent.mkdir(parents=True,exist_ok=True)
    merged.to_csv(a.output,index=False);a.report.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__': main()
