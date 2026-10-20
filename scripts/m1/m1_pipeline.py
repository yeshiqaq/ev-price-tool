"""M1 reproducible cleaning, diagnostics and preprocessing comparison.
No predictive performance is reported on the held-out 20% test set.
"""
from pathlib import Path
import json, re, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import IsolationForest

NUMERIC = ['age', 'mileage', 'seller_rating', 'driver_rating']
BINARY = ['accidents_or_damage', 'one_owner', 'personal_use_only']
PREMIUM = ['Tesla','Porsche','BMW','Mercedes-Benz','Audi','Jaguar','Polestar','Lucid','Rivian','Genesis']
SEED = 42

def directories(root):
    root = Path(root)
    for d in ['data/raw','data/clean','data/diagnostics','figures','reports','notebooks','scripts']:
        (root/d).mkdir(parents=True, exist_ok=True)
    return root

def canonical_model(make, model):
    """Explicit model families; original trim text remains in model.
    The review flag identifies generic labels inferred from fuel/engine evidence.
    """
    u = str(model).upper()
    review = False
    if make == 'Tesla':
        match = re.match(r'MODEL\s+([3SXY])\b',u)
        line = 'Model '+match.group(1) if match else ('Roadster' if u.startswith('ROADSTER') else None)
    elif make == 'Audi':
        line = 'Q4 e-tron' if u.startswith('Q4') else ('e-tron GT' if 'GT' in u else 'e-tron')
    elif make == 'BMW': line = str(model).split()[0]
    elif make == 'Cadillac': line = 'LYRIQ' if u.startswith('LYRIQ') else None
    elif make == 'Chevrolet':
        line = 'Bolt EUV' if u.startswith('BOLT EUV') else ('Bolt EV' if u.startswith('BOLT EV') else ('Spark EV' if u.startswith('SPARK EV') else None))
    elif make == 'Ford':
        if u.startswith('MUSTANG MACH-E'): line = 'Mustang Mach-E'
        elif u.startswith('FOCUS'): line,review = 'Focus Electric', 'ELECTRIC' not in u
        elif u.startswith('F-150'): line,review = 'F-150 Lightning', 'LIGHTNING' not in u
        elif 'TRANSIT' in u: line,review = 'E-Transit', not u.startswith('E-TRANSIT')
        else: line = None
    elif make == 'GMC': line = 'HUMMER EV'
    elif make == 'Hyundai':
        line = 'IONIQ 5' if u.startswith('IONIQ 5') else ('IONIQ EV' if u.startswith('IONIQ EV') else ('Kona EV' if u.startswith('KONA EV') else None))
    elif make == 'Jaguar': line = 'I-PACE'
    elif make == 'Kia':
        line = 'Soul EV' if u.startswith('SOUL EV') else ('Niro EV' if u.startswith('NIRO') else ('EV6' if u.startswith('EV6') else None))
        review = u.startswith('NIRO') and not u.startswith('NIRO EV')
    elif make == 'Mazda': line = 'MX-30'
    elif make == 'Mercedes-Benz':
        line = 'B-Class Electric' if u.startswith('B-CLASS') else next((s for s in ['EQB','EQE','EQS'] if s in u),None)
    elif make == 'Mitsubishi': line = 'i-MiEV'
    elif make == 'Nissan': line = 'Leaf' if u.startswith('LEAF') else ('ARIYA' if u.startswith('ARIYA') else None)
    elif make == 'Porsche': line = 'Taycan'
    elif make == 'Subaru': line = 'Solterra'
    elif make == 'Toyota': line = 'RAV4 EV' if u.startswith('RAV4 EV') else ('bZ4X' if u.startswith('BZ4X') else None)
    elif make == 'Volkswagen': line = 'ID.4' if u.startswith('ID.4') else ('e-Golf' if u.startswith('E-GOLF') else None)
    elif make == 'Volvo':
        line = 'XC40 Recharge' if u.startswith('XC40') else ('C40 Recharge' if u.startswith('C40') else None)
        review = u.startswith('XC40') and 'RECHARGE' not in u
    else: line = None
    return (make+' '+line if line else make+' '+str(model), bool(review or line is None))

def clean_listings(raw, profile, root):
    root = directories(root)
    quality = []
    def log(step,before,after,reason):
        quality.append(dict(step=step,rows_before=int(before),rows_removed=int(before-after),rows_after=int(after),reason=reason))
    log('01 Electric filter',profile['total_rows'],len(raw),'Keep fuel_type Electric; extraction from supplied cars.xlsx')
    mask = raw.model.str.contains(r'Volt|Range Extender|ELR|Mirai|NEXO',case=False,na=False,regex=True)
    rejected = raw.loc[mask].copy()
    rejected['exclusion_reason'] = 'Explicit project non-BEV model rule: PHEV/range extender/hydrogen'
    rejected.to_csv(root/'data/diagnostics/non_bev_exclusions.csv',index=False)
    ev = raw.loc[~mask].copy()
    log('02 Non-BEV exclusions',len(raw),len(ev),'Explicit project non-BEV model-name rule; retain removal evidence')
    orig_cols = [c for c in raw.columns if c != 'source_excel_row']
    dup = ev.duplicated(subset=orig_cols,keep='first')
    ev.loc[dup].to_csv(root/'data/diagnostics/duplicate_exclusions.csv',index=False)
    n=len(ev); ev=ev.loc[~dup].reset_index(drop=True)
    log('03 Exact duplicates',n,len(ev),'Exact original-field duplicates; provenance column excluded from duplicate key')
    # Assign IDs once after cohort definition, before flags or train/test splitting.
    ev.insert(0,'listing_id',np.arange(len(ev),dtype=int))
    for c in orig_cols:
        if ev[c].dtype == object:
            ev[c]=ev[c].map(lambda v: re.sub(r'\s+',' ',v).strip() if isinstance(v,str) else v)
    ev=ev.drop(columns=['mpg','fuel_type'])
    drive = {'awd':'All-wheel Drive','all-wheel drive':'All-wheel Drive','rwd':'Rear-wheel Drive','rear-wheel drive':'Rear-wheel Drive','fwd':'Front-wheel Drive','front-wheel drive':'Front-wheel Drive','4wd':'Four-wheel Drive','four-wheel drive':'Four-wheel Drive','unknown':np.nan}
    ev['drivetrain']=ev.drivetrain.map(lambda v: drive.get(str(v).casefold(),v) if pd.notna(v) else np.nan)
    for c in ['year','mileage','seller_rating','driver_rating','driver_reviews_num','price','price_drop']+BINARY:
        ev[c]=pd.to_numeric(ev[c],errors='coerce')
    for c in BINARY:
        bad=ev[c].notna() & ~ev[c].isin([0,1])
        if bad.any(): raise ValueError('Unexpected binary values in '+c)
    for c in ['seller_rating','driver_rating']:
        bad=ev[c].notna() & ~ev[c].between(1,5)
        if bad.any(): raise ValueError('Unexpected rating scale in '+c)
    if (ev.price.isna() | (ev.price<=0)).any(): raise ValueError('Invalid price target requires separate review, never impute target')
    if (ev.year.isna() | (ev.year>2023)).any(): raise ValueError('Invalid year requires separate review')
    if (ev.mileage.dropna()<0).any(): raise ValueError('Negative mileage requires review')
    pairs=[canonical_model(m,x) for m,x in zip(ev.manufacturer,ev.model)]
    ev['model_line']=[x[0] for x in pairs]
    ev['model_line_review_flag']=[int(x[1]) for x in pairs]
    ev['age']=2023-ev.year
    ev['miles_per_year']=ev.mileage/ev.age.clip(lower=0.5)
    ev['high_mileage_flag']=np.where(ev.miles_per_year.isna(),np.nan,(ev.miles_per_year>15000).astype(int))
    ev['premium_brand_flag']=ev.manufacturer.isin(PREMIUM).astype(int)
    ev['price_drop_raw']=ev.price_drop.copy()
    ev['price_drop_was_missing']=ev.price_drop.isna().astype(int)
    ev['price_drop']=ev.price_drop.fillna(0) # explicit project assumption, not verified fact
    ev['initial_price']=ev.price+ev.price_drop # reconstructed, not original new-car price
    ev['price_cut_flag']=(ev.price_drop>0).astype(int)
    ev['log_price']=np.log(ev.price)
    ev['very_low_mileage_flag']=np.where(ev.mileage.isna(),np.nan,(ev.mileage<100).astype(int))
    from condition_features import add_condition_features
    ev=add_condition_features(ev)
    # Keep source uncertainties visible. This does not establish non-BEV status.
    generic=ev.model_line_review_flag.eq(1)
    evidence=ev.engine.fillna('').astype(str).str.contains(r'electric|\bac\b',case=False,regex=True,na=False)
    rex_engine=(ev.manufacturer.eq('BMW') & ev.model_line.eq('BMW i3') & ev.engine.fillna('').astype(str).str.contains(r'0\.65\s*L|647\s*cc',case=False,regex=True))
    ev['bev_status_review_flag']=((generic & ~evidence) | rex_engine).astype(int)
    ev['bev_status_review_reason']=np.where(rex_engine,'BMW i3 engine lists 0.65L: possible range extender; confirm source',np.where(generic & ~evidence,'Generic model label and missing/zero engine evidence','not_flagged'))
    log('04 Category and feature preparation',len(ev),len(ev),'Standardise drivetrains and model families; preserve gaps; preserve all price outliers')
    pd.DataFrame(quality).to_csv(root/'data/diagnostics/data_quality_log.csv',index=False)
    mapping=ev.groupby(['manufacturer','model','model_line','model_line_review_flag'],dropna=False).agg(listing_count=('listing_id','size'),engine_examples=('engine',lambda s:' | '.join(sorted(s.dropna().astype(str).unique())[:3]))).reset_index()
    mapping.to_csv(root/'data/clean/model_name_mapping_for_M2.csv',index=False)
    ev.groupby(['manufacturer','model_line']).agg(listing_count=('listing_id','size'),first_model_year=('year','min'),last_model_year=('year','max'),review_count=('model_line_review_flag','sum')).reset_index().to_csv(root/'data/clean/model_lines_for_M2.csv',index=False)
    ev[ev.bev_status_review_flag.eq(1)].to_csv(root/'data/diagnostics/bev_status_review.csv',index=False)
    ev.attrs['quality']=quality
    print('Clean cohort:',len(ev),'brands:',ev.manufacturer.nunique(),'model families:',ev.model_line.nunique(),flush=True)
    return ev

def missingness(ev,root):
    root=Path(root)
    reasons={
      'mileage':('MAR/MNAR hypotheses','Very few gaps; missing odometer information may depend on listing process. Mechanism cannot be determined from these data.'),
      'seller_rating':('MAR/MNAR hypotheses','Rates differ across observed seller/brand groups; seller size is not measured, so small sellers cannot be asserted as the cause.'),
      'driver_rating':('MAR/MNAR hypotheses','Inspect driver_reviews_num and model-year; no reviews and new models may lack ratings.'),
      'accidents_or_damage':('MAR/MNAR hypotheses','History unavailable/not disclosed are possible; missing is not equivalent to no accident.'),
      'one_owner':('MAR/MNAR hypotheses','History reporting may depend on seller/listing; unknown is not equivalent to multiple owners.'),
      'personal_use_only':('MAR/MNAR hypotheses','History reporting may depend on seller/listing; unknown use history must remain missing.'),
      'price_drop_raw':('Structural absence or MAR/MNAR; unverified','Project assumes absent recorded reduction = zero. Original values and missing indicator retained; no causal or future-price-cut claim.'),
      'engine':('MAR/MNAR hypotheses','Specifications not supplied or not extracted; no evidence to distinguish mechanisms.'),
      'drivetrain':('MAR/MNAR hypotheses','Not supplied, unknown, or not extracted; evaluate by model/seller; preserve missing.'),
      'exterior_color':('MAR/MNAR hypotheses','Optional seller description or extraction gap; mechanism unconfirmed.'),
      'interior_color':('MAR/MNAR hypotheses','Optional seller description or extraction gap; mechanism unconfirmed.'),
    }
    rows=[]
    for c in ev.columns:
        mechanism,why=reasons.get(c,('No missing values' if not ev[c].isna().any() else 'Undetermined','No filling at handover; downstream preprocessing fits only on training data.'))
        rows.append(dict(field=c,missing_count=int(ev[c].isna().sum()),missing_pct=100*ev[c].isna().mean(),mechanism_hypothesis=mechanism,explanation=why))
    result=pd.DataFrame(rows).sort_values('missing_pct',ascending=False)
    result.to_csv(root/'data/diagnostics/missingness_profile.csv',index=False)
    for group in ['manufacturer','seller_name']:
        rates=ev.assign(seller_rating_missing=ev.seller_rating.isna().astype(int),driver_rating_missing=ev.driver_rating.isna().astype(int)).groupby(group,dropna=False).agg(listings=('listing_id','size'),seller_rating_missing_pct=('seller_rating_missing',lambda s:s.mean()*100),driver_rating_missing_pct=('driver_rating_missing',lambda s:s.mean()*100)).reset_index()
        rates.to_csv(root/f'data/diagnostics/missingness_by_{group}.csv',index=False)
    ev.groupby('driver_reviews_num').agg(listings=('listing_id','size'),driver_rating_missing=('driver_rating',lambda s:s.isna().sum())).reset_index().to_csv(root/'data/diagnostics/rating_missing_vs_reviews.csv',index=False)
    return result

def mark_outliers(ev,root):
    root=Path(root)
    a=ev.groupby(['model_line','year']).log_price.agg(['size',lambda s:s.quantile(.25),lambda s:s.quantile(.75)])
    a.columns=['group_n','q1','q3']
    b=ev.groupby('model_line').log_price.agg(['size',lambda s:s.quantile(.25),lambda s:s.quantile(.75)])
    b.columns=['model_n','q1_model','q3_model']
    bounds=ev[['model_line','year']].join(a,on=['model_line','year']).join(b,on='model_line')
    small=bounds.group_n<10
    q1=bounds.q1.where(~small,bounds.q1_model); q3=bounds.q3.where(~small,bounds.q3_model)
    iqr=q3-q1
    ev['iqr_reference_n']=bounds.group_n.where(~small,bounds.model_n).astype(int)
    ev['iqr_reference_scope']=np.where(small,'whole_model_line','model_line_and_year')
    ev['iqr_outlier']=((ev.log_price<q1-1.5*iqr)|(ev.log_price>q3+1.5*iqr)).astype(int)
    # Descriptive full-cohort diagnostics only. NEVER supply these target-derived flags to a price model.
    x=SimpleImputer(strategy='median').fit_transform(ev[['log_price','mileage','age']])
    iso=IsolationForest(contamination=.01,random_state=SEED,n_jobs=1)
    ev['iso_outlier']=(iso.fit_predict(x)==-1).astype(int)
    ev['iso_anomaly_score']=iso.decision_function(x)
    designated=(ev.manufacturer.eq('Toyota') & ev.model_line.eq('Toyota bZ4X') & ev.price.eq(333333))
    ev['confirmed_error_flag']=designated.astype(int)
    ev['error_review_status']=np.where(designated,'project_designated_error_not_independently_verified',np.where(ev.iqr_outlier.eq(1)|ev.iso_outlier.eq(1),'statistical_flag_retained','not_flagged'))
    ev.loc[designated,['listing_id','source_excel_row','manufacturer','model','year','mileage','price','error_review_status']].assign(reason='Project instruction identifies Toyota bZ4X USD333333 as error; raw source alone cannot independently confirm').to_csv(root/'data/clean/outlier_ids.csv',index=False)
    review=ev[(ev.iqr_outlier==1)|(ev.iso_outlier==1)|designated].copy()
    review['review_decision']='Keep: unusual value alone is not proof of error'
    review.loc[designated,'review_decision']='Exclude from training only per project rule; retain in clean file and held-out test'
    review.loc[(review.mileage<100)&~designated,'review_decision']='Keep: near-new/demo vehicle; use very_low_mileage_flag'
    review.loc[review.model_line.eq('Tesla Roadster')&~designated,'review_decision']='Keep: possible collector vehicle; separate model family'
    review.to_csv(root/'data/diagnostics/outlier_review.csv',index=False)
    cross=pd.crosstab(ev.iqr_outlier,ev.iso_outlier).reindex(index=[0,1],columns=[0,1],fill_value=0)
    cross.to_csv(root/'data/diagnostics/outlier_method_crosstab.csv')
    pd.DataFrame([dict(method='Within-group log-price IQR',flagged=int(ev.iqr_outlier.sum()),removed_from_clean=0,decision='Primary explainable review screen; whole-model fallback for groups under 10'),dict(method='Isolation Forest',flagged=int(ev.iso_outlier.sum()),removed_from_clean=0,decision='Complementary multi-variable review; .01 contamination controls rate, not error certainty')]).to_csv(root/'data/diagnostics/outlier_method_comparison.csv',index=False)
    print('IQR flags',int(ev.iqr_outlier.sum()),'Isolation flags',int(ev.iso_outlier.sum()),'project-designated errors',int(designated.sum()),flush=True)
    return ev,cross

def make_imputation_pipeline(method):
    if method not in ['median','knn']: raise ValueError(method)
    imp=SimpleImputer(strategy='median') if method=='median' else KNNImputer(n_neighbors=5)
    numeric=Pipeline([('scale',StandardScaler()),('impute',imp)])
    # Treat binary history as binary. KNN averages must not invent fractional accident histories.
    history=Pipeline([('impute',SimpleImputer(strategy='most_frequent'))])
    prep=ColumnTransformer([('numeric',numeric,NUMERIC),('history',history,BINARY)])
    return Pipeline([('prep',prep),('ridge',Ridge(alpha=1.0))])

def compare_imputers(ev,root):
    root=Path(root)
    ids=ev.listing_id.to_numpy()
    dev_ids,test_ids=train_test_split(ids,test_size=.2,random_state=SEED)
    split=ev[['listing_id']].copy(); split['split']=np.where(split.listing_id.isin(test_ids),'held_out_test','development')
    split.to_csv(root/'data/clean/train_test_split_ids.csv',index=False)
    dev=ev[ev.listing_id.isin(dev_ids)].sort_values('listing_id').copy()
    cv=KFold(n_splits=5,shuffle=True,random_state=SEED)
    results=[]
    for method in ['median','knn']:
        for fold,(tr,va) in enumerate(cv.split(dev),1):
            t=time.time()
            train=dev.iloc[tr]; valid=dev.iloc[va]
            train=train[train.confirmed_error_flag.eq(0)]
            pipe=make_imputation_pipeline(method)
            pipe.fit(train[NUMERIC+BINARY],train.log_price)
            pred=pipe.predict(valid[NUMERIC+BINARY])
            results.append(dict(method=method,fold=fold,train_rows=len(train),validation_rows=len(valid),mae_log_price=mean_absolute_error(valid.log_price,pred),mae_usd=mean_absolute_error(valid.price,np.exp(pred)),seconds=time.time()-t))
            print('CV',method,fold,'log-MAE',round(results[-1]['mae_log_price'],6),'seconds',round(results[-1]['seconds'],1),flush=True)
            pd.DataFrame(results).to_csv(root/'data/diagnostics/imputation_cv_folds.csv',index=False)
    detail=pd.DataFrame(results)
    summary=detail.groupby('method').agg(mean_mae_log_price=('mae_log_price','mean'),sd_mae_log_price=('mae_log_price','std'),mean_mae_usd=('mae_usd','mean'),total_seconds=('seconds','sum')).reset_index()
    winner=summary.sort_values('mean_mae_log_price').iloc[0]['method']
    summary['selected']=summary.method.eq(winner)
    summary.to_csv(root/'data/diagnostics/imputation_cv_summary.csv',index=False)
    decision=dict(selected_method=winner,criterion='Lower mean 5-fold validation MAE on log price; development 80% only',development_rows=len(dev),held_out_test_rows=len(test_ids),test_performance_evaluated=False,numeric_fields=NUMERIC,binary_fields=BINARY,binary_imputation='most frequent for both methods',scaler_fit='training fold only, before numeric imputer',seed=SEED,knn_neighbors=5,ridge_alpha=1.0)
    (root/'data/diagnostics/imputation_decision.json').write_text(json.dumps(decision,indent=2))
    return summary,decision

def export_clean(ev,root):
    root=Path(root)
    ev.to_csv(root/'data/clean/ev_listings_clean_v2.csv',index=False)
    ev.describe().T.to_csv(root/'data/diagnostics/summary_statistics.csv')
    return root/'data/clean/ev_listings_clean_v2.csv'
