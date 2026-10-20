"""M2 training using M5's supplied fold-specific fair prices and unchanged prep v5.
Run from M2_work. Test labels are used only after parameters, winner and thresholds lock.
"""
from pathlib import Path
import sys, json, hashlib, warnings, platform
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,joblib,sklearn,scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,balanced_accuracy_score
from sklearn.base import clone
from src import prep
import mlflow,mlflow.sklearn

def metrics(y,p,t):
    pred=np.asarray(p)>=t
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(roc_auc=float(roc_auc_score(y,p)),accuracy=float(accuracy_score(y,pred)),
        balanced_accuracy=float(balanced_accuracy_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),
        recall=float(recall_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def threshold(y,p):
    fpr,tpr,t=roc_curve(y,p,drop_intermediate=False)
    ix=np.flatnonzero((tpr>=.8)&np.isfinite(t))
    return float(t[ix[np.argmin(fpr[ix])]])

def frame(df,prices):
    # Explicit whitelist: never pass targets, current price or price-drop fields.
    x=df[prep.RAW_INPUTS+prep.COMPLAINT_RAW].copy()
    prices=np.asarray(prices,dtype=float)
    if not np.isfinite(prices).all() or (prices<=0).any():raise ValueError('Invalid fair prices')
    x['ask_gap_pct']=(df.initial_price.to_numpy()-prices)/prices
    return x

def get_inputs():
    d=pd.read_csv(ROOT/'data/model/model_dataset.csv').rename(columns={'listing_key':'listing_id'})
    o=pd.read_csv(ROOT/'data/model/oof_fair_price_v3.csv')
    n=pd.read_csv(ROOT/'data/model/fair_price_nested_v3.csv')
    original=pd.read_csv(ROOT/'data/clean/ev_listings_clean_v1.csv')
    oldsplit=pd.read_csv(ROOT/'data/model/train_test_split_ids.csv')
    errors=set(pd.read_csv(ROOT/'data/clean/outlier_ids.csv').iloc[:,0])
    if not d.listing_id.is_unique or not o.listing_id.is_unique or n.duplicated(['outer_fold','listing_id']).any():raise ValueError('Duplicate IDs')
    if set(d.listing_id)!=set(original.listing_id):raise ValueError('Warehouse ID mismatch')
    missing=set(d.listing_id)-set(o.listing_id)
    if missing!=errors:raise ValueError('Unexplained missing fair prices')
    if set(o.listing_id)-set(d.listing_id):raise ValueError('Extra fair-price IDs')
    if set(o.split)!= {'train','test'} or set(n.role)!={'classifier_train','classifier_valid'}:raise ValueError('Unexpected split or role')
    if not o.fair_price.gt(0).all() or not np.isfinite(o.fair_price).all():raise ValueError('Invalid fair price')
    merged=d.merge(o,on='listing_id',validate='one_to_one',how='inner')
    tr=merged[merged.split.eq('train')].copy().sort_values('listing_id').reset_index(drop=True)
    te=merged[merged.split.eq('test')].copy().sort_values('listing_id').reset_index(drop=True)
    if errors & set(te.listing_id) or errors & set(tr.listing_id):raise ValueError('Error listing present')
    if not tr.fair_price_source.eq('out_of_fold').all() or not te.fair_price_source.eq('model_fit_on_all_train').all():raise ValueError('Prediction origin mismatch')
    if not tr.prediction_origin.eq('oof_train').all() or not te.prediction_origin.eq('train_only_test').all():raise ValueError('M5 interface provenance mismatch')
    supplied_folds=pd.read_csv(ROOT/'data/model/fold_ids_v1.csv')
    if set(supplied_folds.listing_id)!=set(tr.listing_id):raise ValueError('Supplied fold IDs differ')
    if not tr.merge(supplied_folds,on='listing_id',validate='one_to_one',suffixes=('_prices','_ids')).eval('fold_prices == fold_ids').all():raise ValueError('Supplied fold assignments differ')
    alignment=o.merge(oldsplit,on='listing_id',suffixes=('_m5','_m1'),validate='one_to_one')
    if not (alignment.split_m5.eq('train')==alignment.split_m1.eq('development')).all():raise ValueError('Split mismatch')
    if set(n.outer_fold)!={0,1,2,3,4} or set(tr.fold)!={0,1,2,3,4}:raise ValueError('Expected five folds')
    if not n.fair_price.gt(0).all() or not np.isfinite(n.fair_price).all():raise ValueError('Invalid nested prices')
    fold_checks=[]
    for f in range(5):
        sub=n[n.outer_fold.eq(f)]
        tset=set(sub.loc[sub.role.eq('classifier_train'),'listing_id']);vset=set(sub.loc[sub.role.eq('classifier_valid'),'listing_id'])
        if tset&vset or tset|vset!=set(tr.listing_id):raise ValueError('Nested fold coverage/overlap')
        if vset!=set(tr.loc[tr.fold.eq(f),'listing_id']):raise ValueError('Nested fold differs from M5 fold IDs')
        fold_checks.append({'fold':f,'train_rows':len(tset),'validation_rows':len(vset)})
    # Preserve M3's raw spellings. Derive all engineered fields through M5's recipe.
    canonical=original.set_index('listing_id').loc[d.listing_id]
    for col in ['price','initial_price','price_cut_flag','year','mileage']:
        if not np.allclose(d[col],canonical[col],equal_nan=True):raise ValueError('Source mismatch: '+col)
    rel=pd.read_csv(ROOT/'data/clean/dim_reliability.csv')
    rc=d.merge(rel,on=['make','model_line','year'],validate='many_to_one',how='left',suffixes=('_m3','_m4'),indicator=True)
    if not rc['_merge'].eq('both').all():raise ValueError('Unmatched M4 keys')
    for col in prep.COMPLAINT_RAW:
        if not np.allclose(rc[col+'_m3'],rc[col+'_m4'],atol=.000051,rtol=0,equal_nan=True):raise ValueError('Reliability mismatch: '+col)
    checks={'warehouse_rows':len(d),'train_rows':len(tr),'test_rows':len(te),'excluded_training_ids':sorted(errors),
        'M1_split_matches_M5':True,'M4_features_match_warehouse':True,'fold_checks':fold_checks,
        'nested_test_IDs_present':False,'no_complaint_listings':int(d.avg_sentiment.isna().sum()),
        'M5_provenance_limit':'M5 generation code reviewed: each outer training block uses inner out-of-fold prices and validation cars are priced from the other folds. Price-model hyperparameters were chosen upstream on the overall training data; this is not a fully nested assessment of price-model selection.'}
    return tr,te,n,checks

def main():
    out=ROOT/'results/M2';out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True);(ROOT/'models').mkdir(exist_ok=True)
    tr,te,n,checks=get_inputs()
    (out/'input_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    tr[['listing_id','fold']].to_csv(ROOT/'data/model/train_ids_m2_eligible.csv',index=False)
    te[['listing_id']].to_csv(ROOT/'data/model/test_ids_m2.csv',index=False)
    y=tr.price_cut_flag.astype(int).to_numpy(); yt=te.price_cut_flag.astype(int).to_numpy()
    # Cache fold-fitted preprocessing. Each candidate sees exactly the same fold data.
    folds=[]
    for f in range(5):
        sub=n[n.outer_fold.eq(f)].set_index('listing_id').loc[tr.listing_id]
        prices=sub.fair_price.to_numpy(); x=frame(tr,prices)
        valid=sub.role.eq('classifier_valid').to_numpy(); training=~valid
        derived=prep.build_features(x); transformer=prep.make_prep(extra_num=['ask_gap_pct'])
        ztr=transformer.fit_transform(derived.loc[training]); zv=transformer.transform(derived.loc[valid])
        folds.append((training,valid,ztr,zv))
    grids={
        'logistic_regression':(LogisticRegression(class_weight='balanced',solver='liblinear',random_state=42),
            {'C':[.01,.1,1,10],'l1_ratio':[0,1],'max_iter':[500,1000]}),
        'decision_tree':(DecisionTreeClassifier(class_weight='balanced',random_state=42),
            {'max_depth':[3,5,8],'min_samples_leaf':[20,50],'criterion':['gini','entropy'],'ccp_alpha':[0,.001]})}
    mlflow.set_tracking_uri((out/'mlruns').resolve().as_uri());mlflow.set_experiment('ev_price_cut_M2_final_fair_prices_v3')
    selected={};runs=[]
    for name,(base,grid) in grids.items():
        print('Tuning',name,flush=True);records=[];best=-np.inf
        with mlflow.start_run(run_name=name+'_fold_search') as parent:
            mlflow.set_tags({'owner':'M2','prep_version':prep.VERSION,'price_inputs':'M5_supplied_nested_folds','target':'historical_recorded_cut'})
            for candidate,params in enumerate(ParameterGrid(grid)):
                p=np.zeros(len(tr));scores=[];train_scores=[]
                for training,valid,ztr,zv in folds:
                    clf=clone(base).set_params(**params)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',FutureWarning);clf.fit(ztr,y[training])
                    p[valid]=clf.predict_proba(zv)[:,1]
                    scores.append(float(roc_auc_score(y[valid],p[valid])))
                    train_scores.append(float(roc_auc_score(y[training],clf.predict_proba(ztr)[:,1])))
                mean=float(np.mean(scores));row={'candidate':candidate,'params':json.dumps(params),'mean_cv_auc':mean,'std_cv_auc':float(np.std(scores)),
                    'mean_train_auc':float(np.mean(train_scores)),**{f'fold_{f}_auc':v for f,v in enumerate(scores)}};records.append(row)
                with mlflow.start_run(run_name=name+f'_candidate_{candidate}',nested=True):
                    mlflow.log_params(params);mlflow.log_metrics({k:v for k,v in row.items() if isinstance(v,(int,float))})
                if mean>best:
                    best=mean;bestparams=params;bestp=p.copy();best_train=float(np.mean(train_scores))
            t=threshold(y,bestp)
            selected[name]={'params':bestparams,'mean_cv_auc':best,'mean_fold_train_auc':best_train,'threshold':t,
                'pooled_cv_metrics':metrics(y,bestp,t)}
            pd.DataFrame(records).to_csv(out/f'{name}_cv_results.csv',index=False)
            preds=tr[['listing_id','fold','price_cut_flag']].copy();preds['score_recorded_price_cut']=bestp;preds.to_csv(out/f'{name}_cv_predictions.csv',index=False)
            mlflow.log_metrics({'selected_mean_cv_auc':best,'threshold':t});mlflow.log_artifact(str(out/f'{name}_cv_results.csv'))
            runs.append({'name':name,'run_id':parent.info.run_id})
    winner=max(selected,key=lambda name:selected[name]['mean_cv_auc'])
    # Freeze all decisions before reading test outcomes for evaluation.
    (out/'selection_locked.json').write_text(json.dumps({'winner':winner,'models':selected},indent=2),encoding='utf-8')
    xtr=frame(tr,tr.fair_price);xte=frame(te,te.fair_price)
    fitted={name:prep.make_classifier(clone(base).set_params(**selected[name]['params'])).fit(xtr,y) for name,(base,_) in grids.items()}
    baseline=DummyClassifier(strategy='most_frequent').fit(xtr,y); fitted['majority_baseline']=baseline
    results=[]
    for name,model in fitted.items():
        p=model.predict_proba(xte)[:,1];t=selected[name]['threshold'] if name in selected else .5
        row={'model':name,'threshold':t,'mean_cv_auc':selected[name]['mean_cv_auc'] if name in selected else .5,**metrics(yt,p,t)};results.append(row)
        preds=te[['listing_id','price_cut_flag']].copy();preds['score_recorded_price_cut']=p;preds['decision']=p>=t;preds.to_csv(out/f'{name}_test_predictions.csv',index=False)
        joblib.dump(model,ROOT/'models'/f'{name}.joblib')
        with mlflow.start_run(run_name=name+'_test_evaluation') as run:
            mlflow.log_params({'train_rows':len(tr),'test_rows':len(te),'threshold':t});mlflow.log_metrics({k:v for k,v in row.items() if isinstance(v,(int,float))})
            mlflow.log_input(mlflow.data.from_pandas(tr[['listing_id','price_cut_flag']],name='M5_train_IDs',targets='price_cut_flag'),context='training')
            mlflow.log_artifact(str(out/f'{name}_test_predictions.csv'))
            if name==winner:
                info=mlflow.sklearn.log_model(model,name='price_cut_pipeline',input_example=xtr.head(3),serialization_format='cloudpickle')
                registry={'name':'ev_price_cut_model','status':'ready_for_M5_registration','run_id':run.info.run_id,'scope':'Local tracking only; M5 owns registration in the shared registry.'}
    pd.DataFrame(results).to_csv(out/'model_comparison.csv',index=False)
    joblib.dump(fitted[winner],ROOT/'models/price_cut_model.joblib')
    names=fitted['logistic_regression'].named_steps['prep'].get_feature_names_out();coef=fitted['logistic_regression'].named_steps['clf'].coef_[0]
    pd.DataFrame({'feature':names,'coefficient':coef,'odds_ratio_per_transformed_unit':np.exp(coef)}).to_csv(out/'logistic_odds_ratios.csv',index=False)
    trans=fitted['logistic_regression'].named_steps['prep'].named_transformers_['num'].named_steps['standardscaler']
    numeric=prep.NUM+prep.CONDITION+prep.COMPLAINT+['ask_gap_pct'];i=numeric.index('ask_gap_pct')
    gap_or=float(np.exp(coef[i]*.1/trans.scale_[i]))
    explanations={'ask_gap_odds_ratio_per_10_percentage_points':gap_or,'note':'Adjusted association, holding other engineered inputs fixed. Not causal; outputs are class-weighted scores, not calibrated probabilities.'}
    (out/'explanations.json').write_text(json.dumps(explanations,indent=2),encoding='utf-8')
    tree=fitted['decision_tree'];fig,ax=plt.subplots(figsize=(23,10));plot_tree(tree.named_steps['clf'],max_depth=2,feature_names=tree.named_steps['prep'].get_feature_names_out(),class_names=['No recorded cut','Recorded cut'],filled=True,rounded=True,fontsize=8,ax=ax);fig.tight_layout();fig.savefig(out/'figures/decision_tree_top3.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for name in selected:
        fpr,tpr,_=roc_curve(yt,fitted[name].predict_proba(xte)[:,1]);ax.plot(fpr,tpr,label=name.replace('_',' '))
    ax.plot([0,1],[0,1],'k--',label='Majority baseline');ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Recorded price cuts: team test split');ax.legend();fig.tight_layout();fig.savefig(out/'figures/roc_comparison.png',dpi=150);plt.close(fig)
    status={'status':'M2 completed against supplied M3/M4/M5 files','winner':winner,'threshold':selected[winner]['threshold'],
        'threshold_policy':'Minimum 80% recall; lowest false-positive rate on pooled development CV scores. Scores reuse folds used for tuning; this is not an unbiased threshold-performance estimate.',
        'checks':checks,'models':selected,'test_results':results,'registry':registry,'explanations':explanations,
        'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data/model/model_dataset.csv',ROOT/'data/model/oof_fair_price_v3.csv',ROOT/'data/model/fair_price_nested_v3.csv',ROOT/'src/prep.py',ROOT/'src/condition_features.py',ROOT/'src/make_fair_prices.py']},
        'versions':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__,'mlflow':mlflow.__version__},
        'limitations':['M1 treated missing price drops as no recorded cut.','Existing test labels have been inspected in earlier pilot runs. This rerun does not provide a fresh prospective validation.',
            'Initial asking prices are reconstructed in a single historical snapshot. The gap can retain outcome-related information; results do not establish future price-cut prediction.',
            checks['M5_provenance_limit'],'M5 supplied final price_model_v3 (XGBoost without condition features); M2 classifier retains its previous condition-feature settings.']}
    (out/'model_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    (ROOT/'models/price_cut_metadata.json').write_text(json.dumps({'model':winner,'threshold':status['threshold'],'raw_inputs':prep.RAW_INPUTS,'complaint_inputs':prep.COMPLAINT_RAW,'additional_input':'ask_gap_pct','gap_formula':'(initial_price - fair_price) / fair_price','prep_version':prep.VERSION,'registry':registry,'target':'historical_recorded_price_cut','score_is_calibrated_probability':False},indent=2),encoding='utf-8')
    print(json.dumps({'winner':winner,'results':results,'gap_or':gap_or,'registry':registry},indent=2),flush=True)
if __name__=='__main__':main()
