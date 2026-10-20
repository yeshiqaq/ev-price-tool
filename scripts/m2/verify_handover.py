from pathlib import Path
import sys,json,hashlib
w=Path(__file__).resolve().parents[1];r=w.parent;sys.path.insert(0,str(w.resolve()));sys.path.insert(0,str((w/'scripts').resolve()))
import pandas as pd,numpy as np,joblib
from src import prep
from train_price_cut import get_inputs,frame
from src.price_cut import PriceCutScorer
tr,te,n,checks=get_inputs()
x=frame(te,te.fair_price)
reload={}
for name in ['logistic_regression','decision_tree','majority_baseline']:
    model=joblib.load(w/'models'/f'{name}.joblib');saved=pd.read_csv(w/'results'/f'{name}_test_predictions.csv')
    assert list(saved.listing_id)==list(te.listing_id)
    np.testing.assert_allclose(model.predict_proba(x)[:,1],saved.score_recorded_price_cut,rtol=1e-10,atol=1e-10)
    reload[name]=True
assert not set(x)&set(prep.NEVER_INPUTS)
model=joblib.load(w/'models/price_cut_model.joblib')
with_targets=x.assign(price=1,price_cut_flag=0,price_drop=100000)
np.testing.assert_allclose(model.predict_proba(x),model.predict_proba(with_targets))
assert (w/'src/prep.py').read_bytes()==(w/'sources/M5_received/src/prep.py').read_bytes()
assert (w/'src/condition_features.py').read_bytes()==(w/'sources/M5_received/src/condition_features.py').read_bytes()
# Full scoring helper needs the M5 price model and XGBoost. Keep the same test rows.
shutil_import=__import__('shutil')
shutil_import.copy2(w/'sources/M4_received/dim_reliability.csv',w/'data/clean/dim_reliability.csv')
scorer=PriceCutScorer(w)
result=scorer.predict(te)
delta=np.abs(result.fair_price.to_numpy()-te.fair_price.to_numpy())
within_rounding=bool((delta<=.51).all())
if not within_rounding:raise AssertionError('M5 saved price model differs from supplied test fair prices: '+str(delta.max()))
np.testing.assert_array_equal(result.fair_price.to_numpy(),te.fair_price.to_numpy())
np.testing.assert_allclose(result.recorded_cut_score,model.predict_proba(x)[:,1],rtol=1e-10,atol=1e-10)
# Direct and helper scores should agree when fed the helper's unrounded prices.
np.testing.assert_allclose(model.predict_proba(frame(te,result.fair_price))[:,1],result.recorded_cut_score,rtol=1e-10,atol=1e-10)
invalid=te.head(1).copy();invalid['initial_price']=0
try:scorer.predict(invalid)
except ValueError:pass
else:raise AssertionError('Zero asking price accepted')
example=te.loc[te.avg_sentiment.isna()].head(1)
assert len(example)==1 and np.isfinite(scorer.predict(example).recorded_cut_score).all()
unseen=te.head(1).copy();unseen['model_line']='Unseen model line';unseen['trim_name']='Unknown trim'
assert not scorer.predict(unseen).reliability_matched.iloc[0]
result.head(5).to_csv(w/'results/inference_example_outputs.csv',index=False)
te.head(5)[prep.RAW_INPUTS+['initial_price']].to_csv(w/'results/inference_example_inputs.csv',index=False)
report={'input_checks':checks,'saved_model_predictions_reproduced':reload,'shared_prep_unchanged':True,
    'target_columns_do_not_affect_predictions':True,'full_price_and_cut_scoring_tested':True,'zero_asking_price_rejected':True,
    'no_complaint_and_unseen_model_inputs_tested':True,'M5_test_prices_match_saved_price_model_to_rounding':within_rounding,
    'maximum_price_difference_USD':float(delta.max()),'note':'The helper rounds fair prices to whole dollars, matching M5 CSVs. End-to-end scores match all exported test scores.'}
(w/'results/verification_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
