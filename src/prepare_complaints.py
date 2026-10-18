"""Reproduce M2 Part A from official five-year ZIPs and M1's listing CSV.
Usage: python scripts/prepare_complaints.py --raw-dir data/raw --m1-data data/input --output data
Mapping is an exact alias whitelist, not a fuzzy match. Received date is field 17
(zero-based 16), NOT DATEA (field 16 / zero-based 15) in the supplied guide.
"""
from pathlib import Path
import argparse, zipfile, re, json
from collections import Counter
import pandas as pd
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--raw-dir',type=Path,required=True); p.add_argument('--m1-data',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--lookup',type=Path)
    a=p.parse_args(); clean=a.output/'clean'; diagnostics=a.output/'diagnostics'; clean.mkdir(parents=True,exist_ok=True); diagnostics.mkdir(exist_ok=True)
    lookup_path=a.lookup or clean/'model_lookup.csv'; lookup=pd.read_csv(lookup_path,keep_default_na=False)
    assert not lookup.duplicated(['nhtsa_make','nhtsa_model']).any()
    alias={(r.nhtsa_make,r.nhtsa_model):(r.make,r.model_line) for r in lookup.itertuples()}
    listings=pd.read_csv(a.m1_data/'ev_listings_clean_v1.csv'); assert listings.listing_id.is_unique
    assert set(lookup.model_line).issubset(set(listings.model_line))
    brands=set(listings.manufacturer.str.upper())|{'MERCEDES BENZ','VW'}
    rows=[]; inventory=Counter(); scans=[]
    expected=[f'COMPLAINTS_RECEIVED_{years}.zip' for years in ['2010-2014','2015-2019','2020-2024']]
    for name in expected:
        path=a.raw_dir/name; counts=Counter(); lengths=Counter()
        with zipfile.ZipFile(path) as z:
            for member in [n for n in z.namelist() if n.lower().endswith('.txt')]:
                with z.open(member) as f:
                    for line_no,line in enumerate(f,1):
                        counts['raw_component_rows']+=1
                        fields=line.decode('latin-1').rstrip('\r\n').split('\t'); lengths[len(fields)]+=1
                        if len(fields)!=51: counts['malformed_rows']+=1; continue
                        date=fields[16].strip()
                        if not re.fullmatch(r'\d{8}',date): counts['invalid_date_format']+=1; continue
                        if date>='20230501': counts['after_cutoff_rows']+=1; continue
                        counts['before_cutoff_rows']+=1
                        if fields[45].strip()!='V': counts['non_vehicle_rows']+=1; continue
                        make=fields[3].strip().upper()
                        if make not in brands: continue
                        counts['candidate_brand_component_rows']+=1
                        model=re.sub(r'\s+',' ',fields[4].upper().replace('REDUNDANT ','').strip())
                        inventory[(make,model)]+=1
                        if (make,model) not in alias: continue
                        brand,family=alias[(make,model)]
                        rows.append([fields[1].strip(),make,model,fields[5].strip(),fields[6].strip(),fields[8].strip(),fields[11].strip(),date,fields[19].strip(),fields[15].strip(),name,str(line_no),brand,family])
        scans.append({'file':name,**dict(counts),'field_counts':dict(lengths)}); print(scans[-1],flush=True)
    c=pd.DataFrame(rows,columns=['odino','nhtsa_make','nhtsa_model','year','crash','fire','component','date_received_raw','text','date_added_raw','source_file','source_line','make','model_line'])
    c['date_received']=pd.to_datetime(c.date_received_raw,format='%Y%m%d',errors='coerce'); c['date_added']=pd.to_datetime(c.date_added_raw,format='%Y%m%d',errors='coerce')
    invalid=c.date_received.isna()|c.odino.eq(''); c.loc[invalid].to_csv(diagnostics/'invalid_matched_records.csv',index=False); c=c.loc[~invalid].copy()
    wrong=c.date_added.isna()|c.date_added.ge('2023-05-01'); c.loc[wrong].to_csv(diagnostics/'received_vs_added_cutoff_difference.csv',index=False)
    rex=c.model_line.eq('BMW i3') & c.text.str.contains(r'range[ -]?extender|\brex\b|gasoline engine|gas engine',case=False,regex=True,na=False); rex_ids=set(c.loc[rex,'odino'])
    c.loc[c.odino.isin(rex_ids)].to_csv(diagnostics/'explicit_i3_rex_excluded.csv',index=False); c=c.loc[~c.odino.isin(rex_ids)].copy()
    identity=c.groupby('odino')[['model_line','year']].nunique(); conflicts=set(identity.index[identity.gt(1).any(axis=1)])
    c.loc[c.odino.isin(conflicts)].to_csv(diagnostics/'multiple_model_complaints_review.csv',index=False); c=c.loc[~c.odino.isin(conflicts)].copy()
    def joined(s): return '|'.join(sorted(set(v for v in s if v)))
    def yn(s): return 'Y' if s.eq('Y').any() else ('N' if s.eq('N').any() else 'Unknown')
    ev=c.groupby('odino',sort=True).agg(make=('make','first'),model_line=('model_line','first'),year=('year','first'),crash=('crash',yn),fire=('fire',yn),component=('component',joined),date_received=('date_received','min'),text=('text',lambda s:max(s,key=len)),nhtsa_make=('nhtsa_make',joined),nhtsa_model=('nhtsa_model',joined),date_added=('date_added','min'),source_files=('source_file',joined),source_component_rows=('odino','size'),source_line_numbers=('source_line',joined)).reset_index()
    ev['year']=pd.to_numeric(ev.year,errors='coerce').astype('Int64'); ev['year_review_flag']=~ev.year.between(1900,2023).fillna(False); ev['powertrain_review_flag']=ev.model_line.eq('BMW i3'); ev['powertrain_review_note']=np.where(ev.powertrain_review_flag,'I3 name may include unreported range-extender variants; do not claim exact BEV verification',''); ev['text_missing_flag']=ev.text.eq('')
    ev['date_received']=ev.date_received.dt.strftime('%Y-%m-%d'); ev['date_added']=ev.date_added.dt.strftime('%Y-%m-%d')
    ev.to_csv(clean/'ev_complaints_clean_v1.csv',index=False,encoding='utf-8-sig')
    counts=ev.loc[~ev.year_review_flag].groupby(['make','model_line','year']).size().rename('n_matched_complaints').reset_index()
    coverage=listings[['listing_id','manufacturer','model_line','year','bev_status_review_flag']].merge(counts,left_on=['manufacturer','model_line','year'],right_on=['make','model_line','year'],how='left',validate='many_to_one'); coverage['n_matched_complaints']=coverage.n_matched_complaints.fillna(0).astype(int); coverage['has_matched_complaint']=coverage.n_matched_complaints.gt(0).astype(int); coverage.to_csv(diagnostics/'listing_complaint_coverage.csv',index=False)
    family=coverage.groupby(['manufacturer','model_line']).agg(n_listings=('listing_id','size'),covered_listings=('has_matched_complaint','sum')).reset_index().merge(ev.loc[~ev.year_review_flag].groupby('model_line').size().rename('unique_complaints'),on='model_line',how='left').fillna({'unique_complaints':0}); family['model_year_coverage']=family.covered_listings/family.n_listings; family.to_csv(diagnostics/'model_line_coverage.csv',index=False,encoding='utf-8-sig')
    inv=pd.DataFrame([{'nhtsa_make':k[0],'nhtsa_model':k[1],'component_rows':n} for k,n in inventory.items()]); inv.to_csv(diagnostics/'nhtsa_model_inventory.csv',index=False)
    q={'cutoff_exclusive':'2023-05-01','received_field_zero_based':16,'guide_incorrect_field_zero_based':15,'source_scan':scans,'m1_listing_rows':len(listings),'m1_model_lines':listings.model_line.nunique(),'lookup_rows':len(lookup),'matched_component_rows_before_validation':len(rows),'invalid_matched_rows':int(invalid.sum()),'explicit_rex_complaints_excluded':len(rex_ids),'multiple_model_complaints_quarantined':len(conflicts),'unique_matched_complaints':len(ev),'matched_model_lines':ev.model_line.nunique(),'tesla_complaints':int(ev.make.eq('Tesla').sum()),'model_year_listing_coverage':float(coverage.has_matched_complaint.mean()),'family_listing_coverage':float(listings.model_line.isin(ev.model_line).mean()),'year_review_complaints':int(ev.year_review_flag.sum()),'text_missing_complaints':int(ev.text_missing_flag.sum()),'i3_powertrain_review_complaints':int(ev.powertrain_review_flag.sum()),'received_vs_added_cutoff_difference_component_rows':int(wrong.sum()),'odino_unique':bool(ev.odino.is_unique),'m1_bev_review_listing_rows':int(listings.bev_status_review_flag.sum()),'snapshot_note':'Received-date cutoff excludes later complaints, but older complaints may have been edited after April 2023. Not an archived April 2023 extract.'}
    (diagnostics/'complaints_quality.json').write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in q.items() if k!='source_scan'},indent=2),flush=True)
if __name__=='__main__': main()
