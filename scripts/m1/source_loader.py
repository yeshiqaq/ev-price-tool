"""Scan the full supplied source when available; otherwise use the audited snapshot."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import pandas as pd
import json,hashlib,collections

def load_source(root):
    root=Path(root); rawdir=root/'data/raw'
    xlsx=rawdir/'cars.xlsx'; csv=rawdir/'cars.csv'
    if xlsx.exists():
        ns='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
        with ZipFile(xlsx) as z:
            strings=[''.join(x.itertext()) for _,x in etree.iterparse(z.open('xl/sharedStrings.xml'),events=('end',),tag=ns+'si')]
            rows=[]; total=0; counts=collections.Counter(); headers=None
            for _,row in etree.iterparse(z.open('xl/worksheets/sheet1.xml'),events=('end',),tag=ns+'row'):
                values=[]
                for c in row:
                    letters=''.join(v for v in c.get('r','') if v.isalpha()); idx=0
                    for v in letters: idx=idx*26+ord(v)-64
                    while len(values)<idx: values.append(None)
                    v=c.find(ns+'v')
                    if v is not None:
                        value=v.text
                        if c.get('t')=='s': value=strings[int(value)]
                        elif c.get('t')=='b': value=bool(int(value))
                        elif c.get('t') not in ['str','e']:
                            try: value=float(value); value=int(value) if value.is_integer() else value
                            except (TypeError,ValueError): pass
                        values[idx-1]=value
                    elif c.get('t')=='inlineStr': values[idx-1]=''.join(c.xpath('.//*[local-name()="t"]/text()'))
                if headers is None: headers=values; fuel_idx=headers.index('fuel_type')
                elif any(v is not None for v in values):
                    total+=1; fuel=values[fuel_idx] if len(values)>fuel_idx else None; counts[str(fuel)]+=1
                    if str(fuel).strip().casefold()=='electric':
                        values += [None]*(len(headers)-len(values)); rows.append(values[:len(headers)]+[int(row.get('r'))])
                row.clear()
                while row.getprevious() is not None: del row.getparent()[0]
        raw=pd.DataFrame(rows,columns=headers+['source_excel_row'])
        profile=dict(total_rows=total,electric_rows=len(raw),fuel_counts=dict(counts),columns=headers,source_filename=xlsx.name,source_sha256=hashlib.sha256(xlsx.read_bytes()).hexdigest())
        raw.to_csv(rawdir/'electric_source_snapshot.csv',index=False)
        (rawdir/'source_profile.json').write_text(json.dumps(profile,indent=2))
        return raw,profile
    if csv.exists():
        full=pd.read_csv(csv); full['source_excel_row']=range(2,len(full)+2)
        # source_excel_row here refers to the corresponding header-inclusive CSV row.
        raw=full[full.fuel_type.fillna('').str.strip().str.casefold().eq('electric')].copy()
        profile=dict(total_rows=len(full),electric_rows=len(raw),columns=[c for c in full if c!='source_excel_row'],source_filename=csv.name,source_sha256=hashlib.sha256(csv.read_bytes()).hexdigest())
        raw.to_csv(rawdir/'electric_source_snapshot.csv',index=False)
        (rawdir/'source_profile.json').write_text(json.dumps(profile,indent=2))
        return raw,profile
    profile=json.loads((rawdir/'source_profile.json').read_text())
    raw=pd.read_csv(rawdir/'electric_source_snapshot.csv')
    assert len(raw)==profile['electric_rows'], 'Snapshot count differs from audited source profile'
    print('Using included Electric subset snapshot; full workbook counts come from its audited source profile.')
    return raw,profile
