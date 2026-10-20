"""Download public official complaint ZIPs. Raw archives are not in the handover."""
from pathlib import Path
from urllib.request import urlopen,Request
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,hashlib,json,zipfile
def main():
    a=argparse.ArgumentParser(); a.add_argument('--raw-dir',type=Path,default=Path('data/raw')); args=a.parse_args(); root=args.raw_dir; root.mkdir(parents=True,exist_ok=True)
    names=['COMPLAINTS_RECEIVED_2010-2014.zip','COMPLAINTS_RECEIVED_2015-2019.zip','COMPLAINTS_RECEIVED_2020-2024.zip','CMPL.txt']
    def fetch(name):
        url='https://static.nhtsa.gov/odi/ffdd/cmpl/'+name; p=root/name
        if not p.exists():
            with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=90) as r,p.with_suffix(p.suffix+'.part').open('wb') as out:
                for chunk in iter(lambda:r.read(1024*1024),b''): out.write(chunk)
            p.with_suffix(p.suffix+'.part').replace(p)
        if name.endswith('.zip') and not zipfile.is_zipfile(p): raise ValueError('Not a ZIP: '+name)
        return {'file':name,'url':url,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'download_date':datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()}
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(fetch,n) for n in names]): results.append(f.result()); print(results[-1],flush=True)
    (root/'source_manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
if __name__=='__main__':main()
