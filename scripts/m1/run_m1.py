"""Run from any working directory: python /path/to/scripts/run_m1.py"""
from pathlib import Path
from source_loader import load_source
from m1_pipeline import clean_listings,missingness,mark_outliers,export_clean,compare_imputers
from m1_eda import create_eda
ROOT=Path(__file__).resolve().parent.parent
raw,profile=load_source(ROOT)
ev=clean_listings(raw,profile,ROOT)
missingness(ev,ROOT)
ev,cross=mark_outliers(ev,ROOT)
export_clean(ev,ROOT)
summary,decision=compare_imputers(ev,ROOT)
create_eda(ev,ROOT)
print(summary.to_string(index=False))
print('Rerun completed. Review static written report if data or method choices change.')
