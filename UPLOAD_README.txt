M1 GitHub upload draft — 19 October 2026

Upload these two files to the repository, preserving folders:
  notebooks/01_cleaning.ipynb
  docs/data_dictionary.csv
Do not upload this ZIP itself as a substitute for the two files.

Changes: includes the 48-field condition-feature version, closes the eight-row
review item with a retention decision, records M5's ablation feedback, and embeds
all M1 cleaning functions to avoid imports from absent scripts/ modules.
This draft does not modify M5 src/prep.py or final model/deployment code.

The notebook was executed against the audited Electric source snapshot: 15,155
output rows and 48 fields. Full-source CSV/XLSX loading remains available but was
not rerun in this draft. The snapshot comes from the original full-source scan.

Re-running requires local source data from the shared M1 handover under data/raw/.
Outputs and fixed M1 splits are regenerated locally; preserve M5's final splits.
The dictionary describes M1 data only; a final Appendix A covering SQL/complaint
features must be reconciled with M3/M5's latest extract.

Suggested commit: Add M1 cleaning notebook and 48-field data dictionary
中文：按文件夹上传两个文件；notebook自带清洗函数，但仍需要本地原始数据。
无需上传原始数据或替换组员prep.py。M5最新v5未在远端main看到，本稿不改它。
