# M2 code for the team repository

Copy src and notebooks into the existing repository, preserving other members’ files. This package has no raw data, cleaned complaint narratives, credentials or model binaries. It has not been pushed remotely.

M5 owns src/prep.py, src/condition_features.py and the master requirements files; use their latest versions rather than overwriting them. src/train_price_cut.py expects those modules and src/make_fair_prices.py to be present in the team checkout. It records M5’s generation-code hash for provenance but does not regenerate the price files.

Local training data required (not bundled here): data/model/model_dataset.csv, oof_fair_price_v3.csv, fair_price_nested_v3.csv, fold_ids_v1.csv and the original train_test_split_ids.csv; data/clean/ev_listings_clean_v1.csv, outlier_ids.csv and dim_reliability.csv. Paths are explicit in the script. The original M1 split is retained as a cross-check of M5 membership. Add these files locally without committing raw data or sensitive information.

Run python src/train_price_cut.py from the repository root using the agreed environment. Results go to results/M2 and models. Both selected parameter sets match the previous M2 run. Thresholds are derived from development folds; test recall is 73.77% for the selected tree and does not meet the 80% target. M5 registers the delivered model in the shared registry.

requirements_M2_runtime.txt records the environment used for this handover; it is not a replacement for M5’s master or serving requirements. Notebooks 02 and 06 retain the team numbering. Notebook 02 includes optional raw-data downloads; run those cells only when the downloads are required.
