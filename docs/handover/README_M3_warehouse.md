# M3 — EV Data Warehouse Handover

**Project:** EV Resale Value Prediction (PAL Group 21)  
**Owner:** M3 — Data Warehouse & Deployment  
**Updated:** 13 October 2026  
**Status:** Data warehouse files shared; deployment pending

## 1. Purpose

This folder contains M3's MySQL data warehouse deliverables and the extracted modelling dataset for M2 and M5. The warehouse uses a star schema with one listing per row in the fact table. The six business queries support descriptive/OLAP analysis; they are not the input used to train the models.

## 2. Shared files

| Location / file | Description | Intended user |
|---|---|---|
| `sql_query/01_create_tables.sql` | Creates `ev_dw`, five dimensions and `fact_listing`. | Team / M3 |
| `sql_query/02_load.sql` | Loads staging data into dimensions and the fact table. | Team / M3 |
| `sql_query/03_business_questions.sql` | Six SQL queries answering management questions. | M4 / team |
| `sql_query/04_extract.sql` | Creates the `v_model_dataset` view for modelling. | M2 / M5 |
| `sql_query/answer_for_q3_business_question/` | Exported CSV results for Q1–Q6. Recommended folder name: `Business_Query_Results`. | M4 |
| `model_dataset.csv` | Extracted modelling data; intended for the shared train/test split and model pipelines. | M2 / M5 |
| `data_validation_results` | Warehouse data-quality and integrity checks (check actual file extension). | Team |
| `sql_star_schema` | Star-schema/EER diagram image (check actual extension). | M4 / team |
| `star_schema` | Workbench model, if this is the `.mwb` file; confirm format. | Team |

*File names above reflect the OneDrive screenshot. Verify extensions before copying paths or publishing to GitHub.*

## 3. Data sources and structure

- **Listings:** `ev_listings_clean_v1.csv` from M1; expected source size **15,155 listings**.
- **Reliability:** `dim_reliability.csv` from M4; expected source size **159 model-year summary rows**.
- **Database:** MySQL schema `ev_dw`.
- **Grain:** One row per EV listing in `fact_listing`.
- **Fact:** `fact_listing` — price, initial price, price drop, price-cut flag, mileage, driver rating, and dimension keys.
- **Dimensions:** `dim_vehicle`, `dim_model_year`, `dim_seller`, `dim_history`, `dim_reliability`.
- **Modelling view:** `v_model_dataset` joins the fact and dimensions and exposes vehicle, history, seller, reliability, and target fields.

**Important:** The dataset reflects the **April 2023 listing market**, not verified transaction prices or current market valuations.

## 4. How to reproduce the warehouse

1. Start MySQL and open MySQL Workbench.
2. Execute `01_create_tables.sql` to create the database and tables.
3. Import the cleaned listings and reliability CSVs into `stg_listings` and `stg_reliability` respectively. Use the complete listings CSV, not the earlier partial Workbench import.
4. Run `02_load.sql` **once** after checking the staging columns and NULL handling. Running inserts again may create duplicate dimension records or primary-key errors.
5. Run the validation queries below and inspect the EER diagram.
6. Run `03_business_questions.sql` for descriptive business analysis.
7. Run `04_extract.sql` and export `SELECT * FROM v_model_dataset` to CSV, ensuring Workbench's 1,000-row result limit is disabled.

The staging data files and database credentials are not included in this handover. Request source data from the team's shared folder; do not commit credentials to GitHub.

## 5. Data validation

Run the following in MySQL Workbench:

```sql
USE ev_dw;

SELECT COUNT(*) AS staging_listings FROM stg_listings;
SELECT COUNT(*) AS staging_reliability FROM stg_reliability;

SELECT
    COUNT(*) AS fact_rows,
    COUNT(DISTINCT listing_key) AS unique_listing_keys,
    SUM(vehicle_key IS NULL) AS missing_vehicle_key,
    SUM(model_year_key IS NULL) AS missing_year_key,
    SUM(seller_key IS NULL) AS missing_seller_key,
    SUM(history_key IS NULL) AS missing_history_key,
    SUM(reliability_key IS NULL) AS missing_reliability_key
FROM fact_listing;

SELECT COUNT(*) AS model_dataset_rows FROM v_model_dataset;
```

**Expected checks:** 15,155 staging listings, 159 staging reliability records, 15,155 unique fact listings, no missing dimension keys, and 15,155 rows in the modelling view. **These are targets, not independently verified results from the shared OneDrive files.** Record the actual outputs in `data_validation_results` and investigate discrepancies before declaring the handover final.

## 6. Business questions and result files

| Query | Question | Export |
|---|---|---|
| Q1 | How does average listing price vary by EV brand and vehicle age group? | `Q1_Price_by_Brand_Age.csv` |
| Q2 | How do listing prices differ by accident history within model lines? | `Q2_Price_by_Accident_History.csv` |
| Q3 | Which ten EV model lines have the most listings? | `Q3_Top10_Models_by_Listings.csv` |
| Q4 | Which model lines have the highest complaints per listing? | `Q4_Complaints_by_Model.csv` |
| Q5 | What proportion of listings had a price cut by brand? | `Q5_Price_Cut_Rate_by_Brand.csv` |
| Q6 | How does average listing price vary by seller-rating group? | `Q6_Price_by_Seller_Rating.csv` |

These results are descriptive and do not establish causal relationships. The Q4 rate is based on complaint and listing counts, not a verified vehicle failure probability.

## 7. Handoffs and dependencies

**M2 and M5:** Use `model_dataset.csv` as the extracted source; M5 owns the common `train_ids.csv`, `test_ids.csv`, and `src/prep.py`. Confirm the exact input columns before training.

**M4:** Use the EER diagram for Slide 7 and select useful SQL results for the business findings. Do not report descriptive comparisons as causal model effects.

**M3 next phase:** Obtain `price_model.joblib` and final preprocessing pipeline from M5, and `price_cut_model.joblib` from M2. Build and test FastAPI locally, then Docker, GitHub and AWS deployment. Add API test evidence, deployment URL, logging and drift-monitoring documentation when ready.

## 8. Remaining checklist

- [ ] Confirm `model_dataset.csv` has 15,155 rows and unique `listing_key`.
- [ ] Record actual fact-table and foreign-key validation results.
- [ ] Rename `answer_for_q3_business_question` to `Business_Query_Results`.
- [ ] Confirm `.sql`, `.csv`, `.png`, and `.mwb` extensions and provide clear filenames.
- [ ] Add the deployment source, model integration tests, Dockerfile and architecture diagram when available.
- [ ] Add GitHub repository link and cloud API URL after deployment.
