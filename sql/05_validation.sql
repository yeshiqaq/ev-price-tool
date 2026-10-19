SHOW DATABASES;
USE ev_dw;

SHOW TABLES;

USE ev_dw;

SELECT 'stg_listings' AS table_name, COUNT(*) AS actual_rows
FROM stg_listings
UNION ALL
SELECT 'dim_vehicle', COUNT(*) FROM dim_vehicle
UNION ALL
SELECT 'dim_model_year', COUNT(*) FROM dim_model_year
UNION ALL
SELECT 'dim_seller', COUNT(*) FROM dim_seller
UNION ALL
SELECT 'dim_history', COUNT(*) FROM dim_history
UNION ALL
SELECT 'dim_reliability', COUNT(*) FROM dim_reliability
UNION ALL
SELECT 'fact_listing', COUNT(*) FROM fact_listing;



SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT listing_key) AS unique_listings,
    COUNT(*) - COUNT(DISTINCT listing_key) AS duplicate_count
FROM fact_listing;


SELECT COUNT(*) AS missing_listings
FROM stg_listings s
LEFT JOIN fact_listing f
    ON s.listing_id = f.listing_key
WHERE f.listing_key IS NULL;


SELECT
    SUM(v.vehicle_key IS NULL) AS invalid_vehicle_fk,
    SUM(y.model_year_key IS NULL) AS invalid_year_fk,
    SUM(s.seller_key IS NULL) AS invalid_seller_fk,
    SUM(h.history_key IS NULL) AS invalid_history_fk,
    SUM(
        f.reliability_key IS NOT NULL
        AND r.reliability_key IS NULL
    ) AS invalid_reliability_fk
FROM fact_listing f
LEFT JOIN dim_vehicle v
    ON f.vehicle_key = v.vehicle_key
LEFT JOIN dim_model_year y
    ON f.model_year_key = y.model_year_key
LEFT JOIN dim_seller s
    ON f.seller_key = s.seller_key
LEFT JOIN dim_history h
    ON f.history_key = h.history_key
LEFT JOIN dim_reliability r
    ON f.reliability_key = r.reliability_key;