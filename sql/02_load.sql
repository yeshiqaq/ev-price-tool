INSERT INTO dim_vehicle (make, model_line, trim_name, drivetrain, premium_brand_flag) 
SELECT DISTINCT manufacturer, model_line, model, drivetrain, premium_brand_flag FROM stg_listings; 
-- same pattern for dim_model_year, dim_seller, dim_history, dim_reliability 
INSERT INTO fact_listing 
SELECT s.listing_id, v.vehicle_key, y.model_year_key, se.seller_key, h.history_key, r.reliability_key, 
       s.price, s.initial_price, s.price_drop, s.price_cut_flag, s.mileage, s.miles_per_year, s.driver_rating 
FROM stg_listings s 
JOIN dim_vehicle v ON v.make = s.manufacturer AND v.trim_name = s.model AND v.drivetrain <=> s.drivetrain 
JOIN dim_model_year y ON y.model_year = s.year 
JOIN dim_seller se ON se.seller_name <=> s.seller_name AND se.seller_rating <=> s.seller_rating 
JOIN dim_history h ON h.accidents_or_damage <=> s.accidents_or_damage AND h.one_owner <=> s.one_owner 
                  AND h.personal_use_only <=> s.personal_use_only 
LEFT JOIN dim_reliability r ON r.model_line = s.model_line AND r.model_year = s.year;






-- INSERT dim_seller
USE ev_dw;

INSERT INTO dim_seller (
    seller_name,
    seller_rating
)
SELECT DISTINCT
    seller_name,
    seller_rating
FROM stg_listings;


SELECT COUNT(*) FROM dim_seller;

SELECT * FROM dim_seller LIMIT 10;


-- INSERT dim_history
INSERT INTO dim_history (
    accidents_or_damage,
    one_owner,
    personal_use_only
)
SELECT DISTINCT
    accidents_or_damage,
    one_owner,
    personal_use_only
FROM stg_listings;





-- INSERT dim_reliability
INSERT INTO dim_reliability (
    make,
    model_line,
    model_year,
    summary_level,
    complaints_per_listing,
    avg_sentiment,
    battery_charging_share,
    braking_assist_share,
    software_screen_share,
    crash_fire_share
)
SELECT
    make,
    model_line,
    year,
    summary_level,

    CAST(NULLIF(TRIM(complaints_per_listing), '') AS DECIMAL(12,4)),
    CAST(NULLIF(TRIM(avg_sentiment), '') AS DECIMAL(8,4)),
    CAST(NULLIF(TRIM(battery_charging_share), '') AS DECIMAL(8,4)),
    CAST(NULLIF(TRIM(braking_assist_share), '') AS DECIMAL(8,4)),
    CAST(NULLIF(TRIM(software_screen_share), '') AS DECIMAL(8,4)),
    CAST(NULLIF(TRIM(crash_fire_share), '') AS DECIMAL(8,4))

FROM stg_reliability;







-- INSERT fact_listing
INSERT INTO fact_listing (
    listing_key,
    vehicle_key,
    model_year_key,
    seller_key,
    history_key,
    reliability_key,
    price,
    initial_price,
    price_drop,
    price_cut_flag,
    mileage,
    miles_per_year,
    driver_rating
)
SELECT
    s.listing_id,
    v.vehicle_key,
    y.model_year_key,
    se.seller_key,
    h.history_key,
    r.reliability_key,
    s.price,
    s.initial_price,
    s.price_drop,
    s.price_cut_flag,
    s.mileage,
    s.miles_per_year,
    s.driver_rating

FROM stg_listings s

JOIN dim_vehicle v
    ON v.make <=> s.manufacturer
    AND v.model_line <=> s.model_line
    AND v.trim_name <=> s.model
    AND v.drivetrain <=> s.drivetrain
    AND v.premium_brand_flag <=> s.premium_brand_flag

JOIN dim_model_year y
    ON y.model_year <=> s.year
    AND y.age <=> s.age

JOIN dim_seller se
    ON se.seller_name <=> s.seller_name
    AND se.seller_rating <=> s.seller_rating

JOIN dim_history h
    ON h.accidents_or_damage <=> s.accidents_or_damage
    AND h.one_owner <=> s.one_owner
    AND h.personal_use_only <=> s.personal_use_only

LEFT JOIN dim_reliability r
    ON r.make = s.manufacturer
    AND r.model_line = s.model_line
    AND r.model_year = s.year;
    
    
    SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT listing_key) AS unique_ids,
    SUM(vehicle_key IS NULL) AS missing_vehicle,
    SUM(model_year_key IS NULL) AS missing_year,
    SUM(seller_key IS NULL) AS missing_seller,
    SUM(history_key IS NULL) AS missing_history,
    SUM(reliability_key IS NULL) AS missing_reliability
FROM fact_listing;


USE ev_dw;

SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT listing_key) AS unique_ids,
    SUM(vehicle_key IS NULL) AS missing_vehicle,
    SUM(model_year_key IS NULL) AS missing_year,
    SUM(seller_key IS NULL) AS missing_seller,
    SUM(history_key IS NULL) AS missing_history,
    SUM(reliability_key IS NULL) AS missing_reliability
FROM fact_listing;