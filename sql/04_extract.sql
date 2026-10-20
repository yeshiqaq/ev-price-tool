USE ev_dw;

CREATE OR REPLACE VIEW v_model_dataset AS
SELECT
    f.listing_key,

    v.make,
    v.model_line,
    v.trim_name,
    v.drivetrain,
    v.premium_brand_flag,

    y.model_year AS year,
    y.age,

    f.mileage,
    f.miles_per_year,

    h.accidents_or_damage,
    h.one_owner,
    h.personal_use_only,

    s.seller_rating,
    f.driver_rating,

    r.complaints_per_listing,
    r.avg_sentiment,
    r.battery_charging_share,
    r.braking_assist_share,
    r.software_screen_share,
    r.crash_fire_share,

    f.price,
    f.initial_price,
    f.price_cut_flag

FROM fact_listing f

JOIN dim_vehicle v
    USING (vehicle_key)

JOIN dim_model_year y
    USING (model_year_key)

JOIN dim_seller s
    USING (seller_key)

JOIN dim_history h
    USING (history_key)

LEFT JOIN dim_reliability r
    USING (reliability_key);
    
    
    
-- check
SELECT COUNT(*) AS total_rows
FROM v_model_dataset;

SELECT *
FROM v_model_dataset
LIMIT 10;




SELECT * FROM v_model_dataset;
