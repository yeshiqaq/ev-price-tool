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