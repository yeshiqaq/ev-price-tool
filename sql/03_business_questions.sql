
-- 1
SELECT
    v.make,
    y.age_band,
    COUNT(*) AS listing_count,
    ROUND(AVG(f.price), 2) AS avg_price
FROM fact_listing f
JOIN dim_vehicle v USING (vehicle_key)
JOIN dim_model_year y USING (model_year_key)
GROUP BY v.make, y.age_band
ORDER BY v.make, y.age_band;




-- 2
SELECT
    v.model_line,
    h.accidents_or_damage,
    COUNT(*) AS listing_count,
    ROUND(AVG(f.price), 2) AS avg_price
FROM fact_listing f
JOIN dim_vehicle v USING (vehicle_key)
JOIN dim_history h USING (history_key)
WHERE h.accidents_or_damage IS NOT NULL
GROUP BY v.model_line, h.accidents_or_damage
HAVING COUNT(*) >= 30
ORDER BY v.model_line;







-- 3
SELECT
    v.model_line,
    COUNT(*) AS listing_count
FROM fact_listing f
JOIN dim_vehicle v USING (vehicle_key)
GROUP BY v.model_line
ORDER BY listing_count DESC
LIMIT 10;








-- 4
SELECT
    make,
    model_line,
    SUM(n_complaints) AS total_complaints,
    SUM(n_listings) AS total_listings,
    ROUND(
        SUM(n_complaints) /
        NULLIF(SUM(n_listings), 0), 4
    ) AS complaints_per_listing
FROM stg_reliability
GROUP BY make, model_line
ORDER BY complaints_per_listing DESC;







-- 5
SELECT
    v.make,
    COUNT(*) AS total_listings,
    SUM(f.price_cut_flag) AS price_cut_count,
    ROUND(AVG(f.price_cut_flag) * 100, 2)
        AS price_cut_pct
FROM fact_listing f
JOIN dim_vehicle v USING (vehicle_key)
GROUP BY v.make
ORDER BY price_cut_pct DESC;












-- 6
SELECT
    CASE
        WHEN s.seller_rating IS NULL THEN 'Missing'
        WHEN s.seller_rating < 3 THEN 'Below 3'
        WHEN s.seller_rating < 4 THEN '3-4'
        ELSE '4-5'
    END AS rating_band,
    COUNT(*) AS listing_count,
    ROUND(AVG(f.price), 2) AS avg_price
FROM fact_listing f
JOIN dim_seller s USING (seller_key)
GROUP BY rating_band
ORDER BY rating_band;
