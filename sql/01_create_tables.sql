CREATE DATABASE IF NOT EXISTS ev_dw;
USE ev_dw;

-- 1. Vehicle dimension
CREATE TABLE dim_vehicle (
    vehicle_key INT AUTO_INCREMENT PRIMARY KEY,
    make VARCHAR(40),
    model_line VARCHAR(150),
    trim_name VARCHAR(255),
    drivetrain VARCHAR(100),
    premium_brand_flag TINYINT
);

-- 2. Model year dimension
CREATE TABLE dim_model_year (
    model_year_key INT AUTO_INCREMENT PRIMARY KEY,
    model_year INT,
    age INT,
    age_band VARCHAR(20)
);

-- 3. Seller dimension
CREATE TABLE dim_seller (
    seller_key INT AUTO_INCREMENT PRIMARY KEY,
    seller_name VARCHAR(150),
    seller_rating DECIMAL(3,1)
);

-- 4. Vehicle history dimension
CREATE TABLE dim_history (
    history_key INT AUTO_INCREMENT PRIMARY KEY,
    accidents_or_damage TINYINT,
    one_owner TINYINT,
    personal_use_only TINYINT
);

-- 5. Reliability dimension
CREATE TABLE dim_reliability (
    reliability_key INT AUTO_INCREMENT PRIMARY KEY,
    make VARCHAR(40),
    model_line VARCHAR(100),
    model_year INT,
    summary_level VARCHAR(20),
    complaints_per_listing DECIMAL(12,4),
    avg_sentiment DECIMAL(8,4),
    battery_charging_share DECIMAL(8,4),
    braking_assist_share DECIMAL(8,4),
    software_screen_share DECIMAL(8,4),
    crash_fire_share DECIMAL(8,4)
);

-- 6. Fact table
CREATE TABLE fact_listing (
    listing_key INT PRIMARY KEY,
    vehicle_key INT,
    model_year_key INT,
    seller_key INT,
    history_key INT,
    reliability_key INT,

    price DECIMAL(12,2),
    initial_price DECIMAL(12,2),
    price_drop DECIMAL(12,2),
    price_cut_flag TINYINT,
    mileage INT,
    miles_per_year DECIMAL(12,1),
    driver_rating DECIMAL(3,1),

    FOREIGN KEY (vehicle_key)
        REFERENCES dim_vehicle(vehicle_key),

    FOREIGN KEY (model_year_key)
        REFERENCES dim_model_year(model_year_key),

    FOREIGN KEY (seller_key)
        REFERENCES dim_seller(seller_key),

    FOREIGN KEY (history_key)
        REFERENCES dim_history(history_key),

    FOREIGN KEY (reliability_key)
        REFERENCES dim_reliability(reliability_key)
);

USE ev_dw;

SHOW TABLES;

DESCRIBE fact_listing;
DESCRIBE dim_reliability;

SELECT COUNT(*) FROM stg_listings;
SELECT COUNT(*) FROM stg_reliability;

SELECT * FROM stg_listings LIMIT 5;
SELECT * FROM stg_reliability LIMIT 5;





USE ev_dw;

-- 检查实际行数
SELECT COUNT(*) AS total_rows
FROM stg_listings;

-- 检查 listing_id 是否重复
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT listing_id) AS unique_ids,
    MIN(listing_id) AS min_id,
    MAX(listing_id) AS max_id
FROM stg_listings;

-- 检查导入字段的类型
SHOW COLUMNS FROM stg_listings;
