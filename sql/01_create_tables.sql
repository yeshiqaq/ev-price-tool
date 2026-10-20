CREATE DATABASE ev_dw; USE ev_dw; 
CREATE TABLE dim_vehicle (vehicle_key INT AUTO_INCREMENT PRIMARY KEY, make VARCHAR(40), 
  model_line VARCHAR(60), trim_name VARCHAR(120), drivetrain VARCHAR(50), premium_brand_flag TINYINT); 
CREATE TABLE dim_model_year (model_year_key INT AUTO_INCREMENT PRIMARY KEY, model_year INT, age INT, age_band VARCHAR(10)); 
CREATE TABLE dim_seller (seller_key INT AUTO_INCREMENT PRIMARY KEY, seller_name VARCHAR(150), seller_rating DECIMAL(3,1)); 
CREATE TABLE dim_history (history_key INT AUTO_INCREMENT PRIMARY KEY, accidents_or_damage TINYINT, 
  one_owner TINYINT, personal_use_only TINYINT); 
CREATE TABLE dim_reliability (reliability_key INT AUTO_INCREMENT PRIMARY KEY, make VARCHAR(40), model_line VARCHAR(60), 
  model_year INT, summary_level VARCHAR(12), complaints_per_listing DECIMAL(8,4), avg_sentiment DECIMAL(6,4), 
  battery_charging_share DECIMAL(5,4), braking_assist_share DECIMAL(5,4), software_screen_share DECIMAL(5,4), 
  crash_fire_share DECIMAL(5,4)); 
CREATE TABLE fact_listing (listing_key INT PRIMARY KEY, 
  vehicle_key INT, model_year_key INT, seller_key INT, history_key INT, reliability_key INT, 
  price DECIMAL(10,2), initial_price DECIMAL(10,2), price_drop DECIMAL(10,2), price_cut_flag TINYINT, 
  mileage INT, miles_per_year DECIMAL(10,1), driver_rating DECIMAL(3,1), 
  FOREIGN KEY (vehicle_key) REFERENCES dim_vehicle(vehicle_key), 
  FOREIGN KEY (model_year_key) REFERENCES dim_model_year(model_year_key), 
  FOREIGN KEY (seller_key) REFERENCES dim_seller(seller_key), 
  FOREIGN KEY (history_key) REFERENCES dim_history(history_key), 
  FOREIGN KEY (reliability_key) REFERENCES dim_reliability(reliability_key));