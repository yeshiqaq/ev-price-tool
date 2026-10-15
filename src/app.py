
from pathlib import Path
import logging
import time

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ----------------------------------
# 1. Project paths
# ----------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

# ----------------------------------
# 2. Create FastAPI
# ----------------------------------
app = FastAPI(title="EV Fair Price API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ev-api")

# ----------------------------------
# 3. Load model and reliability data
# ----------------------------------
price_model = joblib.load(
    MODEL_DIR / "price_model_v1.joblib"
)

rel = pd.read_csv(
    MODEL_DIR / "dim_reliability.csv"
)

# ----------------------------------
# 4. Define user input
# ----------------------------------
class CarInput(BaseModel):
    make: str
    model_line: str
    model_year: int = Field(ge=2010, le=2023)
    mileage: float = Field(ge=0, le=250000)
    drivetrain: str

    trim_name: str | None = None
    seller_rating: float | None = None
    driver_rating: float | None = None

    accidents_or_damage: int = Field(0, ge=0, le=1)
    one_owner: int = Field(1, ge=0, le=1)
    personal_use_only: int = Field(1, ge=0, le=1)


# ----------------------------------
# 5. Home endpoint
# ----------------------------------
@app.get("/")
def home():
    return {
        "message": "EV Price Prediction API is running"
    }


# ----------------------------------
# 6. Health endpoint
# ----------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "price_model_v1"
    }


# ----------------------------------
# 7. Prediction endpoint
# ----------------------------------
@app.post("/api/v1/used-ev/price")
def predict_price(car: CarInput):

    start = time.perf_counter()

    # Find reliability features
    match = rel[
        (rel["make"] == car.make)
        & (rel["model_line"] == car.model_line)
        & (rel["year"] == car.model_year)
    ]

    if match.empty:
        raise HTTPException(
            status_code=400,
            detail="Unsupported make, model or year"
        )

    # Vehicle age based on April 2023 data
    age = 2023 - car.model_year

    premium_brands = {
        "Tesla", "Porsche", "BMW",
        "Mercedes-Benz", "Audi", "Jaguar",
        "Polestar", "Lucid", "Rivian", "Genesis"
    }

    # Construct model input
    row = {
        "make": car.make,
        "model_line": car.model_line,
        "trim_name": car.trim_name,
        "drivetrain": car.drivetrain,
        "year": car.model_year,
        "age": age,
        "mileage": car.mileage,
        "miles_per_year": car.mileage / max(age, 0.5),
        "premium_brand_flag": int(
            car.make in premium_brands
        ),
        "seller_rating": car.seller_rating,
        "driver_rating": car.driver_rating,
        "accidents_or_damage": car.accidents_or_damage,
        "one_owner": car.one_owner,
        "personal_use_only": car.personal_use_only
    }

    # Add complaint features
    complaint_cols = [
        "complaints_per_listing",
        "avg_sentiment",
        "battery_charging_share",
        "braking_assist_share",
        "software_screen_share",
        "crash_fire_share"
    ]

    for col in complaint_cols:
        row[col] = match.iloc[0][col]

    X = pd.DataFrame([row])

    # Run model prediction
    try:
        # Model already returns price in USD
        prediction = float(price_model.predict(X)[0])

        # Validate prediction
        if not np.isfinite(prediction) or prediction <= 0:
            raise ValueError(
                f"Invalid predicted price: {prediction}"
            )

        fair_price = prediction

    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Model prediction failed"
        ) from exc

    # Measure response time
    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    logger.info(
        "model=%s, year=%s, price=%.2f, latency=%.2fms",
        car.model_line,
        car.model_year,
        fair_price,
        elapsed_ms
    )

    return {
        "fair_price_usd": round(fair_price, 2),
        "market_basis": "April 2023",
        "model_version": "v1"
    }

from mangum import Mangum

handler = Mangum(app)