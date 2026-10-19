# PriceSpark — EV Fair Price & Price-Cut Review API

**Group 21 | EBA5012 Predictive Analytics Lifecycle (PAL)**  
**Repository:** https://github.com/yeshiqaq/ev-price-tool  
**Deployment status:** Live on AWS Lambda; dual-model endpoint tested (October 2026)  
**Region:** `ap-southeast-1` (Singapore)  
**Deployment image tag:** `v3-cut`

## 1. Overview

PriceSpark is a decision-support prototype for U.S. used-electric-vehicle (EV) dealers. It provides a reference **fair asking price** and, when a planned asking price is supplied, a **price-cut review score and review flag**. The models reflect the **April 2023 U.S. listing market**, not current transaction prices.

The service combines:

- **M5 Fair Price model:** an XGBoost-based v3 price predictor using vehicle details and model-year complaint/reliability features.
- **M2 Price-cut review model:** a decision-tree classifier and scoring helper using the planned asking price and the model's reference fair price.

**Important:** The price-cut review score is **not a calibrated probability** that a future price cut will occur. It is intended to support human review, not automated repricing.

## 2. Live API

| Item | Address / value |
|---|---|
| AWS Lambda function | `ev-price-predictor` |
| Base URL | https://ncnbucti2my7ygmwzpdf5vq3b40etqnp.lambda-url.ap-southeast-1.on.aws/ |
| Interactive Swagger UI | https://ncnbucti2my7ygmwzpdf5vq3b40etqnp.lambda-url.ap-southeast-1.on.aws/docs |
| Health check | `GET /health` |
| Prediction | `POST /api/v1/used-ev/price` |
| API region | `ap-southeast-1` |

The API is exposed directly through an **AWS Lambda Function URL**. **API Gateway is not part of the current deployment**; it is a possible future enhancement. The Function URL can remain unchanged when the Lambda container image is updated, provided the URL configuration is retained.

> This is a public-facing prototype endpoint. Availability, access policy, costs and the deployed version may change; confirm the live service before a demonstration.

## 3. Architecture

```text
Client / Swagger UI
       |
       | HTTPS JSON request
       v
AWS Lambda Function URL
       |
       v
AWS Lambda container (Python 3.13)
       |
       v
FastAPI + Mangum
       |
       +--> Pydantic input validation
       +--> model-year reliability lookup
       +--> M5 fair-price model (XGBoost v3)
       +--> M2 price-cut scoring helper / decision tree
       |
       v
JSON response: fair price + review score / flag

Amazon ECR: container image storage
Amazon CloudWatch: Lambda execution logs and metrics
GitHub: application, Docker and SQL source files
```

The application uses separate Dockerfiles for a conventional Uvicorn container and the AWS Lambda container runtime.

## 4. API input and output

The prediction endpoint accepts vehicle attributes such as make, model line, year, mileage, drivetrain, trim, seller/driver ratings, and vehicle-history flags. `asking_price` is the dealer's proposed listing price in USD.

**Input checks implemented in the API:** model year **2010–2023**, mileage **0–250,000**, binary history flags, positive asking price when supplied, and a supported make/model/year reliability lookup. Optional rating and trim fields can be `null`; missing values can materially change the prediction.

### Example request

```json
{
  "make": "Tesla",
  "model_line": "Tesla Model 3",
  "model_year": 2020,
  "mileage": 35000,
  "drivetrain": "All-wheel Drive",
  "trim_name": "Model 3 Long Range",
  "seller_rating": 4.5,
  "driver_rating": 4.2,
  "accidents_or_damage": 0,
  "one_owner": 1,
  "personal_use_only": 1,
  "asking_price": 42000
}
```

### Verified example response (AWS Swagger test)

The deployed API returned **HTTP 200** for the above request. Key fields were:

```json
{
  "fair_price_usd": 38814.0,
  "asking_price_usd": 42000.0,
  "ask_gap_pct": 0.08208378420157675,
  "price_cut_review_score": 0.620093469415517,
  "price_cut_review_flag": true,
  "price_cut_threshold": 0.4821645328711008,
  "reliability_matched": true,
  "price_model_version": "v3"
}
```

The `ask_gap_pct` field is a **fractional ratio**: `0.08208` means approximately **8.21%** above the predicted fair price. `price_cut_review_flag` indicates that the score meets the review threshold; it is not a recommendation to reduce the asking price automatically.

### Missing-rating sensitivity

For the same Tesla example, changing only `seller_rating` and `driver_rating` from `4.5`/`4.2` to `null` changed the locally observed fair-price prediction from **$38,814** to **$54,011**. This demonstrates input sensitivity; it does **not** prove which field caused the change or which valuation is more accurate. Use verified ratings where available, and do not insert invented values simply to obtain a preferred price.

**Pending independent check:** the original M5 reference case (2020 Tesla Model 3, 35,000 miles, AWD, blank trim and missing ratings) still needs a documented like-for-like comparison with M5's reference output. The example above is **not** that reference case.

## 5. Project structure

The following is the **intended final deployment layout**. Check the actual repository before publishing and update names/paths if the team has organized files differently.

```text
ev-price-tool/
├── src/
│   ├── __init__.py
│   ├── app.py                 # FastAPI routes and Mangum handler
│   ├── server.py              # Local server entry point, if used
│   ├── prep.py                # Shared M5 preprocessing
│   ├── condition_features.py  # Shared feature definitions
│   └── price_cut.py           # M2 scoring helper
├── models/
│   ├── price_model_v3.joblib
│   ├── price_cut_model.joblib
│   ├── price_cut_metadata.json
│   └── dim_reliability.csv
├── data/                      # Local deployment data; see privacy notes
├── sql/                       # SQL scripts; see Section 9
├── Dockerfile
├── Dockerfile.lambda
├── requirements-serve.txt
├── test_api.py
├── test_model.py
├── .dockerignore
├── .gitignore
└── README.md
```

The Lambda Dockerfile must copy every runtime path required by the scoring helper. In the tested integration this includes `src/`, `models/`, and `data/` as applicable. The precise model artifacts and permitted distribution should be confirmed with their owners before committing them to a public repository.

## 6. Runtime and dependencies

The deployed container uses **Python 3.13** and the team's pinned serving requirements, including:

| Package | Version |
|---|---|
| Python | 3.13 |
| scikit-learn | 1.8.0 |
| XGBoost | 2.1.1 |
| pandas | 2.2.3 |
| NumPy | 2.3.5 |
| joblib | 1.6.0 |

Additional serving dependencies, including **FastAPI**, **Pydantic**, **Mangum**, and their compatible transitive dependencies, are specified in `requirements-serve.txt`. Use that file as the authoritative install list rather than mixing it with the broader training/analysis `requirements.txt`.

**Why versions matter:** the original cloud build used scikit-learn 1.7.2 with a model saved under 1.8.0. An API can return HTTP 200 while still producing an unexpected model output. Keep model artifacts, preprocessing and library versions aligned, and validate predictions against reference cases.

## 7. Run locally with Docker

From the repository root, with Docker Desktop running:

```bash
docker build --no-cache -t ev-price-api:v3-cut .
docker run --rm -p 8002:8000 --name ev-price-v3-cut ev-price-api:v3-cut
```

Open **http://localhost:8002/docs**, expand `POST /api/v1/used-ev/price`, click **Try it out**, paste the example JSON, and click **Execute**.

If port `8002` is already in use, choose another free host port (for example `8003:8000`). If the container name already exists, remove or rename the old test container before starting another one.

## 8. AWS Lambda container deployment

The project uses **Amazon ECR** to store a Linux `amd64` Lambda-compatible container image, then deploys that image to the existing `ev-price-predictor` function.

### Build the Lambda image

```bash
docker buildx build --platform linux/amd64 --provenance=false --load \
  -f Dockerfile.lambda -t ev-price-lambda:v3-cut .
```

### Test Lambda locally

```bash
docker run --rm -d -p 9002:8080 \
  --name ev-lambda-v3-cut ev-price-lambda:v3-cut
```

Invoke the Lambda Runtime Interface Emulator using a properly formatted API Gateway HTTP API v2 / Lambda Function URL event saved as `lambda_event_cut.json`:

```bash
curl -X POST \
  http://localhost:9002/2015-03-31/functions/function/invocations \
  -H 'Content-Type: application/json' \
  --data-binary @lambda_event_cut.json
```

The tested local Lambda invocation returned a payload with `statusCode: 200` and matching dual-model results. Check logs with `docker logs ev-lambda-v3-cut` if an invocation fails.

### Push to Amazon ECR

The following example uses a Unix-like shell. On Windows **CMD**, use `set` and `%VARIABLE%` instead of `export` and `$VARIABLE`.

```bash
export AWS_REGION=ap-southeast-1
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/ev-price-api"

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker tag ev-price-lambda:v3-cut "${ECR_URI}:v3-cut"
docker push "${ECR_URI}:v3-cut"
```

The AWS identity must have the necessary ECR permissions. Never put AWS access keys, session tokens, or account credentials into this repository.

### Update the existing Lambda function

In **AWS Console → Lambda → `ev-price-predictor` → Code → Deploy new image**, select the ECR image `ev-price-api:v3-cut` and deploy. Confirm the deployed image **digest**, not just a human-readable model-version string. The existing Lambda Function URL remains available unless its configuration is changed or removed.

Finally, test the **live** `/docs` page with the same JSON and compare the response to the local Docker and Lambda-container outputs.

## 9. Data warehouse and SQL

The M3 data engineering component uses MySQL database **`ev_dw`** with a star schema at **one listing per fact row**:

- Fact table: `fact_listing`
- Dimensions: `dim_vehicle`, `dim_model_year`, `dim_seller`, `dim_history`, `dim_reliability`
- Modelling view: `v_model_dataset`

The final report documents **15,155 fact listings**, **15,155 modelling-view rows**, **zero duplicate listing keys**, and **zero invalid foreign-key references**. These are the reported warehouse-validation results; rerun the SQL against the delivered database when reproducing the project.

### SQL deliverables

| File | Purpose |
|---|---|
| `sql/01_create_tables.sql` | Create the database and star-schema tables |
| `sql/02_load.sql` | Load staging data into dimension and fact tables |
| `sql/03_business_questions.sql` | Six descriptive business-analysis queries |
| `sql/04_extract.sql` | Create/query `v_model_dataset` for modelling |
| `sql/05_validation.sql` | Reconcile row counts, uniqueness and foreign keys |

**Important:** These filenames are the final-report handover specification. Verify that each file actually exists in GitHub; the README does not create the SQL scripts. Staging CSVs, raw datasets and database credentials are not bundled in this public repository.

To reproduce the warehouse, obtain the authorized source files from the team, create the schema, import the staging tables, run the loading script once, validate counts and keys, and then run the business queries and model extract. See the M3 handover notes for staging-column and import details.

## 10. Testing and monitoring

| Test | Evidence / current state |
|---|---|
| Standard Docker FastAPI | Tested; HTTP 200 with both model outputs |
| Local Lambda runtime | Tested; `statusCode: 200` |
| AWS Lambda Function URL | Tested; HTTP 200 with both model outputs |
| Same-input local vs cloud output | Matched for the documented Tesla example |
| AWS CloudWatch | Invocation logs and execution metrics visible |
| M5 original missing-value reference comparison | Awaiting documented confirmation |
| Automated data-drift detection | **Planned, not implemented** |
| Production-grade authentication/rate limiting | **Not implemented in this prototype** |

**Deployment evidence:** the team stores `M3_AWS_DualModel_Swagger.png` and `M3_AWS_CloudWatch_v3cut.png` in **OneDrive → M3 → AWS_Deployment_Oct20**. The final report's **Section 14.4** and **Appendix F** contain or reference the deployment validation evidence.

CloudWatch confirms that Lambda was invoked and records metrics such as duration and memory use. It does **not**, by itself, establish that a prediction is accurate or that automatic drift monitoring has been implemented.

## 11. Known limitations and future work

1. **Historical asking prices:** the reference price is learned from April 2023 listings, not verified sale prices or today's market.
2. **Missing values:** seller/driver ratings and trim can change model predictions substantially; input sensitivity requires further investigation.
3. **Price-cut score:** the classifier has modest discrimination (test ROC-AUC approximately **0.630**, recall **73.77%**, precision **70.40%**) and its output is **uncalibrated**. The 80% test-recall target was not met.
4. **Data coverage:** the project covers U.S. used battery-electric listings and mapped model-year complaint features; unsupported vehicles should not be treated as valid predictions.
5. **Battery health:** direct battery state-of-health measurements are unavailable.
6. **Operational hardening:** add API Gateway or another access-control layer as appropriate, rate limiting, structured monitoring, automated drift checks, CI/CD and updated data before production use.

## 12. Project ownership and reproducibility

| Member | Main area |
|---|---|
| M1 Zihao | Listing cleaning, data dictionary and profiling |
| M2 Ruijing | Complaint matching and price-cut model |
| M3 Zhishuo | MySQL warehouse, SQL, FastAPI integration, Docker and AWS Lambda deployment |
| M4 Caleb | Project coordination, reporting and complaint text analytics |
| M5 Hyunbin | Fair-price model, shared preprocessing, evaluation and MLflow model registry |

Model training, evaluation, feature engineering and cloud deployment are separate stages. The public API demonstrates **inference**, not that a model was trained in Lambda. Model registry status and artifact ownership should be verified with M5 before making additional registry claims.

**Security:** do not commit AWS keys, `.env` files, raw restricted datasets, database credentials, or model artifacts without the owner's permission. Review `.gitignore` and `.dockerignore` before pushing changes.

## 13. References

- Team PriceSpark (2026), *What Drives EV Resale Value?*, Group 21 PAL final report.
- [Project GitHub repository](https://github.com/yeshiqaq/ev-price-tool)
- [AWS Lambda documentation](https://docs.aws.amazon.com/lambda/)
- [FastAPI documentation](https://fastapi.tiangolo.com/)

---

**Project status summary:** Dual-model inference is live and smoke-tested on AWS Lambda. Independent M5 reference-case reconciliation and additional validation remain necessary before treating this as a fully validated dealer-facing production service.
