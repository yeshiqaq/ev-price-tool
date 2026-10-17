# EV Resale Value Prediction Tool

**PAL Group 21 | M3 — Data Warehouse & Deployment**  
**Repository:** https://github.com/yeshiqaq/ev-price-tool  
**Project status:** Local FastAPI, Docker and Lambda-container inference tested; AWS cloud deployment pending verification.

## 1. Project overview

This project supports used-electric-vehicle (EV) pricing decisions. Its first deployed service estimates a vehicle's **fair price in USD**, based on the historical April 2023 market represented by the modelling data. The broader group project also aims to estimate the **probability of a listing price cut**, once the final M2 classifier and its preprocessing requirements have been integrated.

The API is a decision-support prototype, **not** a guaranteed current-market valuation, transaction price or production-grade pricing service. The model has been trained on historical listings; the fairness and accuracy of individual predictions must be assessed separately from successful API execution.

### Team interfaces

| Member | Contribution to the deployment workflow |
|---|---|
| M1 | Clean EV listings and data dictionary |
| M2 | Complaint matching and forthcoming price-cut classifier |
| M3 | MySQL star schema, SQL extraction, FastAPI, Docker, GitHub and cloud deployment |
| M4 | Model-year complaint/reliability features and presentation integration |
| M5 | Shared preprocessing (`src/prep.py`), price model, model evaluation and dependency guidance |

## 2. Current implementation status

| Component | Status | Evidence / note |
|---|---|---|
| MySQL `ev_dw` star schema and SQL handover | Completed in M3 handover | See OneDrive `M3_EV Handover` for SQL, EER diagram, validation results and modelling extract |
| Local FastAPI `/health` | Tested | HTTP 200 |
| Local fair-price POST inference | Tested | HTTP 200 |
| Standard Docker image `ev-price-api:1.0` | Built and tested | Swagger POST returned HTTP 200 |
| GitHub repository | Initial code pushed | `main` branch; subsequent Lambda changes require another push |
| Lambda-specific image `ev-price-lambda:1.0` | Built | Local Docker image |
| Local Lambda runtime invocation | Tested | HTTP response payload `statusCode: 200` |
| Amazon ECR image upload | Pending verification | No successful cloud upload evidenced yet |
| AWS Lambda cloud function | Pending verification | Local Lambda container is not a deployed AWS function |
| API Gateway HTTPS endpoint | Pending verification | Do not report a live URL until tested |
| M2 price-cut model integration | Pending | Requires final model, input schema and test cases |
| Input/performance drift analysis | Planned | Requires request data and/or labelled holdout data |

**Reference smoke-test case (not a model-accuracy claim):** A 2020 Tesla Model 3, 35,000 miles, all-wheel drive, with the example history flags and unspecified optional ratings/trim, returned `fair_price_usd: 56649.33` in both the standard Docker API and the locally invoked Lambda container. A single matching prediction does not establish broad version compatibility or valuation accuracy.

## 3. Architecture

```text
EV listing JSON
      |
      v
API Gateway HTTPS route [planned cloud entry point]
      |
      v
AWS Lambda container (Python runtime + Mangum)
      |
      v
FastAPI input validation
      |
      v
Model-year reliability lookup (dim_reliability.csv)
      |
      v
Feature construction + M5 shared preprocessing pipeline
      |
      v
Saved M5 price model (price_model_v1.joblib)
      |
      v
Fair-price JSON response + application logs -> CloudWatch [once deployed]
```

**Local development path:** Client -> Uvicorn -> FastAPI -> saved model.  
**Lambda path:** API Gateway event -> Mangum handler -> FastAPI -> saved model.  
The normal Docker image and Lambda image package the same application/model files but use different base images and entry points.

## 4. Repository layout

```text
EV_Deployment/
├── models/
│   ├── price_model_v1.joblib
│   └── dim_reliability.csv
├── src/
│   ├── __init__.py
│   ├── prep.py
│   ├── app.py
│   └── server.py
├── test_model.py
├── test_api.py
├── lambda_event.json              # optional, synthetic test event
├── Dockerfile                     # Uvicorn-based container
├── Dockerfile.lambda              # Lambda base image + Mangum
├── requirements.txt               # standard Docker dependencies
├── requirements-lambda.txt        # Lambda-specific dependency set
├── .gitignore
└── README.md
```

The **full group repository** is expected to add `sql/`, `notebooks/`, cleaned/model data and final M2/M5 artifacts when handed over. Do not assume these folders already exist in the initial deployment-only commit.

## 5. Data warehouse (M3 Part A)

**Database:** MySQL `ev_dw`. **Fact grain:** one row per EV listing.  
**Fact:** `fact_listing`. **Dimensions:** `dim_vehicle`, `dim_model_year`, `dim_seller`, `dim_history`, `dim_reliability`.

The warehouse work comprises:

1. `01_create_tables.sql` — star-schema tables, primary keys and foreign keys.
2. `02_load.sql` — staging-to-dimension/fact loading.
3. `03_business_questions.sql` — six management-analysis queries (e.g., price by brand/age, accident history, model counts, complaints, price-cut rate and seller ratings).
4. `04_extract.sql` — joined modelling view / CSV export for M2 and M5.

**Handover artifacts:** EER diagram, SQL scripts, business-query CSV outputs, `data_validation_results`, and `model_dataset.csv` are maintained in the shared OneDrive `PAL Group 21 / M3_EV Handover` folder. Warehouse row counts should be quoted from the final validation export, not from targets in the planning guide. Source and fact-table counts must be reconciled before a final validation claim.

## 6. Prerequisites and dependency compatibility

- Windows + VS Code, Python, Docker Desktop and Git for local development.
- AWS CLI v2 and an authorized AWS identity for ECR/Lambda/API Gateway deployment.
- M5's `src/prep.py` is required to unpickle the saved model; keep its module path unchanged.
- The joblib artifact must be loaded only from a trusted source.
- The tested local Python/Docker environment and the Lambda environment **do not have identical dependencies**. The Lambda build uses Python 3.12 and a separate `requirements-lambda.txt`; the ordinary Docker image uses Python 3.11.

**Compatibility warning:** A successful image build or HTTP 200 does not guarantee numerical equivalence for all vehicles. The saved model previously raised scikit-learn/XGBoost version warnings. Coordinate exact training/export versions with M5 and compare predictions on multiple representative records before accepting a final deployment. Preserve the original model and dependency manifests.

**Prediction scale:** The current saved model returns price-scale values directly (the smoke test returned approximately 56,649 USD). Do **not** apply another `np.exp()` in `src/app.py`. M5 should confirm the saved model's target-transform configuration; applying `np.exp()` twice can overflow and cause HTTP 500.

## 7. Run FastAPI locally

From the project root in Windows CMD:

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m src.server
```

Open `http://127.0.0.1:8000/docs` for Swagger UI, or `http://127.0.0.1:8000/health` for the health check. Keep the server terminal running while testing in a second terminal.

**Prediction endpoint:** `POST /api/v1/used-ev/price`.

```json
{
  "make": "Tesla",
  "model_line": "Tesla Model 3",
  "model_year": 2020,
  "mileage": 35000,
  "drivetrain": "All-wheel Drive",
  "trim_name": null,
  "seller_rating": null,
  "driver_rating": null,
  "accidents_or_damage": 0,
  "one_owner": 1,
  "personal_use_only": 1
}
```

Example observed response (from the existing smoke test, not a guaranteed output for a future model version):

```json
{
  "fair_price_usd": 56649.33,
  "market_basis": "April 2023",
  "model_version": "v1"
}
```

The actual input must match the saved pipeline's feature requirements. Invalid years/mileage may return HTTP 422; unsupported model-year combinations may return HTTP 400; inference failures should return a controlled HTTP 500 and be diagnosed from server logs.

## 8. Standard Docker (local API)

```cmd
docker build -t ev-price-api:1.0 .
docker run -d -p 8000:8000 --name ev-price-container ev-price-api:1.0
docker ps
docker logs ev-price-container
```

Open `http://localhost:8000/docs` and run the same POST test. Stop any local Uvicorn process using port 8000 first. To stop the container:

```cmd
docker stop ev-price-container
```

The ordinary `Dockerfile` uses `python:3.11-slim` and starts Uvicorn on port 8000.

## 9. Lambda container (local runtime test)

`src/app.py` exposes `handler = Mangum(app)`. The Lambda image uses `public.ecr.aws/lambda/python:3.12`, installs `requirements-lambda.txt`, copies `src/` and `models/`, and sets `CMD ["src.app.handler"]`.

```cmd
docker build --platform linux/amd64 -f Dockerfile.lambda -t ev-price-lambda:1.0 .
docker run --rm -d -p 9000:8080 --name ev-lambda-test ev-price-lambda:1.0
docker ps
docker logs ev-lambda-test
```

With `lambda_event.json` containing a valid API Gateway HTTP API v2.0 event for the POST route, invoke the local Lambda runtime:

```cmd
curl.exe -X POST "http://localhost:9000/2015-03-31/functions/function/invocations" -H "Content-Type: application/json" --data-binary "@lambda_event.json"
```

**Observed local test:** The Lambda invocation returned `statusCode: 200` and a response body containing `fair_price_usd: 56649.33`. This proves the local Lambda runtime can execute that example request; it does **not** prove that an AWS Lambda function has been created.

If the image cannot start locally, check whether the Lambda Runtime Interface Emulator is available in the container/test setup, then inspect `docker logs ev-lambda-test`.

## 10. AWS cloud deployment — next actions

### 10.1 Authentication and region

Use AWS SSO/temporary credentials or another approved authentication method. Verify:

```cmd
aws --version
aws sts get-caller-identity
```

The examples use `ap-southeast-1` (Singapore). Replace the account ID placeholder with the output of `aws sts get-caller-identity --query Account --output text`.

### 10.2 ECR repository and image push

```cmd
aws ecr create-repository --repository-name ev-price-api --region ap-southeast-1
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com
docker tag ev-price-lambda:1.0 ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/ev-price-api:1.0
docker push ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/ev-price-api:1.0
```

If the repository already exists, skip creation. Ensure the pushed image is a Lambda-compatible, single-architecture Linux `amd64` image. If AWS rejects its manifest, rebuild with a compatible manifest (for example using Buildx with `--provenance=false --load`) and push again.

### 10.3 Lambda function

In AWS Lambda Console, create a **Container image** function named `ev-price-api`, select the ECR image, and use architecture `x86_64`. Give its execution role basic CloudWatch Logs permissions. Start with a test configuration such as 2048 MB memory and 30 seconds timeout; adjust based on actual runtime logs and cost.

Run the same API Gateway-shaped event in Lambda's **Test** tab. Do not proceed until the cloud function returns `statusCode: 200` and the expected price-scale response.

### 10.4 API Gateway HTTP API

Create an HTTP API integrated with the Lambda function. Configure at least:

- `POST /api/v1/used-ev/price`
- `GET /health` (optional but useful)

Deploy a stage and copy the actual HTTPS invoke URL. Test the POST route using the same JSON input as local Docker. **No live AWS URL is documented yet; add it only after a successful cloud test.**

Before exposing the endpoint broadly, configure suitable authentication/access restrictions, throttling and an AWS budget alert. AWS services may incur charges.

## 11. Logging, monitoring and drift

The application uses Python logging to record the model/version, vehicle model/year, predicted price and latency. In AWS Lambda, stdout/stderr application logs can be viewed through CloudWatch Logs once permissions are configured. Avoid logging secrets or unnecessary personal information.

**Operational monitoring:** requests, failures, latency, timeouts and memory usage.  
**Input drift (planned):** compare production/request distributions of age, mileage and model-line shares with training data.  
**Performance drift (planned):** compare predictions with actual later prices or an appropriately labelled held-back dataset, reporting MAE and percentage error. Without ground truth, performance drift cannot be measured.  
**Retraining proposal from the team guide:** review/retrain with newer listings and register a new version if average percentage error exceeds 10%; verify the metric definition and validation window with M5.

## 12. Validation checklist and evidence

- [x] FastAPI local `/health` test
- [x] FastAPI local POST test
- [x] Standard Docker image built
- [x] Standard Docker POST returned HTTP 200
- [x] Initial GitHub repository pushed
- [x] Lambda-specific Docker image built
- [x] Local Lambda invocation returned `statusCode: 200`
- [ ] Validate multiple representative records across both dependency environments
- [ ] Upload image to Amazon ECR
- [ ] Test cloud AWS Lambda function
- [ ] Test API Gateway HTTPS POST endpoint
- [ ] Capture CloudWatch logs
- [ ] Integrate final M2 price-cut classifier and verify its features
- [ ] Complete input/performance drift analysis
- [ ] Merge final notebooks, SQL and models from all group members

Suggested screenshot filenames:

```text
Docker_Image_Build.png
Docker_Container_Running.png
Docker_API_Prediction.png
Lambda_Local_Invocation.png
AWS_ECR_Image.png
AWS_Lambda_Test.png
AWS_API_Gateway.png
AWS_API_Prediction.png
AWS_CloudWatch_Logs.png
```

## 13. Limitations and pending integration

1. The demonstrated fair-price model is `price_model_v1.joblib`, not necessarily M5's final registered model.
2. The price-cut probability is **not yet implemented** in the deployed API; it requires M2's final artifact, shared feature list and validation.
3. Predictions reflect the April 2023 modelling context, not live 2026 EV prices.
4. The local Docker and Lambda dependency versions differ; a single matching result is insufficient to certify model reproducibility.
5. Reliability features require a supported make/model/year match; unsupported cases should return a clear error rather than silently substituting unrelated features.
6. A public GitHub repository should contain only files the group is authorized to share. Never commit AWS credentials, `.env`, private datasets or proprietary model artifacts without permission.

## 14. Team handover and project references

- **M3 shared deliverables:** OneDrive `PAL Group 21 / M3_EV Handover` (SQL, EER, data validation, modelling extract and deployment evidence).
- **Code repository:** https://github.com/yeshiqaq/ev-price-tool
- **M2/M5:** Use the extracted `model_dataset.csv` from the shared folder and the agreed split/preprocessing files. Confirm feature/schema changes with M3 before integrating a new saved model.
- **M4:** Use the EER diagram for slide 7 and architecture diagram, cloud endpoint and test evidence for slide 11 after cloud deployment succeeds.
- **Guide:** *EV Resale Value Project — Team Work Split & Step-by-Step Guide (v2)*, M3 Steps 1–14.

### Maintaining this README

Update status and evidence when each cloud step is verified. Replace version `v1` references only after integrating and testing the final model. Add the real AWS endpoint, deployment date, package versions, final model metrics and relevant screenshot links before final submission.
