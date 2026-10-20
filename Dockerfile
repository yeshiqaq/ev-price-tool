
FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements-serve.txt .

RUN python -m pip install --no-cache-dir \
    -r requirements-serve.txt

# Copy application and model files
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Expose FastAPI port
EXPOSE 8000

# Start API server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
