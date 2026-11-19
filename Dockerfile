
FROM python:3.11-slim

WORKDIR /app

# Install application dependencies
COPY requirements-docker.txt .

RUN pip install --no-cache-dir \
    -r requirements-docker.txt

# Copy application source
COPY src/ ./src/

# Copy trained fraud detection model
COPY models/random_forest_optimized.joblib \
    ./models/random_forest_optimized.joblib

# Make Python package importable
ENV PYTHONPATH=/app/src

# Document the application port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "fraudlens.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]

