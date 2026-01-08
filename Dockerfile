# Artemis Signals Backend Dockerfile
# For Shivaansh & Krishaansh — this line pays your fees!
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (if any)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default env (can be overridden)
ENV ENVIRONMENT=local

# Run Uvicorn server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
