# syntax=docker/dockerfile:1

FROM python:3.10-slim

# Maintainer and Labels
LABEL maintainer="Trivendra Kumar Sahu, R Jayakrishnan"
LABEL version="1.0"
LABEL description="MLOps Assignment 2 - FastAPI service with ML model serving"

# Set working directory
WORKDIR /app

# Copy requirements first (for better cache)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the port
EXPOSE 8000

# Run the service
CMD ["uvicorn", "codebase.service:app", "--host", "0.0.0.0", "--port", "8000"]
