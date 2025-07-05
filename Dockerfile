# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Maintainer and Labels
LABEL maintainer="Trivendra Kumar Sahu, R Jayakrishnan"
LABEL version="1.0"
LABEL description="MLOps Assignment 2 (Team 5)- FastAPI service with ML model serving"

# Set working directory
WORKDIR /app

COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY ./codebase /app/codebase

EXPOSE 8000

CMD ["uvicorn", "codebase.service:app", "--host", "0.0.0.0", "--port", "8000"]
