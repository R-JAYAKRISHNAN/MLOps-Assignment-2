FROM python:3.10

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY ./codebase /app/codebase

EXPOSE 8000

CMD ["uvicorn", "codebase.service:app", "--host", "0.0.0.0", "--port", "8000"]
