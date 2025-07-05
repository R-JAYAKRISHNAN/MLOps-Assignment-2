FROM python:3.10

WORKDIR /codebase

COPY . /codebase

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
