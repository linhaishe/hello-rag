FROM python:3.11-slim

WORKDIR /app

COPY requirements-new.txt .
RUN pip install --no-cache-dir -r requirements-new.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
