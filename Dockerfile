FROM python:3.11-slim
WORKDIR /app
COPY requirements-ci.txt .
RUN pip install --no-cache-dir -r requirements-ci.txt
COPY . .
RUN pip install --no-cache-dir --no-deps -e .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "flask_app:app"]
