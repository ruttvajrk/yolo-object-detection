# ---- Wildlife counting API (ONNX Runtime, no PyTorch) ----
# Build:  docker build -t wildlife-api .
# Run:    docker run -p 8000:8000 wildlife-api
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
RUN useradd -m -u 1000 app
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY detector_onnx.py api.py count_animals.py ./
COPY models/best.onnx ./models/best.onnx
COPY reports/evaluation.json ./reports/evaluation.json
RUN chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')" || exit 1
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
