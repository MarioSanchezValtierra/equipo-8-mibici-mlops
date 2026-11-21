# Servicio de predicción MiBici: FastAPI y el champion de mibici_salidas_diarias
FROM python:3.13.17-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_URI=/app/models/serving/model

WORKDIR /app

# Las librerías van antes que el modelo para reutilizar esta capa si solo cambia el modelo
COPY requirements-serving.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY mibici_mlops/serving/app.py app.py
COPY models/serving/ models/serving/

RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
