FROM python:3.11-slim
WORKDIR /app
#  Suppress TensorFlow logging and disable oneDNN optimizations for better compatibility in container environments
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY models/ ./models/
EXPOSE 8000
CMD ["uvicorn", "src.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]