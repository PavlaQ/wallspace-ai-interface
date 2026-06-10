FROM python:3.11-slim

WORKDIR /app

# onnxruntime needs OpenMP at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model into the image at build time → no runtime dependency on
# Hugging Face, deterministic deploys. Default is full precision (best quality);
# override with --build-arg MODEL_FILE=model_quantized.onnx for a smaller/faster
# image at a small quality cost.
ARG MODEL_FILE=model.onnx
ARG MODEL_REPO=Xenova/segformer-b2-finetuned-ade-512-512
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/${MODEL_REPO}/resolve/main/onnx/${MODEL_FILE}', 'model.onnx')"

COPY main.py .

# Cloud Run injects $PORT (defaults to 8080).
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
