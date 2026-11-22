FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY docker/requirements-onnx.txt .
RUN uv pip install --system --no-cache -r requirements-onnx.txt

# Download the tokenizer and model architecture at image build time rather than during container startup.
ENV HF_HOME=/app/hf
RUN python -c "from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained('sentence-transformers/multi-qa-mpnet-base-cos-v1')"

COPY model_optimized.onnx docker/server.py docker/app_onnx.py ./

EXPOSE 8000
CMD ["python", "app_onnx.py"]
