FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# torch.compile on CPU needs a C++ compiler.
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/requirements-torch.txt .
RUN uv pip install --system --no-cache -r requirements-torch.txt

# Download the tokenizer and model architecture at image build time rather than during container startup.
ENV HF_HOME=/app/hf
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
n = 'sentence-transformers/multi-qa-mpnet-base-cos-v1'; \
AutoTokenizer.from_pretrained(n); AutoModel.from_pretrained(n)"

COPY docker/server.py docker/app_torch.py ./

EXPOSE 8000
CMD ["python", "app_torch.py"]
