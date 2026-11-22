import torch
from server import run
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-cos-v1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = torch.compile(AutoModel.from_pretrained(MODEL_NAME).eval(), dynamic=True)


def embed(text):
    inputs = tokenizer([text], return_tensors="pt", truncation=True)
    with torch.inference_mode():
        return model(**inputs).last_hidden_state[:, 0].tolist()


embed("warm-up")  # compile before the first request
run(embed)
