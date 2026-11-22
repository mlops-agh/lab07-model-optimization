import onnxruntime as ort
from server import run
from transformers import AutoTokenizer

MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-cos-v1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session = ort.InferenceSession(
    "model_optimized.onnx", sess_options=options, providers=["CPUExecutionProvider"]
)


def embed(text):
    inputs = tokenizer([text], return_tensors="np", truncation=True)
    inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    return session.run(None, inputs)[0][:, 0].tolist()


run(embed)
