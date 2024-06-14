import runpod, os
import torch
from sentence_transformers import SentenceTransformer

# Check if CUDA is available and set the device accordingly
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initialize the model with the determined device
model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True) # device='cpu', 


def handler(event):
    sentences = event['input'].get('sentences', [])
    if not sentences:
        return {"error": "No sentences provided"}

    embeddings = model.encode(sentences, normalize_embeddings=True)
    return {"embeddings": embeddings.tolist()}


runpod.serverless.start({"handler": handler})