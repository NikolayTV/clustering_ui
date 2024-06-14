from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
from sentence_transformers import SentenceTransformer
import uvicorn, os

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def is_running_in_docker():
    """
    Check if the application is running inside a Docker container.
    This method checks the existence of the .dockerenv file and inspects the /proc/1/cgroup file.
    """
    # Check for .dockerenv file
    if os.path.exists('/.dockerenv'):
        return True

    # Check for 'docker' or 'kubepods' in /proc/1/cgroup
    try:
        with open('/proc/1/cgroup', 'rt') as f:
            cgroup_content = f.read()
            if 'docker' in cgroup_content or 'kubepods' in cgroup_content:
                return True
    except Exception:
        pass

# cache_folder=os.listdir(os.environ("HF_HOME"))

if is_running_in_docker():
    cache_folder = os.environ['HF_HOME']
    print('cache_folder', cache_folder)
    print('Available models', os.listdir(cache_folder + '/hub'))
    model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', cache_folder=cache_folder, trust_remote_code=True) # device='cpu', 
else:
    model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True) # device='cpu', 
    
print('Model loaded', model.device)
app = FastAPI()

class InputModel(BaseModel):
    sentences: List[str]

class RequestModel(BaseModel):
    input: InputModel


@app.post("/embed")
async def get_embeddings(request: RequestModel):
    sentences = request.input.sentences
    if not sentences:
        raise HTTPException(status_code=400, detail="No sentences provided")

    embeddings = model.encode(sentences, normalize_embeddings=True)
    return {'output': {"embeddings": embeddings.tolist()}}
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)