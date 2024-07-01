import os
import dotenv
dotenv.load_dotenv()

AVAILABLE_MODELS = [
    {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "model": "lizpreciatior/lzlv-70b-fp16-hf",
        "max_input": 32000
    },
    
    {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "model": "qwen/qwen-2-72b-instruct",
        "max_input": 32000
    },
    
    {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "model": "perplexity/llama-3-sonar-large-32k-chat",
        "max_input": 32000
    },

    {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "model": "cognitivecomputations/dolphin-mixtral-8x22b",
        "max_input": 32000
    },
    
    {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "model": "01-ai/yi-large",
        "max_input": 32000
    },
        
    {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "model": "google/gemma-2-9b-it:free",
        "max_input": 8192
    },

    {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "model": "sao10k/l3-euryale-70b",
        "max_input": 16000
    },
    
    {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "model": "nvidia/nemotron-4-340b-instruct",
        "max_input": 4096
    },
]

USE_LOCAL_EMBED_MODEL = False
EMBED_MODEL = {
    "url": "https://api.runpod.ai/v2/jp223n8tzrt271/runsync",
    "api_key": os.getenv('RUNPOD_API_KEY'),
}


