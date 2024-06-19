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

]

USE_LOCAL_EMBED_MODEL = False
EMBED_MODEL = {
    "url": "https://api.runpod.ai/v2/jp223n8tzrt271/runsync",
    "api_key": os.getenv('RUNPOD_API_KEY'),
}


