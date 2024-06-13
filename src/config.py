import os

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
        
]