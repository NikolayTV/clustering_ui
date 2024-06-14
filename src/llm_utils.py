import streamlit as st
from openai import OpenAI, AsyncOpenAI
import os
import requests
import time, datetime
import json
import asyncio
from dotenv import load_dotenv

from prompts import clustering_system_message
import config

load_dotenv()

async def async_call_llm(messages, model_creds, max_tokens=4096, temperature=0.1, rate_limiter=None):
    if rate_limiter is not None:
        await rate_limiter.wait()
    
    client = AsyncOpenAI(
      base_url = model_creds.get('base_url'),
      api_key = model_creds.get('api_key')
    )

    start_time = time.time()
    response = await client.chat.completions.create(
      model=model_creds.get('model'),
      messages=messages,
      max_tokens=max_tokens,
      temperature=temperature
      )

    execution_time = time.time() - start_time
    
    text_response = response.choices[0].message.content.strip()
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    return {
        "text_response": text_response, 
        "input_tokens": input_tokens, 
        "output_tokens": output_tokens,
        "execution_time": execution_time,
    }


async def async_name_with_llm(documents, keywords, model_creds, max_tokens=4096, temperature=0.1, system_message=clustering_system_message, rate_limiter=None):
    if rate_limiter is not None:
        await rate_limiter.wait()

    client = AsyncOpenAI(
      base_url = model_creds.get('base_url'),
      api_key = model_creds.get('api_key')
    )

    prompt = """
    Given the topic comprising these documents:
    [$DOCUMENTS]

    And characterized by these keywords:
    [$KEYWORDS]
    """
    prompted = prompt.replace('$DOCUMENTS', str(documents)).replace('$KEYWORDS', str(keywords))
    print(prompted)

    messages = [{'role': 'user', 'content': system_message},
                {'role': 'assistant', 'content': 'aknowledged. I will try to opt for specific descriptors. Please provide documents and keywords'},
                {'role': 'user', 'content': prompted},
                ]


    start_time = time.time()
    response = await client.chat.completions.create(
      model=model_creds.get('model'),
      messages=messages,
      max_tokens=max_tokens,
      temperature=temperature
      )

    execution_time = time.time() - start_time
    
    text_response = response.choices[0].message.content.strip()
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    return {
        "text_response": text_response, 
        "input_tokens": input_tokens, 
        "output_tokens": output_tokens,
        "execution_time": execution_time,
    }


import os
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

    return False
def get_service_address():
    """
    Determine the service address based on the environment.
    """
    
    if is_running_in_docker():
        return 'http://embservice:8000/embed'
    else:
        return 'http://localhost:8000/embed'
        # return "http://localhost:8000/rynsync" # runpod handler

# Usage
def get_embedding_runpod(semantic_query):
    if config.USE_LOCAL_EMBED_MODEL:
        url = get_service_address()
    else:
        url = config.EMBED_MODEL['url']
        api_key = config.EMBED_MODEL['api_key']
        print('USING REMOTE EMBED MODEL')
        
    print('EMB URL', url)
    headers = {
        'accept': 'application/json',
        'authorization': api_key,
        'Content-Type': 'application/json'
    }

    data = {'input': {
                    "sentences": [
                        semantic_query
                    ]
                }
            }
    try:
        response = requests.post(url, headers=headers, json=data)
        emb = response.json()['output']['embeddings'][0]
    except Exception as e:
        print(f'Error trying to call local embed model {e}')    
        
        
    return {"embeddings": emb}




class RateLimiter:
    def __init__(self, calls_per_period, period=1.0):
        self.calls_per_period = calls_per_period
        self.period = datetime.timedelta(seconds=period)
        self.calls = []

    async def wait(self):
        now = datetime.datetime.now()
        
        while self.calls and now - self.calls[0] > self.period:
            self.calls.pop(0)
            
        if len(self.calls) >= self.calls_per_period:
            sleep_time = (self.period - (now - self.calls[0])).total_seconds()
            await asyncio.sleep(sleep_time)
            return await self.wait()

        self.calls.append(datetime.datetime.now())


