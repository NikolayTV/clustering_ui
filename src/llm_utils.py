import streamlit as st
from openai import OpenAI, AsyncOpenAI
import os
import requests
import time, datetime
import json
import asyncio
from dotenv import load_dotenv

from prompts import clustering_system_message

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


def get_embedding_runpod(semantic_query):
    # url = 'https://api.runpod.ai/v2/jp223n8tzrt271/runsync'
    url = 'http://localhost:8000/runsync'

    headers = {
        'accept': 'application/json',
        'authorization': os.getenv('RUNPOD_API_KEY_SHARED'),
        'Content-Type': 'application/json'
    }
    data = {
        "input": {"sentences": semantic_query}
    }

    response = requests.post(url, headers=headers, json=data)

    # print(response.status_code)
    # print(response.json())

    return {"embeddings": response.json()['output']['embeddings']}




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


