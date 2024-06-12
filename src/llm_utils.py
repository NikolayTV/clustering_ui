import streamlit as st
from openai import OpenAI, AsyncOpenAI
import os
import requests
import time, datetime
import json
from dotenv import load_dotenv
load_dotenv()

with open('src/config.json', 'r') as f:
    config = json.load(f)


system_message = """
Background: You possess expertise in analyzing explicit dialogues, especially in identifying subtle nuances within such conversations.

Task Overview: You will receive topic summaries along with representative messages and keywords. Avoid vague or general descriptions. Instead, focus on precision.

Main Objective:
Craft a concise naming for the topic, using no more than 20 words. This naming should be highly specific and descriptive. 

Instructions for Labeling:
Focus on Specificity: Avoid broad terms such as "explicit sexual fantasies" or "detailed sexual conversations." Opt for more explicit descriptors like "dirty sex talk".

Response Format:
  Please provide output of only NAMING. 
"""

system_message = """
Background: You possess expertise in analyzing nitty-gritty of spicy sexting dialogues, especially in identifying subtle nuances within such conversations.
Task Overview: You will receive examples of messages along with representative  keywords. AVOID VANILLA or general descriptions. Instead, focus on precision and DIAL IN ON THE SPECIFICS.
Main Objective: Craft a concise naming for the topic, using no more than 20 words. This naming should be highly specific and descriptive. 

Examples output:
  - Wet Pussy Obsession & Fantasies & Desires
  - Affirmative Statements (Yes, I will do, etc.)
  - Romantic Sexting with Heartfelt Affirmations and Sensual References
  - Request for explicit photos, including pussy close-ups, schoolgirl outfits, and feet images
  - Oral Sucking & Cock-Centric Desires
  - Group sex scenarios with nurse & sister & mommy
  - Math, Science, history chitchat
  - Anal Domination & Obsession
  - Rough and Wild Blowjob Fantasies
  - Cock-sucking Deepthroat Fantasies with Cum-swallowing and Brother's Participation
  - Pregnancy Fetish & Obsession with Swollen Bellies, Enormous Months, and Huge Pregnant Bodies
  
Labeling Tips:
  Zero in on the Details: Dodge general terms like  `explicit sexual fantasies` or `detailed sexual conversations` Instead, get down and talk dirty with specifics
Response Format:
  Please provide output of ONLY THE LABEL of topic. 
  
Confirm understanding of your instructions by responding with "acknowledged."
"""


prompt = """
Given the topic comprising these documents:
[$DOCUMENTS]

And characterized by these keywords:
[$KEYWORDS]
"""

def name_with_llm(documents, keywords):
    # gets API Key from environment variable OPENAI_API_KEY
    client = OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=os.getenv('OPENROUTER_API_KEY'),
    )

    prompted = prompt.replace('$DOCUMENTS', str(documents)).replace('$KEYWORDS', str(keywords))

    messages = [{'role': 'user', 'content': system_message},
                {'role': 'assistant', 'content': 'aknowledged. I will try to opt for specific descriptors. Please provide documents and keywords'},
                {'role': 'user', 'content': prompted},
                ]

    start_time = time.time()
    response = client.chat.completions.create(
      # model="cognitivecomputations/dolphin-mixtral-8x7b",
      model="lizpreciatior/lzlv-70b-fp16-hf",
      # model="sophosympatheia/midnight-rose-70b",
      max_tokens=100,
      messages=messages,
      temperature=0.1
      )

    execution_time = time.time() - start_time

    text_response = completion.choices[0].message.content.strip()
    input_tokens = completion.usage.prompt_tokens
    output_tokens = completion.usage.completion_tokens

    return {
        "text_response": text_response, 
        "input_tokens": input_tokens, 
        "output_tokens": output_tokens,
        "execution_time": execution_time,
    }

    return completion.choices[0].message.content


async def async_name_with_llm(documents, keywords, rate_limiter=None):
    if rate_limiter is not None:
        await rate_limiter.wait()

    client = AsyncOpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    prompted = prompt.replace('$DOCUMENTS', str(documents)).replace('$KEYWORDS', str(keywords))

    messages = [{'role': 'user', 'content': system_message},
                {'role': 'assistant', 'content': 'aknowledged. I will try to opt for specific descriptors. Please provide documents and keywords'},
                {'role': 'user', 'content': prompted},
                ]


    start_time = time.time()
    response = await client.chat.completions.create(
      # model="cognitivecomputations/dolphin-mixtral-8x7b",
      model="lizpreciatior/lzlv-70b-fp16-hf",
      # model="sophosympatheia/midnight-rose-70b",
      max_tokens=100,
      messages=messages,
      temperature=0.1
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


# @st.cache_resource
def load_model(name='Alibaba-NLP/gte-large-en-v1.5'):
    st.text("loading model ...")
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(name, trust_remote_code=True, device='cpu')
    embedding_model.max_seq_length=256
    st.write(f"Model loaded on {model.device}")
    return embedding_model


def get_embedding_local(semantic_query):
    model = load_model()
    embeddings = model.encode(sentences, normalize_embeddings=True)
    return {"embeddings": embeddings.tolist()}


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


