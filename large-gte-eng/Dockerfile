# Dockerfile
FROM runpod/base:0.6.1-cuda12.1.0


# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# NOTE: The base image comes with multiple Python versions pre-installed.
#       It is recommended to specify the version of Python when running your code.

# Add src files (Worker Template)
ADD . /

# RUN python3.11 -c "from sentence_transformers import SentenceTransformer; \
#     model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', cache_folder='/root/.cache/huggingface', trust_remote_code=True)"

# CMD python3.11 -u /src/handler.py --rp_serve_api 
CMD python3.11 -u /src/fastapi_handler.py
