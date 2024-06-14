FROM python:3.9-slim-bullseye

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    TZ=Europe/Moscow \
    PATH=/opt/conda/bin:$PATH

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tzdata ffmpeg g++ curl bzip2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# # Install Miniconda conditionally based on OS type
# RUN if [ "$(uname -m)" = "x86_64" ]; then \
#         curl -o ~/miniconda.sh -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; \
#     elif [ "$(uname -m)" = "aarch64" ]; then \
#         curl -o ~/miniconda.sh -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh; \
#     fi \
#     && bash ~/miniconda.sh -b -p /opt/conda \
#     && rm ~/miniconda.sh \
#     && /opt/conda/bin/conda clean -tipy

# Install faiss-cpu using conda
# RUN conda install -c conda-forge faiss-cpu -y
RUN pip install faiss-cpu

# Update pip and install Python dependencies
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt stopwords
RUN pip install --no-cache-dir python-dotenv


# Copy application code
COPY . .

# Command to run the application
EXPOSE 8501
CMD ["streamlit", "run", "src/1_home.py", "--server.port", "8501"]



