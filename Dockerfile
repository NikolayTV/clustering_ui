FROM python:3.12-alpine

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    TZ=Europe/Moscow

WORKDIR /app

RUN apk add --no-cache tzdata ffmpeg g++
RUN apk add --no-cache build-base linux-headers
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN  pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt stopwords

COPY . .

