FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

COPY app ./app

RUN mkdir -p data/pdfs storage/index

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

