FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

ENV HF_HOME=/app/hf_home
RUN mkdir -p $HF_HOME && chmod -R 777 $HF_HOME

ARG MODEL_NAME=roberta-base
ENV MODEL_NAME=${MODEL_NAME}
ARG RUN_MODE=SMALL_RUN
ENV RUN_MODE=${RUN_MODE}

COPY config.py data.py train.py eval.py utils.py main.py /app/

CMD ["sh", "-c", "python main.py --run-mode \"${RUN_MODE}\" --disable-wandb --no-push-to-hub"]
