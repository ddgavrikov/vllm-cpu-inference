# syntax=docker/dockerfile:1
FROM python:3.12.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp VLLM_CACHE_ROOT=/tmp/vllm HF_HOME=/tmp/hf \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libnuma1 libtcmalloc-minimal4 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 10001 app && useradd -u 10001 -g 10001 -M app
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes -r requirements.txt \
    && python -m pip check \
    && python -c "import torch; assert torch.version.cuda is None; from vllm.platforms import current_platform; assert current_platform.is_cpu()"
COPY app /app/app
COPY logging.json /app/logging.json
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["python", "-m", "app.launch"]
