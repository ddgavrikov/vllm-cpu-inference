# syntax=docker/dockerfile:1
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp VLLM_CACHE_ROOT=/tmp/vllm HF_HOME=/tmp/hf \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libnuma1 libtcmalloc-minimal4 ffmpeg libsm6 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 10001 app && useradd -u 10001 -g 10001 -M app
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --only-binary=:all: --extra-index-url https://download.pytorch.org/whl/cpu --require-hashes -r requirements.txt \
    && python -m pip check
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/usr/local/lib/libiomp5.so"
RUN test -f /usr/local/lib/libiomp5.so \
    && python -c "import torch; assert torch.version.cuda is None; from vllm.platforms import current_platform; assert current_platform.is_cpu()"
COPY app /app/app
COPY logging.json /app/logging.json
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["python", "-m", "app.launch"]
