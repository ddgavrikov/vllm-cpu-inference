import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
from app.privacy import safe_event

ALLOWED = {"dtype", "max-model-len", "max-num-seqs", "max-num-batched-tokens", "served-model-name", "enforce-eager", "enable-prefix-caching", "enable-chunked-prefill"}


def command(config, model_path):
    if set(config) - ALLOWED:
        raise ValueError("Unsupported configuration key")
    args = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model_path,
            "--host", "0.0.0.0", "--port", "8000", "--load-format", "safetensors",
            "--no-enable-log-requests", "--no-enable-log-outputs", "--disable-uvicorn-access-log",
            "--disable-fastapi-docs", "--middleware", "app.privacy.AuditMiddleware",
            "--log-config-file", "/app/logging.json"]
    for key, value in config.items():
        if isinstance(value, bool):
            args.append("--" + ("" if value else "no-") + key)
        elif isinstance(value, (str, int, float)):
            args.extend(["--" + key, str(value)])
        else:
            raise ValueError("Unsupported configuration value")
    return args


def forward(stream):
    for line in stream:
        try:

            line = re.sub(r"^\([A-Za-z0-9_:-]+ pid=\d+\) ", "", line)
            event = safe_event(json.loads(line))
            if event:
                print(json.dumps(event, allow_nan=False), flush=True)
        except (ValueError, TypeError):
            pass


def main():
    if not os.environ.get("VLLM_API_KEY", "").strip():
        print('{"event":"startup_failed","reason":"missing_api_key"}', flush=True)
        return 1
    try:
        model = "/models/" + os.environ["MODEL_REVISION"]
        if not Path(model, "complete.json").is_file():
            raise ValueError("Model incomplete")
        cfg = json.loads(Path("/config/server.json").read_text())
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1", VLLM_LOGGING_CONFIG_PATH="/app/logging.json", PYTHONUNBUFFERED="1")
        child = subprocess.Popen(command(cfg, model), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    except Exception:
        print('{"event":"startup_failed","reason":"configuration"}', flush=True)
        return 1
    def stop(signum, frame):
        child.send_signal(signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    threads = [threading.Thread(target=forward, args=(s,), daemon=True) for s in (child.stdout, child.stderr)]
    for thread in threads:
        thread.start()
    code = child.wait()
    for thread in threads:
        thread.join()
    print(json.dumps({"event": "server_exit", "exit_code": code}), flush=True)
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main())
