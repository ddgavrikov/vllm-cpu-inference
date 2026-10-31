import json
import os
import subprocess
import urllib.error
import urllib.request
from benchmark import request

BASE = "http://127.0.0.1:8000"
token = os.environ["API_TOKEN"]
sentinel = "PRIVACY_SENTINEL_2394eea52f"
for auth in [None, "wrong-key"]:
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = "Bearer " + auth
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=b"{}", headers=headers)
    try:
        urllib.request.urlopen(req)
        raise AssertionError("Unauthenticated request accepted")
    except urllib.error.HTTPError as exc:
        assert exc.code == 401, exc.code
request(BASE, token, "Repeat this string exactly: " + sentinel, 16)
bad = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps({"model": "qwen-cpu", "messages": sentinel}).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + token})
try:
    urllib.request.urlopen(bad)
    raise AssertionError("Malformed input accepted")
except urllib.error.HTTPError as exc:
    assert exc.code in {400, 422}, exc.code
logs = subprocess.check_output(["kubectl", "-n", "llm-cpu", "logs", "deployment/llm", "-c", "vllm"], text=True)
assert sentinel not in logs and token not in logs
assert any(json.loads(line).get("event") == "inference_request" for line in logs.splitlines())
print("PASS auth, streaming inference, malformed request, private audit logs")
