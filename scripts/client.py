import json
import os
import urllib.request

req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps({"model": "qwen-cpu", "messages": [{"role": "user", "content": "Explain what a data pipeline is in one sentence."}], "max_tokens": 64}).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["API_TOKEN"]})
with urllib.request.urlopen(req, timeout=600) as response:
    print(json.load(response)["choices"][0]["message"]["content"])
