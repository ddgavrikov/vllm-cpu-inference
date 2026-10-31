import base64
import getpass
import json
import os
import secrets
import subprocess

if __name__ == "__main__":
    hf = os.getenv("HF_TOKEN")
    if hf is None:
        hf = getpass.getpass("HF token (Enter for public Qwen model): ")
    api = os.getenv("API_TOKEN") or getpass.getpass("API token (Enter to generate): ") or secrets.token_urlsafe(32)
    obj = {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": "llm-secrets", "namespace": "llm-cpu"},
           "type": "Opaque", "data": {k: base64.b64encode(v.encode()).decode() for k, v in {"hf-token": hf, "api-token": api}.items()}}
    subprocess.run(["kubectl", "create", "namespace", "llm-cpu"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["kubectl", "apply", "--server-side", "-f", "-"], input=json.dumps(obj), text=True, check=True)
    print("Secret llm-secrets created. Token values were not printed.")
