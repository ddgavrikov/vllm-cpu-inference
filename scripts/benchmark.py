"""Streaming benchmark with measured server token counts; stores no prompt/response."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import statistics
import time
import urllib.request


def request(base, token, prompt, max_tokens=32):
    body = json.dumps({"model": "qwen-cpu", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    start = time.perf_counter()
    ttft = None
    usage = None
    done = False
    with urllib.request.urlopen(req, timeout=600) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                done = True
                break
            event = json.loads(payload)
            if "error" in event:
                raise RuntimeError("Server stream error")
            if event.get("usage"):
                usage = event["usage"]
            if ttft is None and any(c.get("delta", {}).get("content") for c in event.get("choices", [])):
                ttft = time.perf_counter() - start
    elapsed = time.perf_counter() - start
    if not done or ttft is None or not usage or usage["completion_tokens"] <= 0:
        raise RuntimeError("Incomplete generation or missing usage")
    return {"ttft_s": ttft, "latency_s": elapsed, "prompt_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"]}


def percentile(values, q):
    return sorted(values)[max(0, math.ceil(q * len(values)) - 1)]


def run(base, token, count, concurrency):
    rows = []
    for name, prompt in [("short", "Count from one to thirty."),
                         ("long", "Read the following notes. " + "A data pipeline validates records before storage. " * 120 + " Summarize the notes.")]:
        request(base, token, prompt)  # warmup excluded, one per scenario
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            samples = list(pool.map(lambda _: request(base, token, prompt), range(count)))
        wall = time.perf_counter() - start
        rows.append({"scenario": name, "requests": count, "concurrency": concurrency,
                     "prompt_tokens_mean": statistics.mean(s["prompt_tokens"] for s in samples),
                     "ttft_p50_s": statistics.median(s["ttft_s"] for s in samples),
                     "ttft_p95_s": percentile([s["ttft_s"] for s in samples], .95),
                     "latency_p50_s": statistics.median(s["latency_s"] for s in samples),
                     "latency_p95_s": percentile([s["latency_s"] for s in samples], .95),
                     "throughput_requests_s": count / wall,
                     "throughput_output_tokens_s": sum(s["output_tokens"] for s in samples) / wall,
                     "wall_s": wall, "samples": samples})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--output", default="results/benchmark.json")
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("counts must be positive")
    result = run(args.url, os.environ["API_TOKEN"], args.requests, args.concurrency)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    print("| Prompt | Input tokens | TTFT p50/p95 s | Latency p50/p95 s | req/s | output tok/s |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in result:
        print(f"| {row['scenario']} | {row['prompt_tokens_mean']:.0f} | {row['ttft_p50_s']:.3f}/{row['ttft_p95_s']:.3f} | {row['latency_p50_s']:.3f}/{row['latency_p95_s']:.3f} | {row['throughput_requests_s']:.3f} | {row['throughput_output_tokens_s']:.3f} |")
