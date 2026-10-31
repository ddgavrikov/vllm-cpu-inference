import io
import json
import unittest
from unittest.mock import patch

from scripts.benchmark import request


class BenchmarkTests(unittest.TestCase):
    def stream(self, complete=True):
        events = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "hello"}}]},
            {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 1}},
        ]
        payload = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
        return io.BytesIO(payload + (b"data: [DONE]\n\n" if complete else b""))

    def test_ttft_starts_with_content_not_role(self):
        with patch("urllib.request.urlopen", return_value=self.stream()), patch("time.perf_counter", side_effect=[0, .2, .7]):
            sample = request("http://localhost:8000", "test-token", "hello")
        self.assertEqual(sample, {"ttft_s": .2, "latency_s": .7, "prompt_tokens": 12, "output_tokens": 1})

    def test_truncated_stream_fails(self):
        with patch("urllib.request.urlopen", return_value=self.stream(complete=False)):
            with self.assertRaises(RuntimeError):
                request("http://localhost:8000", "test-token", "hello")

    def test_stream_error_fails(self):
        response = io.BytesIO(b'data: {"error": {"message": "failed"}}\n\n')
        with patch("urllib.request.urlopen", return_value=response):
            with self.assertRaises(RuntimeError):
                request("http://localhost:8000", "test-token", "hello")
