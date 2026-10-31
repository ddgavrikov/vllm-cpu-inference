import asyncio
import io
import json
import logging
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from app.launch import command, forward
from app.privacy import AuditMiddleware, PrivacyFormatter, safe_event
from scripts.download_model import download, valid_cache


class PrivacyTests(unittest.TestCase):
    def test_formatter_discards_prompt_response_and_exception(self):
        record = logging.LogRecord("vllm", logging.ERROR, "", 1, "PROMPT %s", ("SECRET_RESPONSE",), None)
        try:
            raise ValueError("PRIVATE_EXCEPTION")
        except ValueError:
            record.exc_info = sys.exc_info()
        result = PrivacyFormatter().format(record)
        self.assertEqual(json.loads(result), {"event": "runtime_log", "level": "ERROR"})

    def test_supervisor_rejects_native_and_arbitrary_json(self):
        lines = io.StringIO('PRIVATE\n{"event":"runtime_log","level":"INFO","message":"PRIVATE"}\n{"event":"PRIVATE"}\n')
        out = io.StringIO()
        with patch("sys.stdout", out):
            forward(lines)
        self.assertEqual(json.loads(out.getvalue()), {"event": "runtime_log", "level": "INFO"})
        self.assertNotIn("PRIVATE", out.getvalue())

    def test_invalid_audit_is_rejected(self):
        for elapsed in [float("nan"), float("inf"), -1]:
            self.assertIsNone(safe_event({"event": "inference_request", "request_id": "00000000-0000-0000-0000-000000000000", "status": 200, "elapsed_ms": elapsed}))

    def test_streaming_middleware_success_and_failure(self):
        async def run(fail):
            sent = []
            async def app(scope, receive, send):
                await send({"type": "http.response.start", "status": 200})
                await send({"type": "http.response.body", "body": b"PRIVATE", "more_body": False})
                if fail:
                    raise ValueError("PRIVATE")
            async def send(message):
                sent.append(message)
            with patch("logging.Logger.info") as log:
                try:
                    await AuditMiddleware(app)({"type": "http", "path": "/v1/chat/completions"}, None, send)
                except ValueError:
                    self.assertTrue(fail)
                self.assertEqual(log.call_count, 1)
                event = log.call_args.kwargs["extra"]["audit_event"]
                self.assertEqual(event["status"], 200)
                self.assertNotIn("PRIVATE", json.dumps(event))
                self.assertEqual(sent[-1]["body"], b"PRIVATE")
        asyncio.run(run(False))
        asyncio.run(run(True))

    def test_launch_cannot_override_privacy_flags(self):
        with self.assertRaises(ValueError):
            command({"enable-log-outputs": True}, "/models/test")
        self.assertIn("--no-enable-log-requests", command({"dtype": "float32"}, "/models/test"))


class CacheTests(unittest.TestCase):
    def test_cold_warm_corrupt_and_partial(self):
        calls = []
        def snapshot(**kwargs):
            calls.append(kwargs)
            path = Path(kwargs["local_dir"])
            path.mkdir(exist_ok=True)
            (path / "config.json").write_text("{}")
            (path / "model.safetensors").write_bytes(b"test weights")
        revision = "a" * 40
        # On Windows only the OS locking primitive is stubbed; Linux CI tests real flock.
        modules = {} if sys.platform != "win32" else {"fcntl": types.SimpleNamespace(LOCK_EX=2, LOCK_UN=8, flock=lambda *a: None)}
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, modules):
            target = download(temp, "test/model", revision, snapshot=snapshot)
            self.assertTrue(valid_cache(target, "test/model", revision))
            download(temp, "test/model", revision, snapshot=lambda **kw: self.fail("Network accessed on cache hit"))
            (target / "model.safetensors").write_bytes(b"truncated")
            self.assertFalse(valid_cache(target, "test/model", revision))
            download(temp, "test/model", revision, snapshot=snapshot)
            self.assertEqual(len(calls), 2)

    def test_revision_must_be_immutable(self):
        with self.assertRaises(ValueError):
            download("/unused", "test/model", "main")


if __name__ == "__main__":
    unittest.main()
