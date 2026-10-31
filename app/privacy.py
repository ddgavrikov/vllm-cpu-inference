"""No message text, exception text, URL, headers or client-supplied IDs in logs."""
import json
import logging
import time
import uuid


def safe_event(value):
    if not isinstance(value, dict):
        return None
    event = value.get("event")
    if event == "runtime_log":
        level = value.get("level")
        return {"event": event, "level": level} if level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"} else None
    if event == "inference_request":
        # Never accept arbitrary strings, even from a log record's extra fields.
        try:
            request_id = str(uuid.UUID(value["request_id"]))
            status = int(value["status"])
            elapsed = float(value["elapsed_ms"])
            if not 100 <= status <= 599 or not 0 <= elapsed < 86_400_000:
                return None
            return {"event": event, "request_id": request_id, "status": status, "elapsed_ms": round(elapsed, 3)}
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
    return None


class PrivacyFormatter(logging.Formatter):
    def format(self, record):
        event = safe_event(getattr(record, "audit_event", None))
        # Intentionally don't call getMessage(), formatException() or formatStack().
        return json.dumps(event or {"event": "runtime_log", "level": record.levelname}, allow_nan=False)


class AuditMiddleware:
    """Pure ASGI middleware: measures complete streams without buffering content."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") not in {"/v1/chat/completions", "/v1/completions"}:
            return await self.app(scope, receive, send)
        start = time.perf_counter()
        status = 500
        async def audit_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)
        try:
            await self.app(scope, receive, audit_send)
        finally:
            logging.getLogger("audit").info("", extra={"audit_event": {
                "event": "inference_request", "request_id": str(uuid.uuid4()),
                "status": status, "elapsed_ms": (time.perf_counter() - start) * 1000,
            }})
