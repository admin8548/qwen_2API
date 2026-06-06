import logging
import re
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

log = logging.getLogger("qwen2api.raw")


class RawRequestLogger(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        safe_path = re.sub(r"(/api/admin/keys/)[^/?#]+", r"\1<redacted>", path)
        method = request.method
        client = request.client
        auth_present = bool(request.headers.get("authorization") or request.headers.get("x-api-key"))
        level = logging.DEBUG if path in {"/healthz", "/readyz"} else logging.INFO
        log.log(
            level,
            "[RAW] %s %s from=%s auth=%s content-type=%s",
            method, safe_path,
            f"{client.host}:{client.port}" if client else "?",
            "<redacted>" if auth_present else "",
            request.headers.get("content-type", ""),
        )
        try:
            response = await call_next(request)
            log.log(level, "[RAW] %s %s -> %s", method, safe_path, response.status_code)
            return response
        except Exception as e:
            log.exception("[RAW] %s %s EXCEPTION: %s", method, path, e)
            raise
