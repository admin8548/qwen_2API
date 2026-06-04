import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

log = logging.getLogger("qwen2api.raw")


class RawRequestLogger(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method
        client = request.client
        auth = request.headers.get("authorization", "")
        auth_preview = auth[:30] + "..." if len(auth) > 30 else auth
        log.warning(
            "[RAW] %s %s from=%s auth=%s content-type=%s",
            method, path,
            f"{client.host}:{client.port}" if client else "?",
            auth_preview,
            request.headers.get("content-type", ""),
        )
        try:
            response = await call_next(request)
            log.warning("[RAW] %s %s -> %s", method, path, response.status_code)
            return response
        except Exception as e:
            log.exception("[RAW] %s %s EXCEPTION: %s", method, path, e)
            raise
