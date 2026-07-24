from __future__ import annotations

from collections.abc import Iterable

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .settings_service import get_runtime_parameters


class DatabaseCORSMiddleware:
    """Resolve allowed origins from the runtime database for every request.

    This avoids requiring a service restart after api.cors_origins changes and
    lets database catalog migrations add required deployment origins safely.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        allow_methods: Iterable[str],
        allow_headers: Iterable[str],
    ) -> None:
        self.app = app
        self.allow_methods = ", ".join(allow_methods)
        self.allow_headers = ", ".join(allow_headers)

    @staticmethod
    def _allowed_origins() -> set[str]:
        values = get_runtime_parameters().list("api.cors_origins")
        return {str(value).rstrip("/") for value in values if str(value).strip()}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        origin = headers.get("origin")
        normalized_origin = origin.rstrip("/") if origin else None
        allowed = normalized_origin in self._allowed_origins() if normalized_origin else False

        if scope["method"] == "OPTIONS" and headers.get("access-control-request-method"):
            response_headers = [
                (b"content-length", b"0"),
                (b"vary", b"Origin"),
            ]
            if allowed and origin:
                response_headers.extend(
                    [
                        (b"access-control-allow-origin", origin.encode("latin-1")),
                        (b"access-control-allow-methods", self.allow_methods.encode("latin-1")),
                        (b"access-control-allow-headers", self.allow_headers.encode("latin-1")),
                        (b"access-control-max-age", b"600"),
                    ]
                )
                await send({"type": "http.response.start", "status": 200, "headers": response_headers})
            else:
                await send({"type": "http.response.start", "status": 400, "headers": response_headers})
            await send({"type": "http.response.body", "body": b""})
            return

        async def send_with_cors(message: Message) -> None:
            if message["type"] == "http.response.start" and allowed and origin:
                mutable = MutableHeaders(scope=message)
                mutable["Access-Control-Allow-Origin"] = origin
                mutable.add_vary_header("Origin")
            await send(message)

        await self.app(scope, receive, send_with_cors)
