"""Local, read-only HTTP server for simulation architecture snapshots."""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Callable
from urllib.parse import urlsplit

from .snapshot import build_snapshot
from .web import INDEX_HTML


class _VisualizerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _handler_type(
    engine: object,
    snapshot_factory: Callable[[object], dict[str, object]],
    *,
    log_requests: bool,
) -> type[BaseHTTPRequestHandler]:
    class VisualizerRequestHandler(BaseHTTPRequestHandler):
        server_version = "GenesisSystemVisualizer/1"
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/snapshot":
                try:
                    payload = snapshot_factory(engine)
                    body = json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                except Exception as error:
                    body = json.dumps(
                        {"error": "snapshot unavailable", "error_type": type(error).__name__},
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                    self._send(HTTPStatus.INTERNAL_SERVER_ERROR, body, "application/json; charset=utf-8")
                    return
                self._send(HTTPStatus.OK, body, "application/json; charset=utf-8")
                return
            if path == "/favicon.ico":
                self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
                return
            body = b'{"error":"not found"}'
            self._send(HTTPStatus.NOT_FOUND, body, "application/json; charset=utf-8")

        def do_HEAD(self) -> None:
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self._send(
                    HTTPStatus.OK,
                    INDEX_HTML.encode("utf-8"),
                    "text/html; charset=utf-8",
                    include_body=False,
                )
            elif path == "/api/snapshot":
                self._send(HTTPStatus.OK, b"", "application/json; charset=utf-8", include_body=False)
            else:
                self._send(HTTPStatus.NOT_FOUND, b"", "application/json; charset=utf-8", include_body=False)

        def do_POST(self) -> None:
            self._method_not_allowed()

        def do_PUT(self) -> None:
            self._method_not_allowed()

        def do_PATCH(self) -> None:
            self._method_not_allowed()

        def do_DELETE(self) -> None:
            self._method_not_allowed()

        def _method_not_allowed(self) -> None:
            self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
            self.send_header("Allow", "GET, HEAD")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _send(
            self,
            status: HTTPStatus,
            body: bytes,
            content_type: str,
            *,
            include_body: bool = True,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'",
            )
            self.end_headers()
            if include_body and body:
                self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            if log_requests:
                super().log_message(format, *args)

    return VisualizerRequestHandler


class VisualizerServer:
    """A running background server that can also be used as a context manager."""

    def __init__(self, server: ThreadingHTTPServer, thread: threading.Thread) -> None:
        self._server = server
        self._thread = thread
        self._stop_lock = threading.Lock()
        self._stopped = False

    @property
    def host(self) -> str:
        return str(self._server.server_address[0])

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive() and not self._stopped

    def stop(self) -> None:
        """Stop serving and release the listening socket. Safe to call repeatedly."""
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=5)

    close = stop

    def __enter__(self) -> VisualizerServer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()


def start_server(
    engine: object,
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    snapshot_factory: Callable[[object], dict[str, object]] = build_snapshot,
    log_requests: bool = False,
) -> VisualizerServer:
    """Start the inspector in a daemon thread and return its stop handle."""
    if not isinstance(host, str) or not host:
        raise ValueError("host must be a non-empty string")
    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer from 0 through 65535")
    if not callable(snapshot_factory):
        raise TypeError("snapshot_factory must be callable")

    server = _VisualizerHTTPServer(
        (host, port),
        _handler_type(engine, snapshot_factory, log_requests=log_requests),
    )
    thread = threading.Thread(
        target=server.serve_forever,
        name=f"simulation-visualizer-{server.server_address[1]}",
        daemon=True,
    )
    try:
        thread.start()
    except BaseException:
        server.server_close()
        raise
    return VisualizerServer(server, thread)


start_visualizer = start_server

__all__ = ["VisualizerServer", "start_server", "start_visualizer"]
