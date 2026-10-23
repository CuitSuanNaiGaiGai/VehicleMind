"""本地 HTTP 与 SSE 接口；浏览器只提交意图和当前动作 ID。"""

from __future__ import annotations

import json
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from apps.vehicle_ai_demo.web_session import SessionRegistry, valid_session_id


UI_ROOT = Path(__file__).with_name("web_ui")
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/graph.js": ("graph.js", "text/javascript; charset=utf-8"),
}


class DemoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, registry: SessionRegistry):
        self.registry = registry
        super().__init__(address, DemoHandler)


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # 高频 SSE 与状态读取不刷屏；错误通过操作事件显示在界面。
        pass

    def _json(self, status: int, payload: dict, cookie: str | None = None):
        data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.send_header("X-Content-Type-Options", "nosniff")
        if cookie:
            self.send_header(
                "Set-Cookie",
                f"vm_session={cookie}; Path=/; HttpOnly; SameSite=Strict; Max-Age=604800",
            )
        self.end_headers()
        self.wfile.write(data)

    def _session_id(self) -> str:
        cookies = SimpleCookie(self.headers.get("Cookie", ""))
        item = cookies.get("vm_session")
        if item is None or not valid_session_id(item.value):
            raise ValueError("请先打开演示首页建立会话")
        return item.value

    def do_GET(self):
        if not self._valid_host():
            self._json(403, {"error": "仅允许本地演示地址"})
            return
        path = urlsplit(self.path).path
        try:
            if path in ASSETS:
                filename, content_type = ASSETS[path]
                data = (UI_ROOT / filename).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; object-src 'none'; frame-ancestors 'none'",
                )
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/bootstrap":
                cookies = SimpleCookie(self.headers.get("Cookie", ""))
                item = cookies.get("vm_session")
                session_id = (
                    item.value
                    if item and valid_session_id(item.value)
                    else self.server.registry.new_id()
                )
                session = self.server.registry.get(session_id)
                self._json(
                    200, {**session.view(), "token": session.token}, cookie=session_id
                )
            elif path == "/api/events":
                self._events()
            elif path == "/api/state":
                self._json(200, self.server.registry.get(self._session_id()).view())
            else:
                self._json(404, {"error": "页面不存在"})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as error:
            self._json(400, {"error": str(error)})

    def _events(self):
        session = self.server.registry.get(self._session_id())
        query = parse_qs(urlsplit(self.path).query)
        cursor = int(self.headers.get("Last-Event-ID", query.get("cursor", ["0"])[0]))
        if cursor < 0:
            raise ValueError("无效的事件位置")
        with session.condition:
            resync = cursor > session.sequence or bool(
                session.events and cursor < session.events[0]["id"] - 1
            )
            if resync:
                cursor = session.sequence
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(
            (
                "event: snapshot\ndata: "
                + json.dumps(
                    {**session.view(), "resync": resync},
                    ensure_ascii=False,
                    default=str,
                )
                + "\n\n"
            ).encode("utf-8")
        )
        self.wfile.flush()
        while True:
            with session.condition:
                pending = [event for event in session.events if event["id"] > cursor]
                if not pending:
                    session.condition.wait(timeout=10)
                    pending = [
                        event for event in session.events if event["id"] > cursor
                    ]
            if not pending:
                self.wfile.write(b": heartbeat\n\n")
            for event in pending:
                body = json.dumps(event, ensure_ascii=False, default=str)
                self.wfile.write(
                    f"id: {event['id']}\nevent: update\ndata: {body}\n\n".encode(
                        "utf-8"
                    )
                )
                cursor = event["id"]
            self.wfile.flush()

    def do_POST(self):
        try:
            if not self._valid_host():
                self._json(403, {"error": "仅允许本地演示地址"})
                return
            host = self.headers.get("Host", "")
            if self.headers.get("Origin") not in {None, f"http://{host}"}:
                self._json(403, {"error": "拒绝跨来源请求"})
                return
            if self.headers.get_content_type() != "application/json":
                self._json(415, {"error": "请求必须使用 JSON"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                self._json(413, {"error": "请求大小无效"})
                return
            payload = json.loads(self.rfile.read(length))
            if type(payload) is not dict:
                raise ValueError("请求必须是 JSON 对象")
            session = self.server.registry.get(self._session_id())
            if self.headers.get("X-VehicleMind-Token") != session.token:
                self._json(403, {"error": "会话令牌无效，请刷新页面"})
                return
            path = urlsplit(self.path).path
            if path == "/api/session":
                if session.busy:
                    raise RuntimeError("请等待当前操作结束后创建新会话")
                session_id = self.server.registry.new_id(payload.get("provider"))
                created = self.server.registry.get(session_id)
                self._json(
                    200, {**created.view(), "token": created.token}, cookie=session_id
                )
            elif path in {"/api/chat", "/api/confirm", "/api/reject", "/api/recover"}:
                session.submit(path.rsplit("/", 1)[1], payload)
                self._json(202, {"accepted": True})
            else:
                self._json(404, {"error": "接口不存在"})
        except RuntimeError as error:
            self._json(409, {"error": str(error)})
        except Exception as error:
            self._json(400, {"error": str(error)})

    def _valid_host(self):
        port = self.server.server_port
        return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}
