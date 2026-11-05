"""HTTP contract tests for tools.server (audit FIND-016/017/018/019/042/060).

Starts the real handler on an ephemeral localhost port with stub engines
(canned strings, raised exceptions, deliberate delays). No external network.
"""
import contextlib
import http.client
import json
import socket
import subprocess
import sys
import threading
import time

import pytest

from tools import server

REPO_ROOT = server.REPO_ROOT


class StubEngine:
    def __init__(self, behavior=None):
        self._behavior = behavior or (lambda text: f"echo:{text}")

    def chat(self, text):
        return self._behavior(text)


class SlowEngine:
    def __init__(self, delay):
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def chat(self, text):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        with self._lock:
            self.active -= 1
        return "slow-ok"


@pytest.fixture()
def static_dir(tmp_path):
    (tmp_path / "index.html").write_text("<html>coo ui</html>", encoding="utf-8")
    return str(tmp_path)


@contextlib.contextmanager
def _run_server(engine, static_dir):
    handler = server.make_handler(engine, static_dir)
    httpd = server.CooServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        httpd.shutdown()
        httpd.server_close()


def _post(port, body, content_type="application/json", path="/chat", timeout=15):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("POST", path, body=body, headers={"Content-Type": content_type})
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, data
    finally:
        conn.close()


def _get(port, path, timeout=10):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, data
    finally:
        conn.close()


def _raw_request(port, payload, timeout=10):
    """Send raw bytes and read the whole response (for header-level faults)."""
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
        sock.sendall(payload)
        sock.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            try:
                data = sock.recv(65536)
            except socket.timeout:
                break
            if not data:
                break
            chunks.append(data)
        return b"".join(chunks)


def test_chat_valid_json_returns_200(static_dir):
    with _run_server(StubEngine(), static_dir) as port:
        status, body = _post(port, json.dumps({"text": "hello coo"}).encode())
    assert status == 200
    assert json.loads(body) == {"response": "echo:hello coo"}


def test_empty_text_gets_friendly_response(static_dir):
    with _run_server(StubEngine(), static_dir) as port:
        for text in ("", "   "):
            status, body = _post(port, json.dumps({"text": text}).encode())
            assert status == 200
            assert json.loads(body) == {"response": "coo..."}


def test_malformed_json_is_400_not_500(static_dir):
    with _run_server(StubEngine(), static_dir) as port:
        status, body = _post(port, b"{definitely not json")
    assert status == 400
    assert b"malformed" in body.lower()


def test_negative_content_length_rejected_without_wedging(static_dir):
    with _run_server(StubEngine(), static_dir) as port:
        response = _raw_request(
            port,
            b"POST /chat HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n"
            b"Content-Length: -5\r\n\r\n",
        )
        assert b" 400 " in response.split(b"\r\n", 1)[0] + b" "

        # The server must still answer a normal request within a bounded time.
        started = time.monotonic()
        status, _ = _post(port, json.dumps({"text": "again"}).encode())
        elapsed = time.monotonic() - started
        assert status == 200
        assert elapsed < 5  # pre-fix, read(-1) wedged until client EOF


def test_non_integer_content_length_is_400(static_dir):
    with _run_server(StubEngine(), static_dir) as port:
        response = _raw_request(
            port,
            b"POST /chat HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n"
            b"Content-Length: many\r\n\r\n",
        )
    assert b" 400 " in response.split(b"\r\n", 1)[0] + b" "


def test_oversized_body_is_413_and_server_stays_responsive(static_dir):
    with _run_server(StubEngine(), static_dir) as port:
        # Declare 2MB but only send headers + a little body: the cap must
        # trigger before the body is read.
        response = _raw_request(
            port,
            b"POST /chat HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n"
            b"Content-Length: 2097152\r\n\r\n" + b"x" * 1024,
        )
        assert b" 413 " in response.split(b"\r\n", 1)[0] + b" "

        status, _ = _get(port, "/index.html")
        assert status == 200


def test_non_json_content_type_is_415(static_dir):
    with _run_server(StubEngine(), static_dir) as port:
        status, _ = _post(port, b"text=hi", content_type="text/plain")
    assert status == 415


def test_engine_exception_returns_generic_500(static_dir):
    def boom(text):
        raise RuntimeError("secret internals: E:\\_work\\1\\s\\onnxruntime\\boom")

    with _run_server(StubEngine(boom), static_dir) as port:
        status, body = _post(port, json.dumps({"text": "hi"}).encode())
    assert status == 500
    assert json.loads(body) == {"error": "internal server error"}
    assert b"RuntimeError" not in body
    assert b"secret" not in body
    assert b"onnxruntime" not in body


def test_engine_valueerror_returns_400(static_dir):
    def reject(text):
        raise ValueError("bad text value")

    with _run_server(StubEngine(reject), static_dir) as port:
        status, body = _post(port, json.dumps({"text": "hi"}).encode())
    assert status == 400
    assert b"bad text value" not in body  # generic body, no internal detail


def test_non_chat_post_is_404(static_dir):
    with _run_server(StubEngine(), static_dir) as port:
        status, body = _post(port, b"{}", path="/other")
    assert status == 404


def test_non_string_text_field_is_400(static_dir):
    with _run_server(StubEngine(), static_dir) as port:
        status, _ = _post(port, json.dumps({"text": 123}).encode())
    assert status == 400


def test_static_get_responsive_while_engine_generating(static_dir):
    slow = SlowEngine(delay=1.0)
    with _run_server(slow, static_dir) as port:
        chat_result = {}

        def do_chat():
            started = time.monotonic()
            status, body = _post(port, json.dumps({"text": "slow"}).encode())
            chat_result["elapsed"] = time.monotonic() - started
            chat_result["status"] = status
            chat_result["body"] = body

        thread = threading.Thread(target=do_chat)
        thread.start()
        time.sleep(0.2)  # let the chat request enter the engine call

        started = time.monotonic()
        status, data = _get(port, "/index.html")
        get_elapsed = time.monotonic() - started

        thread.join(timeout=10)

    assert status == 200 and b"coo ui" in data
    assert get_elapsed < 0.8, "static GET was blocked by the inference lock"
    assert chat_result["status"] == 200
    assert json.loads(chat_result["body"]) == {"response": "slow-ok"}
    assert chat_result["elapsed"] >= 0.9


def test_concurrent_chats_are_serialized_by_inference_lock(static_dir):
    slow = SlowEngine(delay=0.4)
    with _run_server(slow, static_dir) as port:
        results = []

        def do_chat(i):
            started = time.monotonic()
            status, body = _post(port, json.dumps({"text": f"m{i}"}).encode())
            results.append((status, json.loads(body), time.monotonic() - started))

        threads = [threading.Thread(target=do_chat, args=(i,)) for i in range(2)]
        wall_start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        wall = time.monotonic() - wall_start

    assert all(status == 200 and body == {"response": "slow-ok"} for status, body, _ in results)
    # The lock must serialize inference: two 0.4s chats take ~0.8s wall.
    assert wall >= 0.75
    assert slow.max_active == 1


def test_import_has_no_side_effects():
    # FIND-017: importing tools.server must not print a banner, prompt
    # (stdin closed -> input() would raise EOFError) or do any I/O.
    proc = subprocess.run(
        [sys.executable, "-c", "import tools.server; print('IMPORT_OK')"],
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "IMPORT_OK" in proc.stdout
    assert "Starting CooLM" not in proc.stdout
    assert "choice" not in proc.stdout.lower()


def test_cli_help_prints_usage():
    proc = subprocess.run(
        [sys.executable, "-m", "tools.server", "--help"],
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0
    assert "--port" in proc.stdout
    assert "--no-browser" in proc.stdout


def test_bind_failure_exits_cleanly(static_dir, capsys, monkeypatch):
    # Occupy a port, then ask the server to bind it: one-line diagnostic,
    # exit code 1, no traceback (FIND-042). The blocker socket must stay open
    # across run(), and SO_REUSEADDR must be off for the bind to actually
    # fail against an occupied port on Windows.
    monkeypatch.setattr(server.CooServer, "allow_reuse_address", False)
    with contextlib.closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        with pytest.raises(SystemExit) as excinfo:
            server.run(port=port, engine=StubEngine(), static_dir=static_dir,
                       open_browser=False)
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "could not start the server" in out.lower()
    assert "Traceback" not in out
