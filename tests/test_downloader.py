"""Downloader fault-injection tests (audit FIND-008/FIND-041).

Every test serves payloads from a tiny local HTTP server on an ephemeral
port — no external network access, and the real 174MB model is never
touched. The matrix covers: clean download, truncation, checksum mismatch,
size mismatch, HTTP errors, timeouts, corrupt-cache recovery, retries and
protection of an already-valid cache.
"""
import contextlib
import hashlib
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tools.server import download_file, ensure_release_file, validate_cached_file

PAYLOAD = b"CooLM-test-payload-" * 4000  # 76000 bytes
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()
OTHER_PAYLOAD = b"tampered-bytes" * 5000
OTHER_SHA = hashlib.sha256(OTHER_PAYLOAD).hexdigest()


class _QuietMixin:
    def log_message(self, *args):
        pass


def _send_bytes(handler, body, status=200, content_length=None):
    handler.send_response(status)
    handler.send_header("Content-Type", "application/octet-stream")
    handler.send_header("Content-Length", str(len(body) if content_length is None else content_length))
    handler.end_headers()
    handler.wfile.write(body)


def _handler_class(respond):
    class _Handler(_QuietMixin, BaseHTTPRequestHandler):
        def do_GET(self):
            respond(self)
    return _Handler


@contextlib.contextmanager
def _serve(respond):
    """Run a threaded stub HTTP server on an ephemeral port."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler_class(respond))
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/file.bin"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _entry(tmp_path, url, sha=PAYLOAD_SHA, size=len(PAYLOAD)):
    return {
        "description": "test file",
        "url": url,
        "dest": str(tmp_path / "cache.bin"),
        "sha256": sha,
        "size": size,
    }


def _assert_clean_failure(tmp_path):
    dest = tmp_path / "cache.bin"
    assert not dest.exists()
    leftovers = [p.name for p in tmp_path.iterdir()]
    assert leftovers == [], f"temp files left behind: {leftovers}"


def test_clean_download_verifies_hash_and_leaves_no_temp(tmp_path, capsys):
    with _serve(lambda h: _send_bytes(h, PAYLOAD)) as url:
        ok = download_file(url, str(tmp_path / "cache.bin"),
                           expected_sha256=PAYLOAD_SHA, expected_size=len(PAYLOAD))
    assert ok
    assert (tmp_path / "cache.bin").read_bytes() == PAYLOAD
    assert [p.name for p in tmp_path.iterdir()] == ["cache.bin"]


def test_truncated_transfer_rejected_by_content_length(tmp_path, capsys):
    # Declares 10MB, sends 64KB, closes cleanly: the Content-Length check
    # alone must catch it even with no pinned hash/size.
    truncated = b"x" * (64 * 1024)
    with _serve(lambda h: _send_bytes(h, truncated, content_length=10 * 1024 * 1024)) as url:
        ok = download_file(url, str(tmp_path / "cache.bin"), max_attempts=1)
    assert not ok
    assert "incomplete" in capsys.readouterr().out
    _assert_clean_failure(tmp_path)


def test_checksum_mismatch_rejected(tmp_path, capsys):
    # Serve different bytes than the pinned hash (tampered mid-flight).
    with _serve(lambda h: _send_bytes(h, OTHER_PAYLOAD)) as url:
        ok = download_file(url, str(tmp_path / "cache.bin"),
                           expected_sha256=PAYLOAD_SHA, max_attempts=1)
    assert not ok
    assert "Checksum mismatch" in capsys.readouterr().out
    _assert_clean_failure(tmp_path)


def test_size_mismatch_rejected(tmp_path, capsys):
    with _serve(lambda h: _send_bytes(h, PAYLOAD)) as url:
        ok = download_file(url, str(tmp_path / "cache.bin"),
                           expected_size=len(PAYLOAD) + 1, max_attempts=1)
    assert not ok
    assert "Size mismatch" in capsys.readouterr().out
    _assert_clean_failure(tmp_path)


def test_http_404_and_500_fail_with_distinct_messages(tmp_path, capsys):
    with _serve(lambda h: _send_bytes(h, b"nope", status=404)) as url:
        ok = download_file(url, str(tmp_path / "cache.bin"), max_attempts=1)
    out = capsys.readouterr().out
    assert not ok
    assert "HTTP 404" in out
    _assert_clean_failure(tmp_path)

    with _serve(lambda h: _send_bytes(h, b"boom", status=500)) as url:
        ok = download_file(url, str(tmp_path / "cache.bin"), max_attempts=1)
    out = capsys.readouterr().out
    assert not ok
    assert "HTTP 500" in out
    assert "HTTP 404" not in out
    _assert_clean_failure(tmp_path)


def test_hanging_server_fails_within_timeout(tmp_path, capsys):
    def hang(handler):
        time.sleep(5)  # accept the connection, never send a byte

    started = time.monotonic()
    with _serve(hang) as url:
        ok = download_file(url, str(tmp_path / "cache.bin"),
                           max_attempts=1, timeout=0.5)
    elapsed = time.monotonic() - started
    assert not ok
    assert elapsed < 20  # bounded, not the full 5s hang
    assert "network error" in capsys.readouterr().out.lower()
    _assert_clean_failure(tmp_path)


def test_corrupt_cache_is_detected_and_replaced(tmp_path, capsys):
    dest = tmp_path / "cache.bin"
    # Same length as the payload so validation reaches the hash check.
    dest.write_bytes(b"j" * len(PAYLOAD))
    entry = _entry(tmp_path, "http://127.0.0.1:1/file.bin")
    assert validate_cached_file(entry) == "CHECKSUM_MISMATCH"
    dest.write_bytes(b"short junk")
    assert validate_cached_file(entry) == "SIZE_MISMATCH"

    # Server available: validation must detect the corruption and re-download.
    dest.write_bytes(b"j" * len(PAYLOAD))
    with _serve(lambda h: _send_bytes(h, PAYLOAD)) as url:
        entry = _entry(tmp_path, url)
        assert ensure_release_file(entry, max_attempts=2)
    assert dest.read_bytes() == PAYLOAD

    # Server unavailable: the corrupt cache must NOT be silently accepted.
    with contextlib.closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        dead_port = sock.getsockname()[1]
    entry = _entry(tmp_path, f"http://127.0.0.1:{dead_port}/file.bin")
    dest.write_bytes(b"k" * len(PAYLOAD))
    assert not ensure_release_file(entry, max_attempts=1)
    out = capsys.readouterr().out
    assert "CHECKSUM_MISMATCH" in out
    assert not dest.exists()  # bad cache removed, not kept


def test_retry_after_first_failure_succeeds(tmp_path):
    state = {"count": 0}

    def flaky(handler):
        state["count"] += 1
        if state["count"] == 1:
            _send_bytes(handler, b"transient server error", status=500)
        else:
            _send_bytes(handler, PAYLOAD)

    with _serve(flaky) as url:
        ok = download_file(url, str(tmp_path / "cache.bin"),
                           expected_sha256=PAYLOAD_SHA, max_attempts=2)
    assert ok
    assert state["count"] == 2
    assert (tmp_path / "cache.bin").read_bytes() == PAYLOAD


def test_failed_redownload_keeps_previous_valid_cache(tmp_path):
    dest = tmp_path / "cache.bin"
    dest.write_bytes(PAYLOAD)  # previously downloaded, valid

    truncated = PAYLOAD[: len(PAYLOAD) // 2]
    with _serve(lambda h: _send_bytes(h, truncated,
                                      content_length=len(PAYLOAD))) as url:
        ok = download_file(url, str(dest),
                           expected_sha256=PAYLOAD_SHA, max_attempts=1)
    assert not ok
    # The interrupted transfer must not have touched the valid cache.
    assert dest.read_bytes() == PAYLOAD
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == PAYLOAD_SHA
    assert [p.name for p in tmp_path.iterdir()] == ["cache.bin"]


def test_release_manifest_pins_revision_and_hashes():
    from tools.server import RELEASE_FILES, _HF_REVISION

    for key, entry in RELEASE_FILES.items():
        assert _HF_REVISION in entry["url"], entry["url"]
        assert len(entry["sha256"]) == 64
        int(entry["sha256"], 16)  # must be hex
        assert isinstance(entry["size"], int) and entry["size"] > 0
    assert RELEASE_FILES["onnx_model"]["sha256"] == (
        "f354a06b8e28f22b54406e15d5ad90c80b743e384faf2d375b5fc47a42741e7d"
    )
    assert RELEASE_FILES["tokenizer"]["sha256"] == (
        "669730f57db91651a59bc1ec0e642682d3a764a6700c6359208db4f321c20475"
    )
