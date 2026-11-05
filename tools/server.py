"""Local web server for CooLM: static UI from docs/ + JSON chat API on /chat.

Import-safe by design (FIND-017): importing this module performs no I/O, no
prompts and no downloads; the interactive startup flow lives in main() and
only runs under the __main__ guard.
"""
import argparse
import hashlib
import ipaddress
import json
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

__version__ = "1.1.0"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(REPO_ROOT, "docs")

MAX_BODY_BYTES = 1024 * 1024  # 1 MiB cap on POST /chat bodies (FIND-018)
DOWNLOAD_TIMEOUT = 30
DOWNLOAD_ATTEMPTS = 3

# Pinned release manifest (FIND-008/FIND-041): every downloadable file is
# pinned to an immutable Hugging Face revision and verified by sha256 + size,
# so model and tokenizer are guaranteed to be a matching pair from one
# revision and any corruption or silent re-upload is detected.
#
# FIND-005 WARNING: the pinned model.onnx below is the artifact currently
# published on the Hub. The 2026-11-05 audit proved it was exported from a
# DIFFERENT trained run than best_model.pt (69/70 initializers differ). It is
# served with a printed notice, not silently trusted. Once the owner publishes
# the corrected export (audit-work/remediation-artifacts/PUBLICATION.md,
# produced by tools/model_export.py), update _HF_REVISION plus the sha256 and
# size constants here to the new revision.
_HF_REPO = "MdHussain121/coolm-42M"
_HF_REVISION = "40c37e23c8aaf7448662bbbc51b0def90b7de4be"
_HF_HOST = "huggingface.co"
_HF_BASE = f"https://huggingface.co/{_HF_REPO}/resolve/{_HF_REVISION}"

RELEASE_FILES = {
    "onnx_model": {
        "description": "CooLM ONNX model",
        "url": _HF_BASE + "/model.onnx",
        "dest": os.path.join("docs", "model_hf.onnx"),
        "sha256": "f354a06b8e28f22b54406e15d5ad90c80b743e384faf2d375b5fc47a42741e7d",
        "size": 174399570,
    },
    "tokenizer": {
        "description": "CooLM tokenizer",
        "url": _HF_BASE + "/tokenizer.json",
        "dest": os.path.join("data", "tokenizer_hf.json"),
        "sha256": "669730f57db91651a59bc1ec0e642682d3a764a6700c6359208db4f321c20475",
        "size": 210629,
    },
}

HF_MODEL_NOTICE = (
    "NOTE: the pinned model.onnx is the release currently published on Hugging Face,\n"
    "      which the 2026-11-05 audit found INCONSISTENT with best_model.pt (exported\n"
    "      from a different training run). It is served as-is pending re-publication;\n"
    "      after the corrected export is published, update the RELEASE_FILES constants\n"
    "      in tools/server.py (revision + sha256 + size) — see tools/model_export.py\n"
    "      and audit-work/remediation-artifacts/PUBLICATION.md."
)


def repo_path(rel):
    """Resolve a repo-relative path against the repo root, not the CWD."""
    return rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)


def _print_progress(downloaded, declared, start):
    if declared is None or declared < 8 * 1024 * 1024:
        return
    percent = (downloaded / declared) * 100
    speed = downloaded / max((time.time() - start), 0.001) / (1024 * 1024)
    print(
        f"\rProgress: {percent:.1f}% ({downloaded/(1024*1024):.1f}MB / "
        f"{declared/(1024*1024):.1f}MB) - Speed: {speed:.2f} MB/s",
        end="", flush=True,
    )
    if downloaded >= declared:
        print()


def _assert_download_url_allowed(url):
    """Validate a downloader URL against the pinned-release policy.

    SSRF defense for a localhost-only chat server: the only permitted
    remote target is the pinned Hugging Face release host over https
    (scheme, exact hostname and every resolved address are checked, so
    redirects or DNS changes to private/link-local ranges are refused).
    Plain-http loopback URLs are explicitly allowed as a testing
    affordance for tools/test_downloader.py, which serves over
    127.0.0.1. Anything else raises ValueError before any network I/O.
    """
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname
    if parts.scheme == "https" and host == _HF_HOST:
        pass
    elif parts.scheme == "http" and host in ("127.0.0.1", "localhost"):
        pass
    else:
        raise ValueError(f"download URL not allowed by the pinned-release policy: {url}")
    for _, _, _, _, sockaddr in socket.getaddrinfo(host, None):
        addr = ipaddress.ip_address(sockaddr[0])
        if not (addr.is_global or addr.is_loopback):
            raise ValueError(
                f"download URL resolves to a non-public, non-loopback address "
                f"({addr}); refusing per the pinned-release policy"
            )
    return url


def _download_to_temp(url, temp_path, expected_sha256, expected_size, description, timeout):
    """One download attempt into temp_path. Returns (ok, failure_reason)."""
    try:
        _assert_download_url_allowed(url)
    except ValueError as exc:
        print(str(exc))
        return False, str(exc)
    req = urllib.request.Request(
        url, headers={"User-Agent": f"CooLM-local-server/{__version__}"}
    )
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        print(f"HTTP error downloading {description}: HTTP {exc.code} {exc.reason} ({url})")
        return False, f"HTTP {exc.code} {exc.reason}"
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        print(f"Network error downloading {description}: {reason}")
        return False, f"network error: {reason}"

    hasher = hashlib.sha256()
    total = 0
    declared = response.headers.get("Content-Length")
    try:
        declared = int(declared) if declared is not None else None
    except ValueError:
        declared = None
    start = time.time()
    try:
        # The temp file was created by mkstemp; reopen it by descriptor
        # (no path lookup, no symlink race) and stream the response in.
        with os.fdopen(os.open(temp_path, os.O_WRONLY | os.O_TRUNC), "wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                hasher.update(chunk)
                out.write(chunk)
                _print_progress(total, declared, start)
    except (TimeoutError, ConnectionError, OSError) as exc:
        print(f"Network error while downloading {description}: {exc}")
        return False, f"network error: {exc}"
    finally:
        response.close()

    if declared is not None and total != declared:
        print(
            f"Download incomplete for {description}: got {total} of {declared} bytes "
            "(connection closed before the full transfer)."
        )
        return False, "incomplete transfer"
    if expected_size is not None and total != expected_size:
        print(f"Size mismatch for {description}: got {total} bytes, expected {expected_size}.")
        return False, "size mismatch"
    digest = hasher.hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        print(
            f"Checksum mismatch for {description}: got sha256 {digest}, "
            f"expected {expected_sha256}. The file is corrupt or tampered with."
        )
        return False, "checksum mismatch"
    return True, ""


def download_file(url, dest_path, expected_sha256=None, expected_size=None,
                  description="File", max_attempts=DOWNLOAD_ATTEMPTS,
                  timeout=DOWNLOAD_TIMEOUT):
    """Download url to dest_path atomically, verifying size and checksum.

    Streams into a temporary file in the destination directory and only moves
    it into place (os.replace) after the Content-Length, expected size and
    sha256 checks all pass, so a partial or corrupt file can never reach the
    destination. Returns True on success, False on any failure.
    """
    dest_dir = os.path.dirname(os.path.abspath(dest_path))
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Downloading {description} from {url}")
    last_error = "unknown error"
    for attempt in range(1, max_attempts + 1):
        temp_path = None
        try:
            fd, temp_path = tempfile.mkstemp(
                prefix=".download-", suffix=".part", dir=dest_dir
            )
            os.close(fd)
            ok, last_error = _download_to_temp(
                url, temp_path, expected_sha256, expected_size, description, timeout
            )
            if ok:
                os.replace(temp_path, dest_path)
                temp_path = None
                print(f"Successfully downloaded {description} to {dest_path}!")
                return True
        except OSError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
        if attempt < max_attempts:
            print(f"Retrying download ({attempt + 1}/{max_attempts})...")
            time.sleep(1.0 * attempt)
    print(f"Download failed for {description}: {last_error}")
    return False


def _sha256_of_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_cached_file(entry):
    """Classify a cached file: MISSING / SIZE_MISMATCH / CHECKSUM_MISMATCH / VALID."""
    path = repo_path(entry["dest"])
    if not os.path.exists(path):
        return "MISSING"
    if entry.get("size") is not None and os.path.getsize(path) != entry["size"]:
        return "SIZE_MISMATCH"
    if entry.get("sha256") is not None and _sha256_of_file(path) != entry["sha256"]:
        return "CHECKSUM_MISMATCH"
    return "VALID"


def ensure_release_file(entry, max_attempts=DOWNLOAD_ATTEMPTS):
    """Make sure a manifest entry exists on disk and passes its checks.

    A missing or invalid cache is removed and re-downloaded; a download
    failure never silently leaves the bad cache in place. Returns True only
    when the file is present and verified.
    """
    state = validate_cached_file(entry)
    if state == "VALID":
        return True
    path = repo_path(entry["dest"])
    print(f"Cached {entry['description']} at {path} is {state}; re-downloading.")
    if state != "MISSING":
        try:
            os.remove(path)
        except OSError:
            pass
    return download_file(
        entry["url"], path,
        expected_sha256=entry.get("sha256"),
        expected_size=entry.get("size"),
        description=entry["description"],
        max_attempts=max_attempts,
    )


def make_handler(engine, static_dir, inference_lock=None):
    """Build the request handler class bound to an engine and static dir.

    The engine is captured per-server (no module-level global); /chat calls
    are serialized by inference_lock while static GETs stay concurrent
    (FIND-016).
    """
    lock = inference_lock or threading.Lock()

    class CooHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=static_dir, **kwargs)

        def do_GET(self):
            # Lightweight health probe for the UI status badge (FIND-021):
            # reports whether an inference engine is actually loaded instead
            # of letting the page hardcode a green "Active Engine" badge.
            if self.path.split("?", 1)[0] == "/health":
                engine_available = engine is not None
                max_seq_len = getattr(engine, "max_seq_len", None) if engine_available else None
                self._send_json(200, {
                    "status": "ok",
                    "engine": engine_available,
                    "max_seq_len": max_seq_len,
                })
                return
            try:
                super().do_GET()
            except (ConnectionAbortedError, BrokenPipeError):
                pass  # client vanished mid-transfer; cosmetic (FIND-056)

        def do_HEAD(self):
            try:
                super().do_HEAD()
            except (ConnectionAbortedError, BrokenPipeError):
                pass

        def _send_json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (ConnectionAbortedError, BrokenPipeError):
                pass

        def do_POST(self):
            if self.path != "/chat":
                self._send_json(404, {"error": "not found"})
                return
            self._handle_chat()

        def _handle_chat(self):
            content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if content_type != "application/json":
                self._send_json(415, {"error": "unsupported media type: use application/json"})
                return

            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                self._send_json(400, {"error": "missing Content-Length"})
                return
            try:
                length = int(raw_length)
            except ValueError:
                self._send_json(400, {"error": "invalid Content-Length"})
                return
            if length < 0:
                self._send_json(400, {"error": "invalid Content-Length"})
                return
            if length > MAX_BODY_BYTES:
                self._send_json(413, {"error": "request body too large"})
                return

            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                self._send_json(400, {"error": "malformed JSON body"})
                return
            if not isinstance(data, dict):
                self._send_json(400, {"error": "JSON body must be an object"})
                return
            text = data.get("text", "")
            if not isinstance(text, str):
                self._send_json(400, {"error": "field 'text' must be a string"})
                return
            text = text.strip()
            if not text:
                self._send_json(200, {"response": "coo..."})
                return
            if engine is None:
                self._send_json(503, {"error": "inference engine not available"})
                return

            # Long prompts are truncated inside the engine pipeline (FIND-006);
            # the engine emits a UserWarning that appears in this console.
            print(f"[chat] generating response ({len(text)} chars of user text)...")
            with lock:
                try:
                    response_text = engine.chat(text)
                except (ValueError, TypeError) as exc:
                    print(f"[chat] rejected request: {type(exc).__name__}: {exc}")
                    self._send_json(400, {"error": "invalid request"})
                    return
                except Exception as exc:
                    print(f"[chat] inference failed: {type(exc).__name__}: {exc}")
                    self._send_json(500, {"error": "internal server error"})
                    return
            self._send_json(200, {"response": response_text})

    return CooHandler


class CooServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def run(port=8000, host="127.0.0.1", engine=None, static_dir=None, open_browser=True):
    """Start the HTTP server; blocks until Ctrl+C. Exits 1 on bind failure."""
    static_dir = static_dir or DOCS_DIR
    handler = make_handler(engine, static_dir)
    display_host = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    url = f"http://{display_host}:{port}/index.html"
    try:
        httpd = CooServer((host, port), handler)
    except OSError as exc:
        print(
            f"Error: could not start the server on {host}:{port} "
            f"({exc.strerror or exc}). Is another program already using that port?"
        )
        sys.exit(1)

    print()
    print("========================================")
    print(f" Server running at: {url}")
    print(f" Serving UI from: {static_dir}")
    print(" Press Ctrl+C to stop the server.")
    print("========================================")
    print()

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        httpd.server_close()


def _ask(prompt):
    """input() that returns None instead of crashing on closed stdin."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def _pause_exit(code=1):
    _ask("Press Enter to exit...")
    sys.exit(code)


def _resolve_hf_paths():
    model_entry = RELEASE_FILES["onnx_model"]
    tokenizer_entry = RELEASE_FILES["tokenizer"]
    if (validate_cached_file(model_entry) == "VALID"
            and validate_cached_file(tokenizer_entry) == "VALID"):
        print("Using cached Hugging Face version (pinned revision, checksum-verified).")
    else:
        print("Hugging Face files missing or invalid; downloading the pinned release...")
        model_ok = ensure_release_file(model_entry)
        tokenizer_ok = ensure_release_file(tokenizer_entry)
        if not (model_ok and tokenizer_ok):
            print("\nError: failed to obtain verified Hugging Face model/tokenizer files.")
            local_onnx = repo_path(os.path.join("docs", "model.onnx"))
            local_tokenizer = repo_path(os.path.join("data", "tokenizer.json"))
            if os.path.exists(local_onnx) and os.path.exists(local_tokenizer):
                print("Falling back to the local version...")
                return local_onnx, local_tokenizer, False
            print("No local model found. Cannot start server.")
            return None
    return repo_path(model_entry["dest"]), repo_path(tokenizer_entry["dest"]), True


def _resolve_local_paths():
    onnx_path = repo_path(os.path.join("docs", "model.onnx"))
    tokenizer_path = repo_path(os.path.join("data", "tokenizer.json"))
    if os.path.exists(onnx_path) and os.path.exists(tokenizer_path):
        return onnx_path, tokenizer_path, False
    print("\nError: Local model (docs/model.onnx) or tokenizer (data/tokenizer.json) not found.")
    answer = _ask("Would you like to download and use the pinned Hugging Face version instead? (y/n): ")
    if answer is None:
        return None
    if answer.lower() in ("y", "yes"):
        model_ok = ensure_release_file(RELEASE_FILES["onnx_model"])
        tokenizer_ok = ensure_release_file(RELEASE_FILES["tokenizer"])
        if model_ok and tokenizer_ok:
            return (repo_path(RELEASE_FILES["onnx_model"]["dest"]),
                    repo_path(RELEASE_FILES["tokenizer"]["dest"]), True)
        print("Failed to download verified Hugging Face model files. Cannot start server.")
        return None
    print("Please train your model first using train_coolm.bat before using the local version.")
    return None


def _ask_model_choice():
    print("========================================")
    print(" CooLM Chat Server Initialization")
    print("========================================")
    print("Which model version would you like to use?")
    print("1) Hugging Face version (pinned, checksum-verified download)")
    print("2) Local version (your own trained/exported model)")
    print("========================================")
    choice = ""
    while choice not in ("1", "2"):
        choice = _ask("Enter your choice (1 or 2): ")
        if choice is None:
            print("No input available. Exiting.")
            return None
        if choice not in ("1", "2"):
            print("Please enter 1 or 2.")
    return choice


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="CooLM local web server (static UI + JSON chat API)."
    )
    parser.add_argument("--port", type=int, default=8000,
                        help="port to listen on (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to bind (default: 127.0.0.1, localhost only)")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window automatically")
    parser.add_argument("--model", choices=("local", "hf"), default=None,
                        help="skip the interactive model selection "
                             "(default: ask which version to use)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print("Starting CooLM Local Web Server...")
    if args.model is None:
        choice = _ask_model_choice()
        if choice is None:
            _pause_exit(1)
    else:
        choice = "2" if args.model == "local" else "1"

    if choice == "1":
        resolved = _resolve_hf_paths()
    else:
        resolved = _resolve_local_paths()
    if resolved is None:
        _pause_exit(1)
    onnx_path, tokenizer_path, from_hf = resolved

    if from_hf:
        print("Model + tokenizer pairing is guaranteed by the pinned revision and sha256 checks.")
        print(HF_MODEL_NOTICE)

    try:
        from tools.onnx_chat import CooONNXInference
        engine = CooONNXInference(onnx_path=onnx_path, tokenizer_path=tokenizer_path)
    except Exception as exc:
        print(f"Error initializing ONNX inference: {type(exc).__name__}: {exc}")
        _pause_exit(1)

    run(port=args.port, host=args.host, engine=engine,
        open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
