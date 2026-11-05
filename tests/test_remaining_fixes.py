"""Assignment-6 remediation tests: web UI fixes, /health, training resume,
telemetry, and the remaining Medium/Low findings.

UI findings (FIND-021/022/023/043/044/045/046/048, FIND-006) are verified
with static DOM/text assertions; the /health endpoint is exercised over HTTP
with a stub engine; training resume runs a tiny CPU model on tmp data.
"""
import copy
import http.client
import inspect
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch

from conftest import REPO_ROOT, write_inference_tokenizer

HTML_PATH = os.path.join(REPO_ROOT, "docs", "index.html")


def _read_html():
    with open(HTML_PATH, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# UI static checks (docs/index.html)
# ---------------------------------------------------------------------------

def test_ui_accessibility_roles_and_labels():
    html = _read_html()
    # FIND-023: chat log announced to screen readers.
    assert 'role="log"' in html
    assert 'aria-live="polite"' in html
    assert 'aria-label="Chat messages"' in html
    # Typing indicator is a live status region (set in JS).
    assert "setAttribute('role', 'status')" in html or 'setAttribute("role", "status")' in html
    # Input has an accessible name (label + aria-label) and a real form.
    assert 'class="visually-hidden" for="userInput"' in html
    assert 'aria-label="Type a message for Coo"' in html
    assert 'aria-label="Send message"' in html
    assert "<main" in html
    assert 'id="chatForm"' in html
    # Deprecated handler is gone (attribute form); keydown with IME guard is in.
    assert "onkeypress=" not in html
    assert "isComposing" in html
    assert ".visually-hidden" in html


def test_ui_prefers_reduced_motion():
    html = _read_html()
    assert "@media (prefers-reduced-motion: reduce)" in html
    # JS side: physics loop consults the media query and skips rAF updates.
    assert "prefers-reduced-motion: reduce" in html  # matchMedia query in JS
    assert "prefersReducedMotion" in html


def test_ui_no_external_font_requests():
    html = _read_html()
    # FIND-044: the Google Fonts request is gone, replaced by a system stack.
    assert "fonts.googleapis.com" not in html
    assert "system-ui" in html


def test_ui_meta_description_no_longer_claims_in_browser_inference():
    html = _read_html()
    # FIND-044: the false claims are gone...
    assert "Runs completely locally in your browser" not in html
    assert "in your browser using ONNX" not in html
    # ...and the description names the local Python server.
    assert "local Python server" in html


def test_ui_favicon_data_uri():
    html = _read_html()
    assert 'rel="icon"' in html
    assert "data:image/svg+xml" in html


def test_ui_health_probe_wiring():
    html = _read_html()
    # FIND-021: the badge is driven by /health, not hardcoded green.
    assert "'/health'" in html
    assert 'id="serverStatus"' in html
    assert 'class="server-status checking"' in html  # initial honest state
    assert ".server-status.offline" in html
    assert ".status-dot.offline" in html
    # The old hardcoded markup is gone.
    assert '<span>Active Engine</span>' not in html


def test_ui_in_flight_guard_and_timeout():
    html = _read_html()
    # FIND-022: busy flag, disabled controls, AbortController timeout.
    assert "isBusy" in html
    assert "AbortController" in html
    assert "REQUEST_TIMEOUT_MS = 120000" in html
    assert "still generating" in html
    assert "setInputEnabled(false)" in html


def test_ui_response_contract_validation():
    html = _read_html()
    # FIND-046: 200 without a string "response" (or non-JSON body) is an error.
    assert "server returned an unexpected response" in html
    assert "typeof data.response !== 'string'" in html


def test_ui_autoscroll_and_dom_cap():
    html = _read_html()
    # FIND-045: near-bottom check + message cap.
    assert "NEAR_BOTTOM_PX" in html
    assert "MAX_MESSAGES" in html
    assert "scrollHeight - chatBox.scrollTop - chatBox.clientHeight" in html


def test_ui_touch_targets_and_hidden_tab_guard():
    html = _read_html()
    # FIND-048: 44px touch targets on both pill styles...
    assert html.count("min-height: 44px") >= 2
    # ...and no work in hidden tabs.
    assert "document.hidden" in html


def test_ui_long_input_hint():
    html = _read_html()
    # FIND-006: non-blocking hint for long messages.
    assert "LONG_INPUT_CHARS = 1800" in html
    assert "the tail is kept" in html


def test_ui_contrast_values_updated():
    html = _read_html()
    # FIND-043: darkened gradient/button colors with computed-ratio comments.
    assert "#0369a1" in html  # user-bubble gradient start
    assert "#075985" in html  # user-bubble gradient end
    assert "#b91c1c" in html  # send button start
    assert "38bdf8" not in html  # old failing gradient start gone
    # Explicit placeholder rule exists now.
    assert "input::placeholder" in html


# ---------------------------------------------------------------------------
# GET /health (FIND-021)
# ---------------------------------------------------------------------------

from tools import server


class HealthStubEngine:
    max_seq_len = 64

    def chat(self, text):
        return "coo"


@pytest.fixture()
def health_static_dir(tmp_path):
    (tmp_path / "index.html").write_text("<html>coo ui</html>", encoding="utf-8")
    return str(tmp_path)


def _start_server(engine, static_dir):
    handler = server.make_handler(engine, static_dir)
    httpd = server.CooServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


def _get(port, path, timeout=10):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def test_health_endpoint_reports_engine(health_static_dir):
    httpd, port = _start_server(HealthStubEngine(), health_static_dir)
    try:
        status, body = _get(port, "/health")
        assert status == 200
        assert json.loads(body) == {"status": "ok", "engine": True, "max_seq_len": 64}
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_health_endpoint_without_engine(health_static_dir):
    httpd, port = _start_server(None, health_static_dir)
    try:
        status, body = _get(port, "/health")
        assert status == 200
        payload = json.loads(body)
        assert payload == {"status": "ok", "engine": False, "max_seq_len": None}
        # Static serving is unchanged and /health ignores query strings.
        status, _ = _get(port, "/index.html")
        assert status == 200
        status, body = _get(port, "/health?probe=1")
        assert status == 200
        assert json.loads(body)["engine"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# Seeded generation (FIND-034, inference side)
# ---------------------------------------------------------------------------

def test_generate_seed_is_last_parameter_and_defaults_to_none(tiny_model_factory):
    from coolm.model import CooLM
    from coolm.inference import CooInference
    from tools.onnx_chat import CooONNXInference

    for func in (CooLM.generate, CooInference.chat, CooONNXInference.chat):
        params = list(inspect.signature(func).parameters)
        assert params[-1] == "seed", f"{func.__qualname__}: seed must be the last parameter"
        assert inspect.signature(func).parameters["seed"].default is None


def test_generate_seeded_is_deterministic(tiny_model_factory):
    model = tiny_model_factory()
    prompt = torch.tensor([[1, 10, 11, 12]], dtype=torch.long)

    torch.manual_seed(999)  # unrelated global state must not matter
    out1 = model.generate(prompt, max_new_tokens=12, temperature=1.0, top_k=10, seed=123)
    out2 = model.generate(prompt, max_new_tokens=12, temperature=1.0, top_k=10, seed=123)
    assert torch.equal(out1, out2)
    assert out1.size(1) == prompt.size(1) + 12


def test_generate_unseeded_unchanged(tiny_model_factory):
    # Unseeded generation still works and returns the same shape contract.
    model = tiny_model_factory()
    prompt = torch.tensor([[1, 10, 11, 12]], dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=4, temperature=1.0, top_k=10)
    assert out.size(1) == prompt.size(1) + 4
    # Greedy decoding is unchanged by the seed kwarg (argmax either way).
    out_greedy = model.generate(prompt, max_new_tokens=4, temperature=0)
    assert out_greedy.size(1) == prompt.size(1) + 4


def test_chat_seeded_deterministic_torch(tiny_torch_engine):
    engine = tiny_torch_engine
    text = "abc def ghi"
    resp1 = engine.chat(text, temperature=0.5, seed=42)
    resp2 = engine.chat(text, temperature=0.5, seed=42)
    assert resp1 == resp2


def test_chat_seeded_deterministic_onnx(onnx_engine):
    engine = onnx_engine
    text = "abc def ghi"
    resp1 = engine.chat(text, temperature=0.5, seed=42)
    resp2 = engine.chat(text, temperature=0.5, seed=42)
    assert resp1 == resp2


def test_chat_rejects_non_int_seed(tiny_torch_engine, onnx_engine):
    with pytest.raises(TypeError):
        tiny_torch_engine.chat("abc", seed=True)
    with pytest.raises(TypeError):
        onnx_engine.chat("abc", seed=1.5)


# ---------------------------------------------------------------------------
# CWD-relative defaults (FIND-040)
# ---------------------------------------------------------------------------

def test_missing_tokenizer_error_names_the_file(tmp_path):
    from coolm.inference import CooInference

    with pytest.raises(FileNotFoundError) as excinfo:
        CooInference(tokenizer_path=str(tmp_path / "nope" / "tokenizer.json"))
    assert "tokenizer.json" in str(excinfo.value)


def test_onnx_missing_tokenizer_error_names_the_file(tmp_path):
    from tools.onnx_chat import CooONNXInference

    with pytest.raises(FileNotFoundError) as excinfo:
        CooONNXInference(onnx_path=str(tmp_path / "m.onnx"),
                         tokenizer_path=str(tmp_path / "nope" / "tokenizer.json"))
    assert "tokenizer.json" in str(excinfo.value)


def test_repo_default_resolution():
    from tools.onnx_chat import _repo_default
    assert os.path.isabs(_repo_default("data", "tokenizer.json"))
    assert _repo_default("docs", "model.onnx").replace("\\", "/").endswith("CooLM/docs/model.onnx")


# ---------------------------------------------------------------------------
# coolm/prepare_data.py over-length guard (FIND-037)
# ---------------------------------------------------------------------------

def test_tokenize_dataset_raises_on_over_length_row(tmp_path):
    from coolm.prepare_data import tokenize_dataset

    tokenizer_path = write_inference_tokenizer(tmp_path / "tokenizer.json")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    long_input = " ".join(["abc", "def", "ghi", "jhi", "jkl", "ijk",
                           "hij", "gfi", "ehf", "dgc", "cba", "fed"] * 3)
    data_path = tmp_path / "rows.jsonl"
    data_path.write_text(
        json.dumps({"input": "abc def", "output": "ghi."}) + "\n"
        + json.dumps({"input": long_input, "output": "ghi."}) + "\n",
        encoding="utf-8",
    )

    # Row 1 encodes to 11 tokens and fits pad_len 14; row 2 must trip the
    # explicit over-length error naming the row, its length and pad_len.
    with pytest.raises(ValueError, match=r"row 2 encodes to \d+ tokens which exceeds pad_len 14"):
        tokenize_dataset(str(data_path), tokenizer, output_dir=str(tmp_path / "out"), pad_len=14)


# ---------------------------------------------------------------------------
# coolm/train.py: resume, step-0 eval skip, seed (FIND-013/031/032/033/054)
# ---------------------------------------------------------------------------

def _make_tiny_dataset(tmp_path):
    """Tiny .pt dataset built with the real prepare_data pipeline."""
    tokenizer_path = write_inference_tokenizer(tmp_path / "tokenizer.json")
    from tokenizers import Tokenizer
    from coolm.prepare_data import tokenize_dataset

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    words = ["abc", "def", "ghi", "jhi", "jkl", "ijk", "hij", "gfi"]
    data_path = tmp_path / "rows.jsonl"
    data_path.write_text(
        "".join(
            json.dumps({
                "input": f"{words[i % len(words)]} {words[(i + 1) % len(words)]}",
                "output": f"{words[(i + 2) % len(words)]}.",
            }) + "\n"
            for i in range(24)
        ),
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    tokenize_dataset(str(data_path), tokenizer, output_dir=str(data_dir), pad_len=16)
    return data_dir


def _patch_tiny_configs(monkeypatch, max_steps=2):
    import coolm.train as coolm_train
    from coolm.config import ModelConfig, TrainConfig

    model_cfg = ModelConfig(
        vocab_size=128, d_model=32, n_layers=2, n_heads=4,
        ffn_hidden=64, dropout=0.0, max_seq_len=16,
    )
    train_cfg = TrainConfig(
        batch_size=4, learning_rate=1e-3, min_lr=1e-4, weight_decay=0.1,
        warmup_steps=0, max_steps=max_steps, eval_interval=2, save_interval=1,
        log_interval=1, gradient_clip=1.0,
    )
    monkeypatch.setattr(coolm_train, "ModelConfig", lambda: copy.deepcopy(model_cfg))
    monkeypatch.setattr(coolm_train, "TrainConfig", lambda: copy.deepcopy(train_cfg))
    return train_cfg


def test_train_resume_and_step0_eval_skip(tmp_path, monkeypatch, capsys):
    import coolm.train as coolm_train

    data_dir = _make_tiny_dataset(tmp_path)
    ckpt_dir = tmp_path / "checkpoints"
    train_cfg = _patch_tiny_configs(monkeypatch, max_steps=2)

    # Fresh 2-step run with save_interval=1 -> last_model.pt at 2 updates.
    coolm_train.train(seed=123, data_dir=str(data_dir), checkpoint_dir=str(ckpt_dir))
    out1 = capsys.readouterr().out
    assert "using device" in out1
    assert "done. 2 steps in" in out1
    assert "gradscaler skips: 0" in out1  # FIND-033 counter line present

    last_path = ckpt_dir / "last_model.pt"
    assert last_path.exists()
    state = torch.load(last_path, map_location="cpu", weights_only=True)
    assert state["step"] == 2  # completed-update count
    assert "optimizer_state" in state and "scaler_state" in state
    # No eval ran in the 2-step run, so the carried best loss is still +inf.
    assert state["best_val_loss"] == float("inf")
    assert state["config"]["model"]["max_seq_len"] == 16

    # FIND-032: with eval_interval=2, a 0/1-step run must never write
    # best_model.pt (the old code evaluated at step 0 and saved raw init).
    assert not (ckpt_dir / "best_model.pt").exists()

    # A 1-step run in fresh dirs: still no best_model.pt.
    ckpt_dir_1 = tmp_path / "checkpoints_1step"
    train_cfg.max_steps = 1
    coolm_train.train(seed=1, data_dir=str(data_dir), checkpoint_dir=str(ckpt_dir_1))
    out_1 = capsys.readouterr().out
    assert "done. 1 steps in" in out_1
    assert not (ckpt_dir_1 / "best_model.pt").exists()

    # FIND-013: resume from last_model.pt and continue the schedule to 4.
    train_cfg.max_steps = 4
    coolm_train.train(resume=str(last_path), data_dir=str(data_dir), checkpoint_dir=str(ckpt_dir))
    out2 = capsys.readouterr().out
    assert "resuming from" in out2
    assert "restored step 2" in out2  # continues from the saved step
    # First logged iteration of the resumed run is step 2, not 0.
    assert "\n     2 |" in out2 or out2.splitlines()[-1].startswith("  done")
    assert "done. 4 steps in" in out2  # FIND-031: actual completed count
    state2 = torch.load(last_path, map_location="cpu", weights_only=True)
    assert state2["step"] == 4
    # The resumed run passed step 2 (2 % 2 == 0 and step > 0) -> eval + best.
    assert (ckpt_dir / "best_model.pt").exists()
    # best_model.pt keeps its published-artifact format (ModelConfig instance)
    # and loads through the single safe checkpoint loader.
    from coolm.checkpoint import load_checkpoint

    best = load_checkpoint(ckpt_dir / "best_model.pt")
    assert "model_state" in best and "model_config" in best


def test_train_resume_missing_file_named(tmp_path, monkeypatch, capsys):
    import coolm.train as coolm_train

    data_dir = _make_tiny_dataset(tmp_path)
    _patch_tiny_configs(monkeypatch, max_steps=1)
    coolm_train.train(resume=str(tmp_path / "missing" / "last_model.pt"),
                      data_dir=str(data_dir), checkpoint_dir=str(tmp_path / "ck"))
    out = capsys.readouterr().out
    assert "resume checkpoint not found" in out
    assert "last_model.pt" in out


# ---------------------------------------------------------------------------
# generate_data.py cap-exhaustion telemetry (FIND-036)
# ---------------------------------------------------------------------------

def test_generate_data_tiny_run_telemetry_and_byte_identity(tmp_path):
    script = os.path.join(REPO_ROOT, "generate_data.py")
    runs = []
    for i in range(2):
        run_dir = tmp_path / f"run{i}"
        run_dir.mkdir()
        proc = subprocess.run(
            [sys.executable, script, "200"],
            cwd=str(run_dir),
            capture_output=True, text=True, timeout=300,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        assert proc.returncode == 0, proc.stderr
        assert "uniqueness retry cap exhaustions (duplicate inputs emitted): 0" in proc.stdout
        out_file = run_dir / "pigeon_data.jsonl"
        assert out_file.exists()
        runs.append(out_file.read_bytes())

    # FIND-036 constraint: telemetry only — output content byte-identical.
    assert runs[0] == runs[1]
    lines = runs[0].decode("utf-8").splitlines()
    assert len(lines) == 200
    first = json.loads(lines[0])
    assert set(first) == {"input", "output", "category"}


def test_generate_dataset_counter_in_process():
    import generate_data

    samples, cap_exhaustions = generate_data.generate_dataset(150)
    assert len(samples) == 150
    assert cap_exhaustions == 0
    assert len({s["input"] for s in samples}) == 150


# ---------------------------------------------------------------------------
# requirements.txt / LICENSE / CI (FIND-015, FIND-020, FIND-025)
# ---------------------------------------------------------------------------

def test_requirements_pins_and_missing_deps():
    with open(os.path.join(REPO_ROOT, "requirements.txt"), encoding="utf-8") as f:
        content = f.read()
    for pkg in ("torch>=", "tokenizers>=", "onnxruntime>=", "onnx>=", "huggingface_hub>="):
        assert pkg in content
    for upper in ("<", ):
        assert upper in content  # version ranges are bounded


def test_license_file_exists():
    path = os.path.join(REPO_ROOT, "LICENSE")
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        content = f.read()
    assert "MIT License" in content
    assert "MdHussain121" in content


def test_ci_workflow_exists():
    path = os.path.join(REPO_ROOT, ".github", "workflows", "tests.yml")
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        content = f.read()
    assert "pytest" in content
    assert "actions/setup-python" in content
    assert "download.pytorch.org/whl/cpu" in content


# ---------------------------------------------------------------------------
# tools/export_hf.py (FIND-024 remainders)
# ---------------------------------------------------------------------------

def test_export_hf_requires_repo_id():
    proc = subprocess.run(
        [sys.executable, os.path.join("tools", "export_hf.py")],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode != 0
    assert "--repo_id" in (proc.stderr + proc.stdout)


def test_export_hf_has_no_token_flag():
    proc = subprocess.run(
        [sys.executable, os.path.join("tools", "export_hf.py"),
         "--repo_id", "someone/model", "--token", "leaky-secret"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode != 0
    assert "unrecognized arguments" in (proc.stderr + proc.stdout)
    assert "leaky-secret" not in proc.stdout


def test_export_hf_no_placeholder_default(monkeypatch, capsys):
    from tools import export_hf

    # repo_id is now a required positional parameter (no placeholder default).
    params = inspect.signature(export_hf.push_to_hub).parameters
    assert "repo_id" in params
    assert params["repo_id"].default is inspect.Parameter.empty

    src_path = inspect.getsourcefile(export_hf)
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    assert "username/coolm-42M" not in src
    assert "your_username" not in src

    # Without HF_TOKEN nothing is uploaded and the error names the env var.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    rc = export_hf.push_to_hub(repo_id="someone/model")
    out = capsys.readouterr().out
    assert rc == 1
    assert "HF_TOKEN" in out
    # No placeholder upload was attempted (token check precedes everything).
    assert "Creating repository" not in out
    assert "Uploading" not in out
