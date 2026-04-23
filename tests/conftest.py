"""
conftest.py — pytest configuration for the ALPR test suite.

Adds the necessary paths to sys.path so that backend modules can be
imported in tests. Also mocks heavy ML libraries (torch, transformers,
peft) so that tests can run on the CI runner without downloading any
model weights.
"""
import sys
import os
import types

# ── Path setup ──────────────────────────────────────────────────────────────
# Allow tests to import from ALPR-app/backend/
REPO_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR  = os.path.join(REPO_ROOT, "ALPR-app", "backend")
SCRIPTS_DIR  = os.path.join(REPO_ROOT, "scripts")
TRAINING_DIR = os.path.join(REPO_ROOT, "training")

for path in [REPO_ROOT, BACKEND_DIR, SCRIPTS_DIR, TRAINING_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)


# ── Stub out heavy ML dependencies ──────────────────────────────────────────
# These stubs let modules be imported for testing without downloading weights.

def _make_stub(name):
    """Create a minimal stub module."""
    mod = types.ModuleType(name)
    mod.__spec__ = None
    return mod


_torch_stub = _make_stub("torch")
_torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
_torch_stub.nn   = types.SimpleNamespace(
    utils=types.SimpleNamespace(clip_grad_norm_=lambda *a, **k: None)
)
_torch_stub.optim = types.SimpleNamespace(
    AdamW=lambda *a, **k: types.SimpleNamespace(
        zero_grad=lambda: None, step=lambda: None
    )
)

for name in [
    "torch", "torch.nn", "torch.nn.utils", "torch.optim",
    "transformers", "peft",
    "easyocr", "ultralytics", "nafnetlib",
    "cv2",
]:
    if name not in sys.modules:
        sys.modules[name] = _make_stub(name)

sys.modules["torch"] = _torch_stub
