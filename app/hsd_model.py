# hsd_model.py — TF SavedModel inference + externalized rule-based overrides
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # force CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"           # TF logs -> ERROR only
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"          # avoid extra init chatter

from pathlib import Path
import json
import base64
import re
import numpy as np
import contextlib, io, sys

# --- Quiet import helpers -----------------------------------------------------
@contextlib.contextmanager
def _squelch_import_stderr():
    """
    Some TF/absl init logs are emitted to raw STDERR before absl is configured,
    e.g. cuDNN/cuBLAS 'factory already registered' and 'computation_placer' lines.
    We temporarily redirect STDERR during import to keep the console clean.
    """
    _old_stderr = sys.stderr
    try:
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stderr = _old_stderr

# Import TensorFlow (+ TF-Text) with stderr silenced to hide early warnings
with _squelch_import_stderr():
    import tensorflow as tf
    try:
        import tensorflow_text  # registers text kernels like CaseFoldUTF8
    except Exception as _e:
        # Re-raise as a friendly error AFTER stderr is restored
        raise RuntimeError(
            "This SavedModel may require TensorFlow Text, but 'tensorflow_text' "
            f"could not be imported.\nOriginal error: {_e}\n"
            "Install matching TF & TF-Text versions."
        )

# After import, clamp down loggers again (covers runtime messages)
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass
try:
    tf.get_logger().setLevel("ERROR")
except Exception:
    pass
try:
    tf.config.set_visible_devices([], "GPU")  # extra safety in case CUDA is present
except Exception:
    pass

# ── MODEL LOCATION ─────────────────────────────────────────────────────────────
MODEL_DIR = "cnn_lstm_hate_speech_bert"  # must contain saved_model.pb + variables/

# ── CLASS MAPPING (confirmed) ─────────────────────────────────────────────────
LABEL_MAP = {0: "hate speech", 1: "offensive", 2: "neutral"}

# ── RULE-BASED OVERRIDES (EXTERNALIZED) ───────────────────────────────────────
_SLUR_RE   = None
_THREAT_RE = None

def _compile_list(lst):
    if not lst:
        return None
    pat = "|".join(lst)
    return re.compile(pat, re.IGNORECASE)

def _load_pack_from_env():
    """Returns dict or None."""
    b64 = os.getenv("HSD_SENSITIVE_PACK_B64")
    path = os.getenv("HSD_SENSITIVE_PACK_PATH")
    if b64:
        data = base64.b64decode(b64).decode("utf-8", "strict")
        return json.loads(data)
    if path:
        p = Path(path)
        if not p.is_file():
            raise RuntimeError(f"HSD_SENSITIVE_PACK_PATH not found: {path}")
        return json.loads(p.read_text(encoding="utf-8"))
    return None

def _init_sensitive_pack():
    global _SLUR_RE, _THREAT_RE
    pack = _load_pack_from_env()
    if pack:
        _SLUR_RE   = _compile_list(pack.get("slurs") or [])
        _THREAT_RE = _compile_list(pack.get("threats") or [])

def set_sensitive_pack(pack: dict):
    """Programmatic way to provide the pack at runtime (e.g., from a UI upload)."""
    global _SLUR_RE, _THREAT_RE
    if not isinstance(pack, dict):
        raise TypeError("set_sensitive_pack expects a dict.")
    _SLUR_RE   = _compile_list(pack.get("slurs") or [])
    _THREAT_RE = _compile_list(pack.get("threats") or [])

_init_sensitive_pack()

def _override_label(text: str, current: str) -> str:
    """
    If a slur pattern matches → 'hate speech'.
    If a threat pattern matches → 'offensive' (tune policy as needed).
    If no pack provided, or no match → return base label.
    """
    t = text.lower()
    if _SLUR_RE and _SLUR_RE.search(t):
        return "hate speech"
    if _THREAT_RE and _THREAT_RE.search(t):
        return "offensive"
    return current

# ── CACHE ─────────────────────────────────────────────────────────────────────
_RELOADED = None
_SERVE_FN = None

# ── CORE LOADING ──────────────────────────────────────────────────────────────
def ensure_model() -> str:
    p = Path(MODEL_DIR)
    if not p.is_dir() or not (p / "saved_model.pb").exists() or not (p / "variables").is_dir():
        raise RuntimeError(
            f"SavedModel not found at '{MODEL_DIR}'. Expected:\n"
            f"{MODEL_DIR}/\n  ├─ saved_model.pb\n  └─ variables/"
        )
    return str(p)

def load_model():
    global _RELOADED, _SERVE_FN
    if _RELOADED is not None:
        return _RELOADED
    model_dir = ensure_model()
    _RELOADED = tf.saved_model.load(model_dir)
    sigs = getattr(_RELOADED, "signatures", None)
    if sigs and "serving_default" in sigs:
        _SERVE_FN = sigs["serving_default"]
    else:
        _SERVE_FN = None
    return _RELOADED

# ── INFERENCE HELPERS ────────────────────────────────────────────────────────
def _call_model(texts):
    m = load_model()
    texts = [t.strip() if isinstance(t, str) else str(t) for t in texts]
    arr = tf.convert_to_tensor(texts).numpy()  # object array of strings
    if _SERVE_FN is not None:
        try:
            return _SERVE_FN(tf.constant(arr))
        except Exception:
            pass
    return m(arr)

def _to_numpy(outputs):
    if isinstance(outputs, dict):
        for k in ("logits", "predictions", "output_0", "Identity", "dense"):
            if k in outputs:
                outputs = outputs[k]
                break
        if isinstance(outputs, dict):
            outputs = next(iter(outputs.values()))
    if hasattr(outputs, "numpy"):
        outputs = outputs.numpy()
    out = np.asarray(outputs)
    if out.ndim == 1:
        out = out[None, :]
    return out

def _infer_raw(texts):
    return _to_numpy(_call_model(texts))

def _softmax(x, T=1.0):
    x = np.array(x, dtype=np.float64) / float(T)
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-12)

# ── PUBLIC API ────────────────────────────────────────────────────────────────
def predict_label(text: str) -> str:
    """Argmax label, then apply rule-based overrides if a pack is loaded."""
    if not isinstance(text, str):
        raise TypeError("predict_label expects a string.")
    scores = _infer_raw([text])[0]
    probs  = _softmax(scores)
    idx = int(np.argmax(probs))
    base = LABEL_MAP.get(idx, "unknown")
    return _override_label(text, base)

def predict_proba(text: str):
    """Return probabilities per class as a dict (sum ≈ 1)."""
    scores = _infer_raw([text])[0]
    p = _softmax(scores)
    return {LABEL_MAP.get(i, f"class_{i}"): float(p[i]) for i in range(p.shape[0])}

# Optional debug helpers
def raw_scores(text: str):
    return _infer_raw([text])[0]

def debug_scores_and_map(text: str):
    return raw_scores(text), LABEL_MAP
