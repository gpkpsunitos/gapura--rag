from __future__ import annotations

import logging
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from app.models.types import Language

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=r"`load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar\.",
)

_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / ".cache"
_MODEL_PATH = _MODEL_DIR / "lid.176.ftz"

_model: Any = None


def _ensure_model():
    global _model
    if _model is not None:
        return _model

    import fasttext

    if not _MODEL_PATH.exists():
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading fasttext language model to %s", _MODEL_PATH)
        urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar\.",
        )
        _model = fasttext.load_model(str(_MODEL_PATH))
    return _model


# Complexity: Time O(1) | Space O(M)
def preload_models() -> None:
    """Eagerly load the language model to reduce startup/first-request latency."""
    _ensure_model()


def _safe_predict(model, text: str, k: int = 2):
    """Wraps fasttext predict to handle NumPy 2.x copy=False incompatibility."""
    original_array = np.array

    def patched_array(*args, **kwargs):
        kwargs.pop("copy", None)
        return original_array(*args, **kwargs)

    np.array = patched_array
    try:
        return model.predict(text, k=k)
    finally:
        np.array = original_array


# Complexity: Time O(1) | Space O(1) per prediction
def detect_language(text: str, confidence_threshold: float = 0.5) -> Language:
    if not text.strip():
        return Language.EN

    model = _ensure_model()
    labels, scores = _safe_predict(model, text.replace("\n", " ")[:500], k=2)

    top_label = labels[0].replace("__label__", "")
    top_score = float(scores[0])

    if top_score < confidence_threshold:
        return Language.EN

    if top_label == "id":
        return Language.ID
    return Language.EN
