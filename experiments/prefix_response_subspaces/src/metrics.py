from __future__ import annotations

import numpy as np


LOG_TWO = float(np.log(2.0))


def normalized_js_distance(js_distance):
    """Return D_NJS = D_JS / log(2), with D_JS measured in natural-log nats."""
    values = np.asarray(js_distance, dtype=np.float64)
    normalized = values / LOG_TWO
    if np.ndim(js_distance) == 0:
        return float(normalized)
    return normalized


def softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=-1, keepdims=True)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    kl_p = np.sum(p * (np.log(np.clip(p, eps, None)) - np.log(np.clip(m, eps, None))), axis=-1)
    kl_q = np.sum(q * (np.log(np.clip(q, eps, None)) - np.log(np.clip(m, eps, None))), axis=-1)
    return 0.5 * (kl_p + kl_q)


def distribution_metrics(original_logits: np.ndarray, modified_logits: np.ndarray, top_k: int = 5) -> dict[str, float]:
    p, q = softmax(original_logits), softmax(modified_logits)
    eps = 1e-12
    top1 = int(np.argmax(original_logits))
    p_top = set(np.argsort(original_logits)[-top_k:].tolist())
    q_top = set(np.argsort(modified_logits)[-top_k:].tolist())
    return {
        "js": float(js_divergence(p, q)),
        "kl": float(np.sum(p * (np.log(np.clip(p, eps, None)) - np.log(np.clip(q, eps, None))))),
        "top1_agreement": float(top1 == int(np.argmax(modified_logits))),
        "top5_overlap": float(len(p_top & q_top) / top_k),
        "original_top1_logit_difference": float(modified_logits[top1] - original_logits[top1]),
    }
