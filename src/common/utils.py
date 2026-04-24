"""Small dependency-light utilities used across modules."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Iterable


def set_seed(seed: int) -> None:
    random.seed(seed)


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def risk_class_from_score(score: float, medium_threshold: float = 0.4, high_threshold: float = 0.7) -> str:
    if score >= high_threshold:
        return "high"
    if score >= medium_threshold:
        return "medium"
    return "low"


def diagnosis_from_probability(probability: float) -> str:
    if probability >= 0.68:
        return "cancer"
    if probability >= 0.42:
        return "precancer"
    return "benign"


def stable_hash_float(text: str, low: float = 0.0, high: float = 1.0) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12 - 1)
    return low + (high - low) * value


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    positive = {key: max(0.0, float(value)) for key, value in scores.items()}
    total = sum(positive.values())
    if total <= 0:
        return {key: 0.0 for key in scores}
    return {key: value / total for key, value in positive.items()}


def fixed_length_embedding(values: Iterable[float], dim: int, seed_text: str) -> list[float]:
    """Project a short numeric vector into a deterministic fixed-length embedding."""

    base = list(values)
    if not base:
        base = [stable_hash_float(seed_text, -0.25, 0.25)]
    embedding: list[float] = []
    for index in range(dim):
        source = base[index % len(base)]
        phase = stable_hash_float(f"{seed_text}:{index}", -0.5, 0.5)
        value = math.tanh(source * (0.7 + (index % 7) * 0.08) + phase)
        embedding.append(round(value, 6))
    return embedding
