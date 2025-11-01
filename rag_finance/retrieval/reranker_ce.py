from __future__ import annotations
import math
from typing import List, Sequence, Tuple
from sentence_transformers import CrossEncoder
import torch

def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def anchor_trim(text: str, aliases: Sequence[str], keywords: Sequence[str], max_chars: int = 1800) -> str:
    if len(text) <= max_chars:
        return text
    low = text.lower()
    anchors = list(aliases) + list(keywords)
    idxs = [low.find(a.lower()) for a in anchors if a]
    idxs = [i for i in idxs if i >= 0]
    if not idxs:
        return text[:max_chars]
    idx = min(idxs)
    half = max_chars // 2
    start = max(0, idx - half)
    end = min(len(text), start + max_chars)
    return text[start:end]

class CrossEncoderReranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", device: str | None = None, batch_size: int = 32, use_sigmoid: bool = True):
        self.model = CrossEncoder(model_name, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.use_sigmoid = use_sigmoid

    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        with torch.inference_mode():
            scores = self.model.predict(pairs, batch_size=self.batch_size, convert_to_numpy=True)
        scores = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        if self.use_sigmoid:
            scores = [_sigmoid(s) for s in scores]
        return scores
