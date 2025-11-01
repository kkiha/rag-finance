from __future__ import annotations
import math
from typing import List, Sequence
import numpy as np
from langchain_core.documents import Document

def minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return values
    vmin, vmax = min(values), max(values)
    if vmax - vmin < 1e-12:
        return [0.5 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]

def kw_bonus_score(text: str, picked_keywords: Sequence[str], cap_per_kw: int = 1) -> float:
    if not text or not picked_keywords:
        return 0.0
    t = "".join(text.lower().split())
    hits = 0
    for kw in picked_keywords:
        k = "".join((kw or "").lower().split())
        if not k:
            continue
        if k in t:
            hits += 1 if cap_per_kw <= 1 else min(t.count(k), cap_per_kw)
    L = max(1, len(t))
    return hits / (1.0 + math.log1p(L))

def build_hybrid_pre(
    query: str,
    docs: List[Document],
    embedding_model,
    ent_bonus_scale: float = 0.02,
    kw_picked: Sequence[str] = (),
    alpha_kw: float = 0.08,
    cap_per_kw: int = 1,
) -> List[float]:
    q_emb = embedding_model.embed_query(query)
    doc_embs = embedding_model.embed_documents([d.page_content for d in docs])
    sims = np.dot(np.array(doc_embs), np.array(q_emb)).tolist()
    ent_bonus = [ent_bonus_scale * (d.metadata.get("match_strength", 0)) for d in docs]
    kw_raw = [kw_bonus_score(d.page_content, kw_picked, cap_per_kw=cap_per_kw) for d in docs]
    from .hybrid import minmax_norm as _mmn  # self import ok
    kw_norm = _mmn(kw_raw)
    return [s + e + (alpha_kw * k) for s, e, k in zip(sims, ent_bonus, kw_norm)]
