from __future__ import annotations
from typing import List
from langchain_core.documents import Document

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def mmr_by_text(docs: List[Document], scores: List[float], k: int, lambda_mult: float = 0.5) -> List[int]:
    selected, picked = [], set()
    while len(selected) < min(k, len(docs)):
        best_idx, best_val = -1, -1e9
        for i, d in enumerate(docs):
            if i in picked:
                continue
            div_pen = 0.0
            for j in picked:
                div_pen = max(div_pen, _jaccard(d.page_content, docs[j].page_content))
            mmr = lambda_mult * scores[i] - (1 - lambda_mult) * div_pen
            if mmr > best_val:
                best_val, best_idx = mmr, i
        picked.add(best_idx)
        selected.append(best_idx)
    return selected
