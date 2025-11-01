from __future__ import annotations
import os, json, re
from typing import List, Tuple

def load_company_keywords(keyword_dir: str, company_name: str) -> List[str]:
    if not company_name:
        return []
    path = os.path.join(keyword_dir, f"{company_name}_keyword.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if company_name in data and isinstance(data[company_name], list):
                return [str(k).strip() for k in data[company_name] if str(k).strip()]
            # dict지만 키워드만 있는 경우
            flat: List[str] = []
            for v in data.values():
                if isinstance(v, list):
                    flat.extend([str(k).strip() for k in v if str(k).strip()])
            return list(dict.fromkeys(flat))
        elif isinstance(data, list):
            return [str(k).strip() for k in data if str(k).strip()]
    except Exception:
        return []
    return []

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).lower()

def select_keywords_for_query(query: str, kw_list: List[str], hard_n: int = 5, soft_n: int = 3) -> Tuple[List[str], List[str]]:
    if not kw_list:
        return [], []
    q = _norm_space(query)
    scored = []
    for kw in kw_list:
        k = str(kw).strip()
        if not k:
            continue
        score = 0
        tokens = re.split(r"[ /,\-]", _norm_space(k))
        for t in tokens:
            if t and t in q:
                score += 1
        score = score / (1 + max(0, len(k) - 8) * 0.05)
        scored.append((k, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    hard = [k for k,_ in scored[:hard_n]]
    soft = [k for k,_ in scored[:soft_n]]
    return hard, soft
