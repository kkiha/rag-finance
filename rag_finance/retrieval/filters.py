from __future__ import annotations
import re
from typing import Iterable, List, Tuple
from langchain_core.documents import Document

def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).lower()

def text_contains_company(text: str, q_name: str, q_code: str) -> Tuple[bool, int]:
    if not text:
        return False, 0
    t = _norm(text)
    if q_code and re.search(rf"(?<!\d){re.escape(_norm(q_code))}(?!\d)", t):
        return True, 2
    if q_name and _norm(q_name) in t:
        return True, 1
    return False, 0

def contains_by_name_or_code(doc: Document, q_name: str, q_code: str) -> Tuple[bool, int]:
    meta = doc.metadata or {}
    d_name = (meta.get("company") or "").strip()
    d_code = (meta.get("company_code") or "").strip()
    if q_code:
        if d_code and d_code == q_code:
            return True, 2
        if q_name and d_name == q_name:
            return True, 1
        return False, 0
    if q_name:
        return (d_name == q_name, 1 if d_name == q_name else 0)
    return True, 0

def dedup_docs(docs: Iterable[Document]) -> List[Document]:
    seen = set()
    out: List[Document] = []
    for d in docs:
        key = (str(d.metadata.get("file_name", "")), str(d.metadata.get("chunk_index", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out
