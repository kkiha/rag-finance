from __future__ import annotations
import re
from typing import Iterable, List


_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+")
_ONLY_NUM_RE = re.compile(r"^\d{1,4}[.,]?\d*\s*$")
_SEPARATOR_RE = re.compile(r"^[-=•■●◆▶▷◀]+$")


def normalize_space(s: str) -> str:
    """여러 공백을 단일 공백으로 치환하고 strip."""
    return _WS_RE.sub(" ", (s or "")).strip()


def strip_urls(s: str) -> str:
    """문자열에서 URL 제거."""
    return _URL_RE.sub("", s or "")


def simple_tokenize(s: str) -> List[str]:
    """아주 단순한 토큰화(공백 기준)."""
    return normalize_space(s).split(" ")


def length_penalty(text_len: int) -> float:
    """
    길이 보정용 계수(너무 긴 청크에 유리하지 않도록 1/log(1+len)).
    하이브리드/키워드 보너스에서 사용할 수 있음.
    """
    import math
    L = max(1, int(text_len))
    return 1.0 / (1.0 + math.log1p(L))


def remove_noisy_lines(lines: Iterable[str]) -> List[str]:
    """
    전처리 공통 필터: 빈 줄, 저작권/출처 문구, 구분선, 숫자-only 라인 제거.
    """
    out: List[str] = []
    for line in lines:
        if not line or not line.strip():
            continue
        stripd = line.strip()
        if re.search(r"(기사원문 링크|ⓒ|출처|저작권자|링크|원문|자료:|사진=|이미지=)", stripd):
            continue
        if _SEPARATOR_RE.match(stripd):
            continue
        if _ONLY_NUM_RE.match(stripd):
            continue
        if len(stripd) < 5:
            continue
        out.append(stripd)
    return out
