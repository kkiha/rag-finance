from __future__ import annotations
import re
from typing import Iterable, List
from rag_finance.utils.text_utils import normalize_space, strip_urls, remove_noisy_lines


def clean_text(raw_text: str) -> str:
    """
    노이즈 제거용 공통 클리너.
    - URL 제거
    - 저작권/출처/구분선/숫자-only 라인 제거
    - 너무 짧은 라인 제거
    - 공백 정규화
    """
    if not raw_text:
        return ""

    # 1) 줄 단위로 나눠 필터링
    lines = raw_text.splitlines()
    lines = [strip_urls(x) for x in lines]
    lines = remove_noisy_lines(lines)

    # 2) 다시 합치고 공백 정규화
    cleaned = "\n".join(lines).strip()
    cleaned = normalize_space(cleaned.replace("\n", "\n"))
    return cleaned
