from __future__ import annotations
import os
from typing import Dict, Iterable, List, Tuple
from tqdm import tqdm

from bs4 import BeautifulSoup

from rag_finance.utils.io_utils import (
    safe_glob, read_text, split_ext, guess_source_type
)
from rag_finance.ingestion.cleaning import clean_text


def load_raw_files(
    raw_dir: str,
    patterns: Iterable[str] | None = None
) -> List[str]:
    """
    raw_dir 하위에서 파일 경로 목록을 가져온다.
    patterns 가 없으면 기본으로 txt/html.
    """
    if patterns is None:
        patterns = [
            os.path.join(raw_dir, "**", "*.txt"),
            os.path.join(raw_dir, "**", "*.html"),
            os.path.join(raw_dir, "**", "*.htm"),
        ]
    return safe_glob(patterns)


def extract_text_from_html(html: str) -> str:
    """BeautifulSoup로 HTML → 텍스트 변환."""
    soup = BeautifulSoup(html, "html.parser")
    # 필요시 스크립트/스타일 제거 가능:
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(separator="\n")
    return text.strip()


def load_and_clean_documents(file_paths: Iterable[str]) -> List[Dict]:
    """
    파일 목록을 받아, 텍스트 추출→클리닝→메타데이터 구성까지 반환.
    반환 dict 형식:
    {
      "file_name": str,
      "file_path": str,
      "text": str,
      "text_length": int,
      "file_type": "txt" | "html",
      "source_type": "report" | "etc",
    }
    """
    results: List[Dict] = []
    for fp in tqdm(list(file_paths), desc="[ingestion] load & clean", unit="file"):
        base, ext = split_ext(fp)
        try:
            raw = read_text(fp)
            if ext in ("html", "htm"):
                raw = extract_text_from_html(raw)
            text = clean_text(raw)
            results.append({
                "file_name": os.path.basename(fp),
                "file_path": fp,
                "text": text,
                "text_length": len(text),
                "file_type": ext,
                "source_type": guess_source_type(fp),
            })
        except Exception:
            # 필요시 로깅으로 넘기기
            continue
    return results
