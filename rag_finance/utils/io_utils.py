from __future__ import annotations
import json
import os
from glob import glob
from typing import Any, Dict, Iterable, List, Tuple


def safe_glob(patterns: Iterable[str]) -> List[str]:
    """
    여러 glob 패턴을 받아 중복 없이 파일 경로 리스트를 반환.
    예: safe_glob(["data/raw/**/*.txt", "data/raw/**/*.html"])
    """
    paths: List[str] = []
    seen = set()
    for p in patterns:
        for f in glob(p, recursive=True):
            if f in seen:
                continue
            if os.path.isfile(f):
                seen.add(f)
                paths.append(f)
    return sorted(paths)


def ensure_dir(path: str) -> None:
    """디렉토리가 없으면 생성(하위 경로까지)."""
    os.makedirs(path, exist_ok=True)


def read_text(path: str, encoding: str = "utf-8") -> str:
    """텍스트 파일 읽기(단순)."""
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        return f.read()


def write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    """텍스트 파일 쓰기."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding=encoding) as f:
        f.write(text)


def read_json(path: str, encoding: str = "utf-8") -> Any:
    """JSON 파일 읽기."""
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)


def write_json(path: str, obj: Any, encoding: str = "utf-8", indent: int = 2) -> None:
    """JSON 파일 쓰기."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding=encoding) as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def split_ext(path: str) -> Tuple[str, str]:
    """파일 경로에서 (basename, 확장자) tuple 반환(확장자에 dot 제외)."""
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    return name, ext.lstrip(".").lower()


def guess_source_type(path: str) -> str:
    """
    간단 소스 타입 분류.
    - 경로에 포함된 디렉토리 이름(Report/News/Policy 등)을 우선 고려
    - 그 외에는 확장자 기반 기본값 적용
    """
    lower_path = (path or "").lower()

    if "report" in lower_path:
        return "report"
    if "policy" in lower_path:
        return "policy"
    if "news" in lower_path:
        return "news"

    _name, ext = split_ext(path)
    if ext in {"html", "htm"}:
        return "report"
    return "etc"
