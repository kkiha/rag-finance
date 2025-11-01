from __future__ import annotations
import os
from typing import Any, Dict
from omegaconf import OmegaConf

def load_config(path: str | None = None) -> Dict[str, Any]:
    """
    기본: configs/default.yaml 로드. 경로 인자 주면 그걸 사용.
    """
    if path is None:
        path = os.path.join(os.getcwd(), "configs", "default.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = OmegaConf.load(path)
    # dict로 변환해서 외부 의존성 최소화
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore
