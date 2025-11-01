from __future__ import annotations
from typing import Dict, Tuple

def rrf_fusion(ranks_by_source: Dict[str, Dict[Tuple[str, str], int]], k_const: int = 60) -> Dict[Tuple[str, str], float]:
    fused: Dict[Tuple[str, str], float] = {}
    for rank_map in ranks_by_source.values():
        for key, r0 in rank_map.items():
            fused[key] = fused.get(key, 0.0) + 1.0 / (k_const + (r0 + 1))
    return fused
