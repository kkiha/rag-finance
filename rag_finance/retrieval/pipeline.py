from __future__ import annotations
import os
from typing import Any, Dict, List, Tuple

from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag_finance.entities.company_maps import extract_company_from_query
from rag_finance.entities.keyword_store import load_company_keywords, select_keywords_for_query
from rag_finance.retrieval.filters import dedup_docs, contains_by_name_or_code, text_contains_company
from rag_finance.retrieval.rrf import rrf_fusion
from rag_finance.retrieval.hybrid import build_hybrid_pre, minmax_norm
from rag_finance.retrieval.reranker_ce import CrossEncoderReranker, anchor_trim
from rag_finance.retrieval.mmr import mmr_by_text

def _doc_key(d: Document) -> Tuple[str, str]:
    return (str(d.metadata.get("file_name", "")), str(d.metadata.get("chunk_index", "")))

def retrieve_with_keywords(
    query: str,
    config: Dict[str, Any],
    embedding_model,  # 이미 build_index에 사용한 동일 모델 인스턴스 or 동일 설정으로 생성
    topk: int = 10,
    show_progress: bool = True,
) -> Tuple[List[Document], Dict[str, Any]]:
    paths = config["paths"]
    retrieval = config["retrieval"]
    kw_cfg = retrieval["keywords"]
    ce_cfg = retrieval["ce"]

    indexes_dir = paths["indexes_dir"]
    keyword_dir = paths["keyword_dir"]

    # 1) 인덱스 로드
    vs_all = FAISS.load_local(
        os.path.join(indexes_dir, "all"),
        embedding_model,
        allow_dangerous_deserialization=True
    )

    # 2) 회사/코드 + 키워드
    q_name, q_code = extract_company_from_query(query)
    company_keywords = load_company_keywords(keyword_dir, q_name)
    kw_hard, kw_soft = select_keywords_for_query(
        query, company_keywords,
        hard_n=kw_cfg.get("hard_n", 5),
        soft_n=kw_cfg.get("soft_n", 3),
    )
    aliases = [x for x in {q_name, q_code} if x]

    # 3) 듀얼 리트리벌: BM25(하드 확장), FAISS(소프트 확장)
    bm25_query = query + (" " + " ".join(aliases) if aliases else "") + (" " + " ".join(kw_hard) if kw_hard else "")
    faiss_query = query + (f" (중점:{', '.join(kw_soft)})" if kw_soft else "")

    faiss_ret = vs_all.as_retriever(search_kwargs={"k": retrieval["pool_k_faiss"]})
    # BM25는 vs_all 내부 docstore를 그대로 활용
    bm25_ret = BM25Retriever.from_documents(vs_all.docstore._dict.values())
    bm25_ret.k = retrieval["pool_k_bm25"]

    faiss_pool = faiss_ret.get_relevant_documents(faiss_query)
    bm25_pool = bm25_ret.get_relevant_documents(bm25_query)
    pooled = dedup_docs(faiss_pool + bm25_pool)
    if not pooled:
        return [], {"note": "no pooled", "company": q_name, "code": q_code}

    # 4) 타입 분리 + 엔티티 필터(리포트 강/기타 약)
    reports_pooled = [d for d in pooled if (d.metadata or {}).get("type") == "report"]
    others_pooled  = [d for d in pooled if (d.metadata or {}).get("type") != "report"]

    report_filtered: List[Document] = []
    if q_name or q_code:
        for d in reports_pooled:
            ok, s = contains_by_name_or_code(d, q_name, q_code)
            if ok:
                d.metadata["match_strength"] = s
                report_filtered.append(d)
    else:
        report_filtered = reports_pooled
        for d in report_filtered:
            d.metadata["match_strength"] = 0

    if len(report_filtered) < retrieval["min_needed_report"] and (q_name or q_code):
        if q_name:
            name_relaxed = [d for d in reports_pooled if (d.metadata or {}).get("company") == q_name]
            for d in name_relaxed:
                d.metadata["match_strength"] = max(d.metadata.get("match_strength", 0), 1)
            report_filtered = dedup_docs(report_filtered + name_relaxed)
        if len(report_filtered) < retrieval["min_needed_report"]:
            for d in reports_pooled:
                d.metadata["match_strength"] = max(d.metadata.get("match_strength", 0), 0)
            report_filtered = dedup_docs(report_filtered + reports_pooled)

    report_candidates = report_filtered[:retrieval["pool_k_report"]]

    other_candidates: List[Document] = []
    if (q_name or q_code):
        strict: List[Document] = []
        for d in others_pooled:
            meta = d.metadata or {}
            d_name = (meta.get("company") or "").strip()
            d_code = (meta.get("company_code") or "").strip()
            meta_strength = 2 if (q_code and d_code == q_code) else (1 if (q_name and d_name == q_name) else 0)
            ok, text_strength = text_contains_company(d.page_content, q_name, q_code)
            strength = max(meta_strength, text_strength)
            if strength > 0:
                d.metadata["match_strength"] = strength
                strict.append(d)
        other_candidates = strict
        if len(other_candidates) < retrieval["min_needed_other"] and q_name:
            relax: List[Document] = []
            for d in others_pooled:
                ok, s = text_contains_company(d.page_content, q_name, None)
                if ok:
                    d.metadata["match_strength"] = max(d.metadata.get("match_strength", 0), max(1, s))
                    relax.append(d)
            other_candidates = dedup_docs(other_candidates + relax)
        if len(other_candidates) < retrieval["min_needed_other"]:
            for d in others_pooled:
                d.metadata["match_strength"] = max(d.metadata.get("match_strength", 0), 0)
            other_candidates = dedup_docs(other_candidates + others_pooled)
    else:
        other_candidates = others_pooled
        for d in other_candidates:
            d.metadata["match_strength"] = 0

    other_candidates = other_candidates[:retrieval["pool_k_other"]]

    merged = dedup_docs(report_candidates + other_candidates)
    if not merged:
        return [], {"note": "no merged", "company": q_name, "code": q_code}

    # 5) RRF
    faiss_rank = { _doc_key(d): i for i, d in enumerate(faiss_pool) }
    bm25_rank  = { _doc_key(d): i for i, d in enumerate(bm25_pool)  }
    ranks_by_source = {
        "faiss": { _doc_key(d): faiss_rank.get(_doc_key(d), 10**9) for d in merged if _doc_key(d) in faiss_rank },
        "bm25" : { _doc_key(d): bm25_rank.get(_doc_key(d), 10**9)  for d in merged if _doc_key(d) in bm25_rank  },
    }
    rrf_scores = rrf_fusion(ranks_by_source, k_const=retrieval["rrf_k_const"])
    rrf_sorted = sorted([(i, rrf_scores.get(_doc_key(d), 0.0)) for i, d in enumerate(merged)],
                        key=lambda x: x[1], reverse=True)

    # 6) 사전 하이브리드 (임베딩 + 엔티티 + 키워드 보너스)
    picked_for_bonus = kw_hard or kw_soft
    hybrid_pre = build_hybrid_pre(
        query=query,
        docs=merged,
        embedding_model=embedding_model,
        ent_bonus_scale=0.02,
        kw_picked=picked_for_bonus,
        alpha_kw=kw_cfg.get("alpha_kw", 0.08),
        cap_per_kw=kw_cfg.get("cap_per_kw", 1),
    )

    # 7) CE 리랭크 (상위 N)
    ce_enabled = ce_cfg.get("enable", True)
    ce_alpha = ce_cfg.get("alpha", 0.7)

    if ce_enabled:
        take_top_n = min(ce_cfg.get("take_top_n", 150), len(rrf_sorted))
        top_indices = [i for i, _ in rrf_sorted[:take_top_n]]

        ce = CrossEncoderReranker(
            model_name=ce_cfg["model_name"],
            device=ce_cfg["device"],
            batch_size=ce_cfg["batch_size"],
            use_sigmoid=ce_cfg["use_sigmoid"],
        )

        pairs = []
        ce_hint = ", ".join(kw_soft[:3]) if kw_soft else ""
        loop = top_indices if not show_progress else tqdm(top_indices, desc="[rerank] CE scoring", unit="doc")
        for i in loop:
            txt = anchor_trim(merged[i].page_content, aliases, kw_soft, max_chars=1800)
            qtext = (
                f"타깃 기업: {', '.join(aliases) if aliases else 'N/A'}\n"
                f"중점 키워드(참고): {ce_hint if ce_hint else '없음'}\n"
                f"질의: {query}"
            )
            pairs.append((qtext, txt))

        ce_scores = ce.predict(pairs)
        hyb_norm = minmax_norm([hybrid_pre[i] for i in top_indices])
        ce_norm  = minmax_norm(ce_scores)
        fused    = [ce_alpha * h + (1 - ce_alpha) * c for h, c in zip(hyb_norm, ce_norm)]
        fused_order = sorted(list(zip(top_indices, fused)), key=lambda x: x[1], reverse=True)

        ordered_docs   = [merged[i] for i, _ in fused_order]
        ordered_scores = [s for _, s in fused_order]
    else:
        # CE 비활성화 시 hybrid_pre 점수를 그대로 사용하여 랭킹 구성
        hyb_norm_all = minmax_norm(hybrid_pre)
        fused_order = sorted([(i, hyb_norm_all[i]) for i in range(len(merged))], key=lambda x: x[1], reverse=True)
        ordered_docs   = [merged[i] for i, _ in fused_order]
        ordered_scores = [score for _, score in fused_order]

    # 8) (옵션) MMR
    if ce_cfg.get("apply_mmr_after", True):
        final_idx = mmr_by_text(ordered_docs, ordered_scores, k=topk, lambda_mult=ce_cfg.get("mmr_lambda", 0.5))
        final_docs = [ordered_docs[i] for i in final_idx]
    else:
        final_docs = ordered_docs[:topk]

    dbg = {
        "company": q_name, "code": q_code,
        "kw_hard": kw_hard, "kw_soft": kw_soft,
        "pooled": len(pooled), "merged": len(merged),
        "rrf_topN": len(ordered_docs),
        "alpha_kw": kw_cfg.get("alpha_kw", 0.08), "ce_alpha": ce_alpha,
        "ce_enabled": ce_enabled,
    }
    return final_docs, dbg
