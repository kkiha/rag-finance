from __future__ import annotations
import argparse
import os

from rag_finance.config import load_config
from rag_finance.indexing.faiss_index import _build_embedding  # 재사용
from rag_finance.retrieval.pipeline import retrieve_with_keywords

def _print_results(docs, query: str, max_len: int = 320):
    print("=" * 100)
    print(f"검색 쿼리: {query}")
    print(f"총 결과: {len(docs)}")
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        text = d.page_content.strip()
        if len(text) > max_len:
            text = text[:max_len] + "..."
        print("-" * 100)
        print(f"[{i}] {meta.get('file_name', '?')}  | chunk={meta.get('chunk_index','?')}  | type={meta.get('type','?')}")
        print(text)
    print("=" * 100)

def main():
    ap = argparse.ArgumentParser(prog="rag-finance")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # retrieve
    sp_r = sub.add_parser("retrieve", help="Run retrieval (BM25+FAISS -> RRF -> CE -> (MMR))")
    sp_r.add_argument("--config", type=str, default="configs/default.yaml")
    sp_r.add_argument("--q", type=str, required=True, help="query text")
    sp_r.add_argument("--topk", type=int, default=10)

    args = ap.parse_args()

    if args.cmd == "retrieve":
        cfg = load_config(args.config)
        emb_cfg = cfg["embedding"]
        embedding = _build_embedding(
            model_name=emb_cfg["model_name"],
            device=emb_cfg["device"],
            normalize=emb_cfg["normalize"],
        )
        docs, dbg = retrieve_with_keywords(
            query=args.q,
            config=cfg,
            embedding_model=embedding,
            topk=args.topk,
            show_progress=True,
        )
        print(f"[dbg] {dbg}")
        _print_results(docs, query=args.q)

if __name__ == "__main__":
    main()
