from __future__ import annotations
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import argparse

from rag_finance.config import load_config
from rag_finance.utils.io_utils import ensure_dir
from rag_finance.ingestion.loaders import load_raw_files, load_and_clean_documents
from rag_finance.chunking.splitter import make_chunks
from rag_finance.indexing.faiss_index import build_and_save_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    raw_dir     = cfg["paths"]["raw_dir"]
    indexes_dir = cfg["paths"]["indexes_dir"]
    ensure_dir(indexes_dir)

    # 1) 로드
    file_paths = load_raw_files(raw_dir)
    print(f"[build_index] found {len(file_paths)} raw files")

    # 2) 클린
    rows = load_and_clean_documents(file_paths)
    print(f"[build_index] cleaned docs: {len(rows)}")

    # 3) 청킹
    chunks = make_chunks(
        rows,
        chunk_size=cfg["chunk"]["size"],
        chunk_overlap=cfg["chunk"]["overlap"],
        min_char_len=cfg["chunk"]["min_len"],
    )
    print(f"[build_index] chunks: {len(chunks)}")

    # 4) 인덱스
    save_path = build_and_save_index(
        chunks,
        indexes_dir=indexes_dir,
        index_name="all",
        embedding_model_name=cfg["embedding"]["model_name"],
        embedding_device=cfg["embedding"]["device"],
        normalize_embeddings=cfg["embedding"]["normalize"],
    )
    print(f"[build_index] index saved to: {save_path}")


if __name__ == "__main__":
    main()
