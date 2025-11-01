from __future__ import annotations
import os
from typing import Dict, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


def _build_embedding(model_name: str, device: str = "cuda", normalize: bool = True):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize},
    )


def docs_to_langchain(chunks: List[Dict]) -> List[Document]:
    """
    chunking.splitter.make_chunks 결과를 LangChain Document로 변환
    """
    docs: List[Document] = []
    for r in chunks:
        docs.append(
            Document(
                page_content=r["text"],
                metadata={
                    "type": r.get("source_type", "etc"),
                    "file_name": r.get("file_name", ""),
                    "chunk_index": r.get("chunk_index", -1),
                    "chunk_id": r.get("chunk_id", ""),
                    "company": r.get("company", ""),
                    "company_code": r.get("company_code", ""),
                },
            )
        )
    return docs


def build_and_save_index(
    chunks: List[Dict],
    indexes_dir: str = "indexes",
    index_name: str = "all",
    embedding_model_name: str = "jhgan/ko-sroberta-nli",
    embedding_device: str = "cuda",
    normalize_embeddings: bool = True,
) -> str:
    """
    청크 목록 → 임베딩 → FAISS 인덱스 생성 및 저장.
    반환: 저장 경로
    """
    os.makedirs(indexes_dir, exist_ok=True)
    save_path = os.path.join(indexes_dir, index_name)

    embedding = _build_embedding(
        model_name=embedding_model_name,
        device=embedding_device,
        normalize=normalize_embeddings,
    )

    lc_docs = docs_to_langchain(chunks)
    if not lc_docs:
        raise ValueError("No chunks to index.")
    vs = FAISS.from_documents(lc_docs, embedding=embedding)
    vs.save_local(save_path)
    return save_path
