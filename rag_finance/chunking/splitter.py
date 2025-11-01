from __future__ import annotations
from typing import Dict, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from rag_finance.entities.company_maps import resolve_company_from_text

def make_chunks(
    cleaned_docs: List[Dict],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    min_char_len: int = 300,
) -> List[Dict]:
    """
    ingestion.loaders.load_and_clean_documents 결과(List[dict])를 입력 받아
    문서별로 청킹한 리스트를 반환.
    반환 아이템 예:
    {
      "file_name": str,
      "source_type": "report" | "etc",
      "chunk_index": int,
      "text": str,
    }
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    out: List[Dict] = []
    for row in tqdm(cleaned_docs, desc="[chunking] split docs", unit="doc"):
        text = row.get("text", "") or ""
        if not text:
            continue
        source_type = row.get("source_type", "etc")

        company_name = ""
        company_code = ""
        if source_type == "report":
            company_name, company_code = resolve_company_from_text(text)

        chunks = splitter.split_text(text)
        chunks = [c for c in chunks if len(c.strip()) >= min_char_len]
        for i, ck in enumerate(chunks):
            out.append({
                "file_name": row["file_name"],
                "source_type": source_type,
                "chunk_index": i,
                "text": ck,
                "chunk_id": f"{row['file_name']}_chunk_{i}",
                "company": company_name,
                "company_code": company_code,
            })
    return out
