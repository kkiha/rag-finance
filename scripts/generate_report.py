from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional

from groq import Groq

from rag_finance.config import load_config
from rag_finance.indexing.faiss_index import _build_embedding
from rag_finance.llm import generate_finance_report
from rag_finance.llm.report_generator import format_report_sections, parse_report_sections
from rag_finance.retrieval.pipeline import retrieve_with_keywords
from rag_finance.utils.io_utils import read_json, write_text
from rag_finance.utils.pdf_utils import export_report_pdf
from rag_finance.utils.tabular_format import format_tabular_prompt

try:  # 선택 의존성
    from dotenv import load_dotenv  # type: ignore
except OSError:  # pragma: no cover - 일부 플랫폼에서 I/O 에러 발생 가능
    load_dotenv = None
except ImportError:  # pragma: no cover - python-dotenv 미설치 시
    load_dotenv = None


def _load_api_key(explicit: Optional[str], env_file: Optional[str]) -> str:
    if load_dotenv:
        if env_file and os.path.isfile(env_file):
            load_dotenv(env_file)
        else:
            load_dotenv()
    api_key = explicit or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY 환경변수 또는 --api-key가 필요합니다.")
    return api_key


def _print_retrieved_docs(docs, *, max_chars: int) -> None:
    print("\n[retrieved documents]\n" + "-" * 80)
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        meta_fields = []
        for key in ("type", "file_name", "chunk_index", "match_strength", "company", "company_code"):
            value = meta.get(key)
            if value not in (None, ""):
                meta_fields.append(f"{key}={value}")
        meta_line = ", ".join(meta_fields) if meta_fields else "(no metadata)"
        print(f"[{idx}] {meta_line}")

        snippet = (doc.page_content or "").strip().replace("\n", " ")
        if max_chars and len(snippet) > max_chars:
            snippet = snippet[: max_chars - 3].rstrip() + "..."
        print(snippet)
        print("-" * 80)


def _print_messages(messages) -> None:
    print("\n[groq messages]\n")
    for msg in messages:
        role = str(msg.get("role", "")).upper()
        content = msg.get("content", "")
        print(f"[{role}]\n{content}\n")


def _serialize_docs(docs):
    serialized = []
    for doc in docs:
        serialized.append(
            {
                "content": doc.page_content,
                "metadata": doc.metadata or {},
            }
        )
    return serialized


def _load_tabular_payload(tabular_dir: Optional[str], company: Optional[str]) -> Optional[Dict[str, object]]:
    if not tabular_dir:
        return None
    if not os.path.isdir(tabular_dir):
        print(f"[generate_report] tabular_dir가 존재하지 않습니다: {tabular_dir}", file=sys.stderr)
        return None
    if not company:
        print("[generate_report] 기업명을 찾지 못해 tabular 데이터를 건너뜁니다.", file=sys.stderr)
        return None

    normalized = company.replace(" ", "")
    finance_path = os.path.join(tabular_dir, f"finance_{normalized}.json")
    stock_path = os.path.join(tabular_dir, f"stock_{normalized}.json")

    payload: Dict[str, object] = {"company": company}
    loaded_any = False

    if os.path.isfile(finance_path):
        try:
            payload["finance"] = read_json(finance_path)
            loaded_any = True
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[generate_report] 재무 JSON을 읽지 못했습니다 ({finance_path}): {exc}", file=sys.stderr)

    if os.path.isfile(stock_path):
        try:
            payload["stock"] = read_json(stock_path)
            loaded_any = True
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[generate_report] 주가 JSON을 읽지 못했습니다 ({stock_path}): {exc}", file=sys.stderr)

    if not loaded_any:
        print(f"[generate_report] tabular 데이터를 찾지 못했습니다 (company={company}).", file=sys.stderr)
        return None

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a finance report with Groq LLM")
    parser.add_argument("--config", default="configs/default.yaml", help="설정 파일 경로")
    parser.add_argument("--q", required=True, help="사용자 질의")
    parser.add_argument("--topk", type=int, default=10, help="Retrieval 결과 문서 수")
    parser.add_argument("--model", default="llama-3.3-70b-versatile", help="Groq 모델 이름")
    parser.add_argument("--api-key", help="Groq API Key (미지정 시 환경변수 사용)")
    parser.add_argument("--env-file", help=".env 파일 경로")
    parser.add_argument("--examples-dir", help="few-shot JSONL 디렉터리")
    parser.add_argument("--max-examples", type=int, default=1, help="few-shot 샘플 최대 개수")
    parser.add_argument("--no-few-shot", action="store_true", help="few-shot 예시 사용 안 함")
    parser.add_argument("--style-hint", default="", help="추가 스타일 지침")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--output", help="LLM 원문을 저장할 경로")
    parser.add_argument("--context-out", help="Retrieval 컨텍스트 저장 경로")
    parser.add_argument("--messages-out", help="Groq 메시지 JSON 저장 경로")
    parser.add_argument("--docs-out", help="Retrieval 원문 JSON 저장 경로")
    parser.add_argument("--pretty", action="store_true", help="콘솔 출력 시 포맷팅 적용")
    parser.add_argument("--print-context", action="store_true", help="콘솔에 컨텍스트 일부 출력")
    parser.add_argument("--print-docs", action="store_true", help="Retrieval 문서와 스니펫 출력")
    parser.add_argument("--docs-chars", type=int, default=320, help="문서 스니펫 최대 문자 수")
    parser.add_argument("--print-messages", action="store_true", help="Groq에 전달한 메시지 출력")
    parser.add_argument("--quiet", action="store_true", help="Retrieval 진행률 숨김")
    parser.add_argument("--tabular-dir", help="정형 데이터(JSON) 디렉터리")
    parser.add_argument("--pdf-output", help="생성 리포트를 PDF로 저장할 경로")
    args = parser.parse_args()

    try:
        api_key = _load_api_key(args.api_key, args.env_file)
    except RuntimeError as exc:
        parser.error(str(exc))

    client = Groq(api_key=api_key)

    cfg = load_config(args.config)
    embedding_cfg = cfg["embedding"]
    embedding_model = _build_embedding(
        model_name=embedding_cfg["model_name"],
        device=embedding_cfg["device"],
        normalize=embedding_cfg["normalize"],
    )

    # Retrieval → 근거 문서 확보
    docs, debug_info = retrieve_with_keywords(
        query=args.q,
        config=cfg,
        embedding_model=embedding_model,
        topk=args.topk,
        show_progress=not args.quiet,
    )

    if not docs:
        print("[generate_report] 검색 결과가 없습니다.", file=sys.stderr)
        sys.exit(1)

    print(f"[generate_report] company={debug_info.get('company')} code={debug_info.get('code')}")
    print(f"[generate_report] pooled={debug_info.get('pooled')} merged={debug_info.get('merged')} topN={len(docs)}")

    if args.print_docs:
        _print_retrieved_docs(docs, max_chars=max(0, args.docs_chars))

    include_few_shot = not args.no_few_shot

    tabular_payload = _load_tabular_payload(args.tabular_dir, debug_info.get("company"))
    tabular_text = format_tabular_prompt(tabular_payload)
    if tabular_payload:
        finance_rows = len((tabular_payload.get("finance") or {}))
        stock_rows = len(((tabular_payload.get("stock") or {}).get("monthly_prices") or {}))
        print(
            f"[generate_report] tabular 로드: finance_years={finance_rows} monthly_points={stock_rows}"
        )

    report_text, messages, context_text = generate_finance_report(
        client=client,
        query=args.q,
        docs=docs,
        model=args.model,
        few_shot_dir=args.examples_dir,
        few_shot_max_examples=args.max_examples,
        include_few_shot=include_few_shot,
        style_hint=args.style_hint,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        tabular_text=tabular_text,
    )

    if args.print_context:
        preview = context_text[:600]
        print("\n[context preview]\n" + preview + ("..." if len(context_text) > len(preview) else ""))

    display_text = format_report_sections(report_text) if args.pretty else report_text
    print("\n[generated report]\n")
    print(display_text)

    if args.print_messages:
        _print_messages(messages)

    if args.output:
        write_text(args.output, report_text)
        print(f"[generate_report] 리포트 저장: {args.output}")

    if args.pdf_output:
        sections = parse_report_sections(report_text)
        try:
            export_report_pdf(args.pdf_output, sections, tabular_payload)
            print(f"[generate_report] PDF 저장: {args.pdf_output}")
        except RuntimeError as exc:
            print(f"[generate_report] PDF 저장 실패: {exc}", file=sys.stderr)

    if args.context_out:
        write_text(args.context_out, context_text)
        print(f"[generate_report] 컨텍스트 저장: {args.context_out}")

    if args.messages_out:
        payload = {
            "model": args.model,
            "messages": messages,
            "debug": debug_info,
            "tabular_text": tabular_text,
            "tabular_company": debug_info.get("company"),
            "tabular_payload": tabular_payload,
        }
        write_text(args.messages_out, json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"[generate_report] 메시지 로그 저장: {args.messages_out}")

    if args.docs_out:
        documents_payload = {
            "query": args.q,
            "topk": len(docs),
            "documents": _serialize_docs(docs),
        }
        write_text(args.docs_out, json.dumps(documents_payload, ensure_ascii=False, indent=2))
        print(f"[generate_report] Retrieval 원문 저장: {args.docs_out}")


if __name__ == "__main__":
    main()
