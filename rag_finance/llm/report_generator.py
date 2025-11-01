from __future__ import annotations

import json
import os
import textwrap
from typing import Dict, List, Optional, Sequence, Tuple

from groq import Groq
from langchain_core.documents import Document


def documents_to_context(
    docs: Sequence[Document],
    *,
    max_chars_per_doc: int = 1200,
    max_total_chars: int = 12_000,
    include_metadata: bool = True,
) -> str:
    """Convert retrieved documents into a compact context string."""
    fragments: List[str] = []
    total_chars = 0

    for idx, doc in enumerate(docs, start=1):
        text = (doc.page_content or "").strip()
        if not text:
            continue

        if max_chars_per_doc and len(text) > max_chars_per_doc:
            text = text[: max_chars_per_doc - 3].rstrip() + "..."

        header = ""
        if include_metadata:
            meta = doc.metadata or {}
            header = "[문서 {idx}] type={type} file={file} chunk={chunk} match={match}".format(
                idx=idx,
                type=meta.get("type", "?"),
                file=meta.get("file_name", "?"),
                chunk=meta.get("chunk_index", "?"),
                match=meta.get("match_strength", "?"),
            )

        fragment = f"{header}\n{text}" if header else text
        if max_total_chars and total_chars + len(fragment) > max_total_chars:
            remain = max_total_chars - total_chars
            fragment = fragment[: max(remain, 0)]
            if fragment:
                fragments.append(fragment.rstrip() + "...")
            break

        fragments.append(fragment)
        total_chars += len(fragment)

    return "\n\n".join(fragments)


def load_few_shot_examples(jsonl_dir: Optional[str], max_examples: int = 1) -> List[Tuple[str, str]]:
    """Load few-shot pairs from JSONL files."""
    if not jsonl_dir:
        return []
    if not os.path.isdir(jsonl_dir):
        return []

    examples: List[Tuple[str, str]] = []
    for entry in sorted(os.listdir(jsonl_dir)):
        if not entry.endswith(".jsonl"):
            continue
        path = os.path.join(jsonl_dir, entry)
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(examples) >= max_examples:
                        return examples
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    prompt = (item.get("input") or "").strip()
                    output = (item.get("output") or "").strip()
                    if prompt and output:
                        examples.append((prompt, output))
        except OSError:
            continue
    return examples


def build_messages(
    *,
    context_text: str,
    query: str,
    few_shot_examples: Sequence[Tuple[str, str]] | None = None,
    style_hint: str = "",
) -> List[Dict[str, str]]:
    """Build Groq chat messages using the unified prompt template."""
    system_lines = [
        "너는 한국어를 사용하는 증권사 애널리스트야.",
        "다음 표준 형식만 사용하여 특정 기업의 한국어 리포트를 작성해야 해.",
        "[Title]",
        "[Summary]",
        "[Table]",
        "[Analysis]",
        "[Opinion]",
        "형식 외의 추가 설명은 출력하지 마.",
        "출처 표시는 하지 마.",
        "숫자(실적, 성장률, 가이던스 등)는 가능하면 정량적으로 제시하고, 출처 추정/상상은 하지 마.",
    ]
    if style_hint.strip():
        system_lines.append(style_hint.strip())

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "\n".join(system_lines)}
    ]

    if few_shot_examples:
        for shot_input, shot_output in few_shot_examples:
            messages.append({"role": "user", "content": shot_input})
            messages.append({"role": "assistant", "content": shot_output})

    user_prompt = (
        "[참고 문서]\n"
        "아래는 최신 수집 문서에서 발췌한 내용이야. 리포트 작성시 이 정보를 우선적으로 활용해 줘.\n\n"
        f"{context_text.strip()}\n\n"
        "[질문]\n"
        f"{query.strip()}"
    )
    messages.append({"role": "user", "content": user_prompt})
    return messages


def generate_finance_report(
    *,
    client: Groq,
    query: str,
    docs: Sequence[Document],
    model: str = "llama3-70b-8192",
    few_shot_dir: Optional[str] = None,
    few_shot_max_examples: int = 1,
    include_few_shot: bool = True,
    style_hint: str = "",
    temperature: float = 0.1,
    top_p: float = 0.95,
    max_tokens: int = 1024,
    max_chars_per_doc: int = 1200,
    max_total_chars: int = 12_000,
) -> Tuple[str, List[Dict[str, str]], str]:
    """Run the Groq completion call and return (report_text, sent_messages, context_text)."""
    context_text = documents_to_context(
        docs,
        max_chars_per_doc=max_chars_per_doc,
        max_total_chars=max_total_chars,
        include_metadata=True,
    )

    few_shot_examples: Sequence[Tuple[str, str]] = []
    if include_few_shot:
        few_shot_examples = load_few_shot_examples(
            few_shot_dir,
            max_examples=few_shot_max_examples,
        )

    messages = build_messages(
        context_text=context_text,
        query=query,
        few_shot_examples=few_shot_examples,
        style_hint=style_hint,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    report_text = response.choices[0].message.content.strip()
    return report_text, messages, context_text


def format_report_sections(text: str, width: int = 92) -> str:
    """Pretty-print the Groq response with simple section headers."""
    sections = {
        "[Title]": "제목",
        "[Summary]": "요약",
        "[Table]": "테이블",
        "[Analysis]": "분석",
        "[Opinion]": "투자의견",
    }

    lines = []
    current_label = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line in sections:
            current_label = sections[line]
            lines.append(f"\n{current_label}\n" + "-" * len(current_label))
            continue
        wrapped = textwrap.fill(line, width=width, subsequent_indent="    ")
        lines.append(wrapped)

    return "\n".join(lines).strip()
