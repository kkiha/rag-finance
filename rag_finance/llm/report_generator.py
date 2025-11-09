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
    tabular_text: str = "",
) -> List[Dict[str, str]]:
    """Build Groq chat messages using the unified prompt template."""
    system_lines = [
        "너는 한국어를 사용하는 증권사 애널리스트야.",
        "반드시 다음 순서와 태그를 그대로 사용해서 리포트를 작성해: [Title] → [Summary] → [Analysis] → [Opinion] → [Table].",
        "각 섹션별 지침은 아래와 같아.",
        "1) [Title]: 기업명과 핵심 이슈를 반영한 한 문장 헤드라인.",
        "2) [Summary]: 최소 3문장으로 최근 실적, 수요/공급 변화, 전망을 요약하고 구체적인 목표주가 또는 기대 주가 범위를 포함해.",
        "3) [Analysis]: 최소 4문장 이상 작성하고, 참고 문서와 정형 데이터에서 도출한 핵심 수치·증감률을 연결해 사업부 동향, 수요 요인, 리스크를 서술해.",
        "4) [Opinion]: 최소 2문장으로 투자 판단(매수/보유/중립 등)과 근거, 모니터링 포인트를 제시해.",
    "5) [Table]: 마지막에 Markdown 표 한 개만 제공하되, 표 위에는 간단한 제목 한 줄을 먼저 작성해. 표는 2025년 1분기부터 2026년 4분기까지의 분기별 매출액·영업이익·영업이익률 전망을 3열(또는 이상)로 정리하고, 정형 데이터와 문서 흐름을 근거로 합리적인 수치를 추론해 제시해. 원본 숫자를 그대로 반복하지 말고 파생 지표(전년 대비 성장률 등)를 포함해.",
        "정형 데이터 값은 참고용이므로 본문이나 표에서 그대로 나열하지 말고, 꼭 추세와 의미를 해석해서 전달해.",
        "출처 표시는 하지 마.",
        "숫자(실적, 성장률, 가이던스 등)는 가능하면 정량적으로 제시하고, 존재하지 않는 값은 만들지 마.",
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

    tabular_section = tabular_text.strip()
    if tabular_section:
        messages.append({
            "role": "user",
            "content": "[정형 데이터]\n" + tabular_section,
        })

    context_section = context_text.strip()
    user_prompt_parts: List[str] = []
    if context_section:
        user_prompt_parts.append(
            "[참고 문서]\n"
            "아래는 최신 수집 문서에서 발췌한 내용이야. 리포트 작성 시 이 정보를 우선적으로 활용해 줘.\n\n"
            + context_section
        )

    user_prompt_parts.append("[질문]\n" + query.strip())
    user_prompt = "\n\n".join(user_prompt_parts)
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
    tabular_text: str = "",
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
        tabular_text=tabular_text,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    report_text = response.choices[0].message.content.strip()

    combined_context_parts: List[str] = []
    if tabular_text and tabular_text.strip():
        combined_context_parts.append("[정형 데이터]\n" + tabular_text.strip())
    if context_text and context_text.strip():
        combined_context_parts.append(context_text.strip())
    combined_context = "\n\n".join(combined_context_parts)

    return report_text, messages, combined_context


def format_report_sections(text: str, width: int = 92) -> str:
    """Pretty-print the Groq response with simple section headers."""
    section_map = {
        "Title": "제목",
        "Summary": "요약",
        "Analysis": "분석",
        "Opinion": "투자의견",
        "Table": "테이블",
    }

    parsed = parse_report_sections(text)
    lines: List[str] = []

    for key in ("Title", "Summary", "Analysis", "Opinion", "Table"):
        content = parsed.get(key, "").strip()
        if not content:
            continue
        label = section_map[key]
        if key != "Table":
            lines.append(f"\n{label}\n" + "-" * len(label))
        if key == "Table":
            table_lines, other_lines = _split_markdown_table(content)
            heading_line = None
            if other_lines:
                heading_line = other_lines[0]
                other_lines = other_lines[1:]
            lines.append(f"\n{(heading_line or label)}\n" + "-" * len((heading_line or label)))
            if table_lines:
                table_rows = _markdown_lines_to_rows(table_lines)
                if table_rows:
                    lines.extend(_render_ascii_table(table_rows))
            for raw in other_lines:
                wrapped = textwrap.fill(raw, width=width, subsequent_indent="    ")
                lines.append(wrapped)
            continue

        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            wrapped = textwrap.fill(stripped, width=width, subsequent_indent="    ")
            lines.append(wrapped)

    return "\n".join(lines).strip()


def parse_report_sections(text: str) -> Dict[str, str]:
    """Parse the structured report text into section-to-content mapping."""
    markers = {"[Title]": "Title", "[Summary]": "Summary", "[Table]": "Table", "[Analysis]": "Analysis", "[Opinion]": "Opinion"}
    current_key: Optional[str] = None
    collected: Dict[str, List[str]] = {value: [] for value in markers.values()}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in markers:
            current_key = markers[line]
            continue
        if current_key:
            collected[current_key].append(line)

    return {key: "\n".join(value).strip() for key, value in collected.items() if value}


def _split_markdown_table(section_text: str) -> Tuple[List[str], List[str]]:
    table_lines: List[str] = []
    other_lines: List[str] = []
    for raw in section_text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("|") and stripped.count("|") >= 2:
            table_lines.append(stripped)
        else:
            other_lines.append(stripped)
    return table_lines, other_lines


def _markdown_lines_to_rows(lines: Sequence[str]) -> List[List[str]]:
    rows: List[List[str]] = []
    for idx, line in enumerate(lines):
        stripped = line.strip().strip("|")
        cells = [cell.strip() for cell in stripped.split("|")]
        if idx == 1 and all(set(cell.replace(":", "")) <= {"-"} for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        return []
    max_len = max(len(r) for r in rows)
    normalized: List[List[str]] = []
    for row in rows:
        if len(row) < max_len:
            row = row + [""] * (max_len - len(row))
        normalized.append(row)
    return normalized


def _render_ascii_table(rows: Sequence[Sequence[str]]) -> List[str]:
    if not rows:
        return []
    widths = [max(len(str(row[col])) for row in rows) for col in range(len(rows[0]))]

    def _fmt(row: Sequence[str]) -> str:
        return " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row))

    rendered = [_fmt(rows[0])]
    rendered.append("-+-".join("-" * w for w in widths))
    for row in rows[1:]:
        rendered.append(_fmt(row))
    return rendered
