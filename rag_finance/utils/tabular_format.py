from __future__ import annotations

from typing import Any, Dict, List, Optional


def _humanize_number(value: Any) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        abs_val = abs(value)
        if abs_val >= 1000:
            return f"{value:,.0f}"
        if abs_val >= 1:
            return f"{value:,.2f}"
        return f"{value:,.4f}"
    return str(value)


def format_tabular_prompt(
    payload: Optional[Dict[str, Any]],
    *,
    max_finance_years: int = 5,
    max_months: int = 6,
) -> str:
    if not payload:
        return ""

    sections: List[str] = []

    company = str(payload.get("company") or "").strip()
    if company:
        sections.append(f"기업명: {company}")

    finance_data = payload.get("finance")
    if isinstance(finance_data, dict) and finance_data:
        years = sorted(finance_data.keys(), reverse=True)
        finance_lines = ["최근 재무지표 (단위: 원)"]
        for year in years[:max_finance_years]:
            metrics = finance_data.get(year, {})
            if not isinstance(metrics, dict):
                continue
            formatted_metrics = ", ".join(
                f"{k}={_humanize_number(v)}"
                for k, v in metrics.items()
            )
            finance_lines.append(f"- {year}: {formatted_metrics}")
        if len(finance_lines) > 1:
            sections.append("\n".join(finance_lines))

    stock_data = payload.get("stock")
    if isinstance(stock_data, dict) and stock_data:
        stock_lines: List[str] = []
        period = stock_data.get("period")
        if period:
            stock_lines.append(f"주가 기간: {period}")

        summary = stock_data.get("summary")
        if isinstance(summary, dict) and summary:
            stock_lines.append("주요 주가 통계")
            for key, value in summary.items():
                stock_lines.append(f"- {key}: {_humanize_number(value)}")

        monthly = stock_data.get("monthly_prices")
        if isinstance(monthly, dict) and monthly:
            monthly_items = sorted(monthly.items())
            if max_months > 0:
                monthly_items = monthly_items[-max_months:]
            stock_lines.append("최근 월별 종가")
            for month, price in monthly_items:
                stock_lines.append(f"- {month}: {_humanize_number(price)}")

        if stock_lines:
            sections.append("\n".join(stock_lines))

    return "\n\n".join(s for s in sections if s).strip()


def build_finance_rows(finance_data: Optional[Dict[str, Any]], *, max_years: int = 5) -> List[List[str]]:
    if not isinstance(finance_data, dict) or not finance_data:
        return []

    years = sorted(finance_data.keys(), reverse=True)
    metrics_order: List[str] = []
    for payload in finance_data.values():
        if not isinstance(payload, dict):
            continue
        for key in payload.keys():
            if key not in metrics_order:
                metrics_order.append(key)

    if not metrics_order:
        return []

    rows: List[List[str]] = [["연도", *metrics_order]]
    for year in years[:max_years]:
        metrics = finance_data.get(year, {}) or {}
        row = [str(year)]
        for key in metrics_order:
            row.append(_humanize_number(metrics.get(key, "")))
        rows.append(row)
    return rows


def build_stock_summary_rows(stock_data: Optional[Dict[str, Any]]) -> List[List[str]]:
    if not isinstance(stock_data, dict):
        return []
    summary = stock_data.get("summary")
    if not isinstance(summary, dict) or not summary:
        return []

    rows = [["지표", "값"]]
    for key, value in summary.items():
        rows.append([str(key), _humanize_number(value)])
    return rows


def build_stock_monthly_rows(stock_data: Optional[Dict[str, Any]], *, max_months: int = 6) -> List[List[str]]:
    if not isinstance(stock_data, dict):
        return []
    monthly = stock_data.get("monthly_prices")
    if not isinstance(monthly, dict) or not monthly:
        return []

    items = sorted(monthly.items())
    if max_months > 0:
        items = items[-max_months:]

    rows = [["월", "종가"]]
    for month, price in items:
        rows.append([str(month), _humanize_number(price)])
    return rows
