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


def _format_percent_change(current: Any, previous: Any) -> Optional[str]:
    try:
        cur = float(current)
        prev = float(previous)
    except (TypeError, ValueError):
        return None
    if prev == 0:
        return None
    change = ((cur - prev) / abs(prev)) * 100.0
    sign = "+" if change >= 0 else ""
    return f"{sign}{change:.1f}%"


def _format_absolute_change(current: Any, previous: Any) -> Optional[str]:
    try:
        cur = float(current)
        prev = float(previous)
    except (TypeError, ValueError):
        return None
    diff = cur - prev
    sign = "+" if diff >= 0 else ""
    return f"{sign}{_humanize_number(diff)}"


def format_tabular_prompt(
    payload: Optional[Dict[str, Any]],
    *,
    max_finance_years: int = 3,
    max_months: int = 3,
) -> str:
    if not payload:
        return ""

    sections: List[str] = []

    finance_data = payload.get("finance")
    if isinstance(finance_data, dict) and finance_data:
        years = sorted(finance_data.keys(), reverse=True)
        selected_years = years[:max_finance_years]
        finance_lines = ["최근 재무 추이 요약 (단위: 원, %)"]
        for idx, year in enumerate(selected_years):
            metrics = finance_data.get(year, {}) or {}
            if not isinstance(metrics, dict):
                continue
            revenue = metrics.get("매출액")
            operating = metrics.get("영업이익")
            margin = metrics.get("영업이익률")

            rev_change = None
            op_change = None
            if idx + 1 < len(selected_years):
                prev_year_key = selected_years[idx + 1]
                prev_metrics = finance_data.get(prev_year_key, {}) or {}
                rev_change = _format_percent_change(revenue, prev_metrics.get("매출액"))
                op_change = _format_percent_change(operating, prev_metrics.get("영업이익"))

            margin_txt = f"영업이익률 {margin:.2f}%" if isinstance(margin, (int, float)) else "영업이익률 정보 없음"
            line_parts = [
                f"매출액 {_humanize_number(revenue) if revenue is not None else '정보 없음'}",
                f"영업이익 {_humanize_number(operating) if operating is not None else '정보 없음'}",
                margin_txt,
            ]
            if rev_change:
                line_parts.append(f"매출 전년 대비 {rev_change}")
            if op_change:
                line_parts.append(f"영업이익 전년 대비 {op_change}")

            finance_lines.append(f"- {year}: " + ", ".join(line_parts))

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
            max_price = summary.get("max_price")
            min_price = summary.get("min_price")
            avg_price = summary.get("avg_price")
            stock_lines.append(
                "주요 통계: "
                + ", ".join(
                    part for part in [
                        f"최고가 {_humanize_number(max_price)}" if max_price is not None else None,
                        f"최저가 {_humanize_number(min_price)}" if min_price is not None else None,
                        f"평균가 {_humanize_number(avg_price)}" if avg_price is not None else None,
                    ]
                    if part
                )
            )
            if summary.get("std_dev") is not None:
                stock_lines.append(f"표준편차 {_humanize_number(summary.get('std_dev'))}")
            if summary.get("avg_volume") is not None:
                stock_lines.append(f"평균 거래량 {_humanize_number(summary.get('avg_volume'))}")

        monthly = stock_data.get("monthly_prices")
        if isinstance(monthly, dict) and monthly:
            monthly_items = sorted(monthly.items())
            if max_months > 0:
                monthly_items = monthly_items[-max_months:]
            if monthly_items:
                month_lines = [f"{month}: {_humanize_number(price)}" for month, price in monthly_items]
                stock_lines.append("최근 월별 종가: " + "; ".join(month_lines))
                if len(monthly_items) >= 2:
                    last_month, last_price = monthly_items[-1]
                    prev_month, prev_price = monthly_items[-2]
                    abs_change = _format_absolute_change(last_price, prev_price)
                    pct_change = _format_percent_change(last_price, prev_price)
                    trend_parts = []
                    if abs_change:
                        trend_parts.append(f"전월 대비 {abs_change}")
                    if pct_change:
                        trend_parts.append(pct_change)
                    if trend_parts:
                        stock_lines.append(f"최근 추세: {prev_month}→{last_month}, " + ", ".join(trend_parts))

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
