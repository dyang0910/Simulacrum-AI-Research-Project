"""Chat utilities for answering questions about uploaded bills."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency fallback
    OpenAI = None


def _format_dataset_context(df: pd.DataFrame, records: List[Dict[str, Any]]) -> str:
    view = df[
        [
            "filename",
            "document_type",
            "vendor",
            "issue_date",
            "due_date",
            "currency",
            "total_amount",
            "tax_amount",
        ]
    ].copy()
    view["issue_date"] = view["issue_date"].dt.strftime("%Y-%m-%d")
    view["due_date"] = view["due_date"].dt.strftime("%Y-%m-%d")
    table_csv = view.to_csv(index=False)

    snippets = []
    for record in records[:15]:
        snippets.append(
            f"File: {record['filename']}\nVendor: {record['vendor']}\n"
            f"Type: {record['document_type']}\nAmount: {record['total_amount']}\n"
            f"Excerpt:\n{record.get('text_excerpt', '')[:500]}"
        )
    return f"Structured table:\n{table_csv}\n\nDocument snippets:\n\n" + "\n\n---\n\n".join(snippets)


def _local_fallback_answer(question: str, df: pd.DataFrame) -> str:
    q = question.lower().strip()
    if df.empty:
        return "No uploaded bills are available yet. Upload documents first."

    total_spend = df["total_amount"].sum()
    average = df["total_amount"].mean()
    largest_idx = df["total_amount"].idxmax()
    largest_row = df.loc[largest_idx]

    if "total" in q and ("spend" in q or "paid" in q or "amount" in q):
        return f"Total spend across uploaded bills is {total_spend:,.2f}."

    if "average" in q or "mean" in q:
        return f"Average bill value is {average:,.2f}."

    if "highest" in q or "largest" in q or "biggest" in q:
        return (
            f"Largest bill is {largest_row['filename']} from {largest_row['vendor']} "
            f"with amount {largest_row['total_amount']:,.2f}."
        )

    if "vendor" in q and ("most" in q or "top" in q):
        top = (
            df.groupby("vendor", as_index=False)["total_amount"]
            .sum()
            .sort_values("total_amount", ascending=False)
            .head(3)
        )
        lines = [f"- {row.vendor}: {row.total_amount:,.2f}" for row in top.itertuples()]
        return "Top vendors by spend:\n" + "\n".join(lines)

    if "month" in q or "trend" in q or "over time" in q:
        monthly = (
            df.assign(month=df["issue_date"].dt.to_period("M").dt.to_timestamp())
            .groupby("month", as_index=False)["total_amount"]
            .sum()
            .sort_values("month")
        )
        lines = [f"- {row.month.strftime('%Y-%m')}: {row.total_amount:,.2f}" for row in monthly.itertuples()]
        return "Monthly spend trend:\n" + "\n".join(lines)

    return (
        "I can answer questions about totals, averages, top vendors, and monthly trends. "
        "Set OPENAI_API_KEY (or provide it in the sidebar) for richer natural-language answers."
    )


def _openai_answer(
    question: str,
    df: pd.DataFrame,
    records: List[Dict[str, Any]],
    api_key: str,
    model: str,
) -> Optional[str]:
    if OpenAI is None:
        return None
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)
    context = _format_dataset_context(df, records)
    system_prompt = (
        "You are a billing analyst assistant. Use ONLY the provided bill data context. "
        "If data is missing, clearly say so. Keep answers concise and numeric where possible."
    )
    user_prompt = f"Question: {question}\n\nBill Data Context:\n{context}"

    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ],
            temperature=0.1,
        )
        return (response.output_text or "").strip()
    except Exception:
        return None


def answer_bill_question(
    question: str,
    df: pd.DataFrame,
    records: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    if model is None:
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")

    llm_answer = _openai_answer(
        question=question,
        df=df,
        records=records,
        api_key=resolved_key,
        model=model,
    )
    if llm_answer:
        return llm_answer

    return _local_fallback_answer(question, df)

