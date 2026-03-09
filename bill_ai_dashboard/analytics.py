"""Analytics helpers for bill dashboard visualizations."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "id",
                "filename",
                "document_type",
                "vendor",
                "issue_date",
                "due_date",
                "currency",
                "subtotal_amount",
                "tax_amount",
                "total_amount",
                "text_excerpt",
            ]
        )

    df = pd.DataFrame(records)
    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
    df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce")
    df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce").fillna(0.0)
    df["tax_amount"] = pd.to_numeric(df["tax_amount"], errors="coerce")
    df["subtotal_amount"] = pd.to_numeric(df["subtotal_amount"], errors="coerce")
    return df.sort_values("issue_date").reset_index(drop=True)


def calculate_summary_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "documents": 0,
            "total_spend": 0.0,
            "average_bill": 0.0,
            "largest_bill": 0.0,
        }

    return {
        "documents": float(df.shape[0]),
        "total_spend": float(df["total_amount"].sum()),
        "average_bill": float(df["total_amount"].mean()),
        "largest_bill": float(df["total_amount"].max()),
    }


def monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "total_amount"])

    monthly = (
        df.dropna(subset=["issue_date"])
        .assign(month=lambda frame: frame["issue_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)["total_amount"]
        .sum()
        .sort_values("month")
    )
    return monthly


def spend_by_vendor(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["vendor", "total_amount"])
    vendor = (
        df.groupby("vendor", as_index=False)["total_amount"]
        .sum()
        .sort_values("total_amount", ascending=False)
    )
    return vendor.head(15)


def spend_by_doc_type(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["document_type", "total_amount"])
    return (
        df.groupby("document_type", as_index=False)["total_amount"]
        .sum()
        .sort_values("total_amount", ascending=False)
    )

