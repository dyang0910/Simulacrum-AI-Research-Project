"""Streamlit app for bill upload, analytics dashboard, and AI chat."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from bill_ai_dashboard.analytics import (
    calculate_summary_metrics,
    monthly_spend,
    records_to_dataframe,
    spend_by_doc_type,
    spend_by_vendor,
)
from bill_ai_dashboard.chat import answer_bill_question
from bill_ai_dashboard.extraction import parse_bill_document


st.set_page_config(page_title="AI Bill Analysis Dashboard", page_icon=":bar_chart:", layout="wide")
st.title("AI Bill Analysis Dashboard")
st.caption("Upload bills/invoices/receipts, explore trends over time, and ask questions in chat.")

if "records" not in st.session_state:
    st.session_state.records = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Upload bills and click *Analyze uploaded bills*. "
                "Then ask questions like: *What changed month to month?*"
            ),
        }
    ]

with st.sidebar:
    st.header("Upload and AI Settings")
    uploaded_files = st.file_uploader(
        "Upload bills, invoices, receipts",
        type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "csv"],
        accept_multiple_files=True,
    )

    if st.button("Analyze uploaded bills", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            parsed_records: List[Dict[str, Any]] = []
            with st.spinner("Extracting bill details from files..."):
                for file_obj in uploaded_files:
                    try:
                        parsed_records.append(parse_bill_document(file_obj))
                    except Exception as exc:
                        parsed_records.append(
                            {
                                "id": f"error-{file_obj.name}",
                                "filename": file_obj.name,
                                "document_type": "error",
                                "vendor": "Parse Error",
                                "issue_date": pd.Timestamp.now().date().isoformat(),
                                "due_date": None,
                                "currency": "UNKNOWN",
                                "subtotal_amount": None,
                                "tax_amount": None,
                                "total_amount": 0.0,
                                "text_excerpt": f"Parsing failed: {exc}",
                            }
                        )
            st.session_state.records = parsed_records
            st.success(f"Analyzed {len(parsed_records)} files.")

records: List[Dict[str, Any]] = st.session_state.records
df = records_to_dataframe(records)

tab_dashboard, tab_documents, tab_chat = st.tabs(["Dashboard", "Documents", "AI Chat"])

with tab_dashboard:
    if df.empty:
        st.info("No bills analyzed yet. Upload files from the sidebar and click Analyze.")
    else:
        metrics = calculate_summary_metrics(df)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Documents", int(metrics["documents"]))
        m2.metric("Total spend", f"{metrics['total_spend']:,.2f}")
        m3.metric("Average bill", f"{metrics['average_bill']:,.2f}")
        m4.metric("Largest bill", f"{metrics['largest_bill']:,.2f}")

        trend = monthly_spend(df)
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Spend over time")
            if trend.empty:
                st.caption("No valid issue dates found for trend chart.")
            else:
                st.line_chart(trend.set_index("month")["total_amount"])

        with col_right:
            st.subheader("Spend by vendor")
            vendor = spend_by_vendor(df)
            if vendor.empty:
                st.caption("No vendor data.")
            else:
                st.bar_chart(vendor.set_index("vendor")["total_amount"])

        st.subheader("Spend by document type")
        by_type = spend_by_doc_type(df)
        if by_type.empty:
            st.caption("No document type data.")
        else:
            st.bar_chart(by_type.set_index("document_type")["total_amount"])

with tab_documents:
    if df.empty:
        st.info("No document data yet.")
    else:
        display_df = df.copy()
        display_df["issue_date"] = display_df["issue_date"].dt.strftime("%Y-%m-%d")
        display_df["due_date"] = display_df["due_date"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            display_df[
                [
                    "filename",
                    "document_type",
                    "vendor",
                    "issue_date",
                    "due_date",
                    "currency",
                    "subtotal_amount",
                    "tax_amount",
                    "total_amount",
                ]
            ],
            use_container_width=True,
        )

with tab_chat:
    st.caption("Ask questions about the uploaded bills.")
    if not records:
        st.info("Upload and analyze documents to enable chat analysis.")
    else:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_question = st.chat_input("Ask about totals, trends, vendors, anomalies, or date ranges...")
        if user_question:
            st.session_state.chat_messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing bills..."):
                    answer = answer_bill_question(
                        question=user_question,
                        df=df,
                        records=records,
                        model=model_name,
                    )
                st.markdown(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})

