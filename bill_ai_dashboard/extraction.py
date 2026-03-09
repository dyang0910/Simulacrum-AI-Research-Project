"""Utilities for extracting structured bill data from uploaded files."""

from __future__ import annotations

import io
import re
import uuid
from datetime import date
from typing import Any, Dict, Optional

from dateutil import parser as date_parser
from pypdf import PdfReader

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency fallback
    Image = None

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency fallback
    pytesseract = None


AMOUNT_PATTERN = re.compile(
    r"(?<![\d/\-.])(?:USD|EUR|GBP|INR|\$|€|£|₹)?\s*"
    r"(\d{1,3}(?:,\d{3})+(?:\.\d{2})?|\d+\.\d{2}|\d{2,6})(?![\d/\-.])"
)
DATE_PATTERN = re.compile(
    r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})",
    re.IGNORECASE,
)
NON_VENDOR_WORDS = {
    "invoice",
    "receipt",
    "bill",
    "statement",
    "tax",
    "total",
    "subtotal",
    "amount",
    "due",
}


def _safe_decode(data: bytes) -> str:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def extract_text_from_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    chunks = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks).strip()


def extract_text_from_image(data: bytes) -> str:
    if Image is None or pytesseract is None:
        return ""
    image = Image.open(io.BytesIO(data))
    return pytesseract.image_to_string(image)


def extract_text_from_upload(uploaded_file: Any) -> str:
    filename = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(data)
    if filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return extract_text_from_image(data)
    return _safe_decode(data)


def _to_float(value: str) -> Optional[float]:
    cleaned = value.replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_amount_after_keyword(text: str, keyword: str) -> Optional[float]:
    pattern = re.compile(
        rf"{keyword}\s*[:\-]?\s*(?:USD|EUR|GBP|INR|\$|€|£|₹)?\s*"
        r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})|\d+(?:\.\d{2})?)",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    return _to_float(match.group(1)) if match else None


def _extract_currency(text: str) -> str:
    upper = text.upper()
    if "USD" in upper or "$" in text:
        return "USD"
    if "EUR" in upper or "€" in text:
        return "EUR"
    if "GBP" in upper or "£" in text:
        return "GBP"
    if "INR" in upper or "₹" in text:
        return "INR"
    return "UNKNOWN"


def _extract_vendor(text: str, fallback_name: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:8]:
        lower = line.lower()
        if any(word in lower for word in NON_VENDOR_WORDS):
            continue
        if len(line) < 3:
            continue
        if re.search(r"[A-Za-z]", line):
            return line[:80]
    return fallback_name.rsplit(".", maxsplit=1)[0]


def _parse_date(raw_date: str) -> Optional[date]:
    try:
        return date_parser.parse(raw_date, fuzzy=True).date()
    except Exception:
        return None


def _extract_dates(text: str) -> Dict[str, Optional[date]]:
    issue_date = None
    due_date = None

    issue_match = re.search(r"(invoice|bill|issue)\s*date\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    if issue_match:
        issue_date = _parse_date(issue_match.group(2))

    due_match = re.search(r"due\s*date\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    if due_match:
        due_date = _parse_date(due_match.group(1))

    if not issue_date or not due_date:
        parsed_dates = [_parse_date(m) for m in DATE_PATTERN.findall(text)]
        parsed_dates = [d for d in parsed_dates if d is not None]
        parsed_dates = sorted(set(parsed_dates))
        if parsed_dates:
            issue_date = issue_date or parsed_dates[0]
            due_date = due_date or parsed_dates[-1]

    return {"issue_date": issue_date, "due_date": due_date}


def _extract_total_amount(text: str) -> float:
    for keyword in ("grand total", "amount due", "total due", "total amount", "total"):
        amount = _extract_amount_after_keyword(text, keyword)
        if amount is not None:
            return amount

    values = [_to_float(match.group(1)) for match in AMOUNT_PATTERN.finditer(text)]
    values = [val for val in values if val is not None]
    values = [val for val in values if not (val.is_integer() and 1900 <= val <= 2100)]
    if values:
        return max(values)
    return 0.0


def _infer_document_type(filename: str, text: str) -> str:
    text_lower = text.lower()
    name_lower = filename.lower()
    if "electric" in text_lower or "utility" in text_lower:
        return "electric_bill"
    if "invoice" in text_lower or "invoice" in name_lower:
        return "invoice"
    if "receipt" in text_lower or "receipt" in name_lower:
        return "receipt"
    return "other"


def parse_bill_document(uploaded_file: Any) -> Dict[str, Any]:
    text = extract_text_from_upload(uploaded_file)
    filename = uploaded_file.name

    dates = _extract_dates(text)
    subtotal = _extract_amount_after_keyword(text, "subtotal")
    tax = _extract_amount_after_keyword(text, "tax")
    total = _extract_total_amount(text)

    issue_date = dates["issue_date"] or date.today()

    return {
        "id": str(uuid.uuid4())[:8],
        "filename": filename,
        "document_type": _infer_document_type(filename, text),
        "vendor": _extract_vendor(text, filename),
        "issue_date": issue_date.isoformat(),
        "due_date": dates["due_date"].isoformat() if dates["due_date"] else None,
        "currency": _extract_currency(text),
        "subtotal_amount": subtotal,
        "tax_amount": tax,
        "total_amount": total,
        "text_excerpt": text[:7000],
    }

