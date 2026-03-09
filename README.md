# AI Bill Analysis Dashboard

Web application for:

- Uploading bills/invoices/receipts (PDF, images, text, CSV)
- Visualizing changes in spend over time
- Asking questions in an AI chat interface about uploaded bills

## Features

1. **Document upload and extraction**
   - Supports electric bills, invoices, receipts, and similar documents
   - Extracts vendor, dates, tax, subtotal, total amount, and document type
2. **Dashboard analytics**
   - Total spend, average bill, largest bill, count of documents
   - Spend trend over time (monthly)
   - Spend by vendor and by document type
3. **AI chat**
   - Ask natural-language questions about uploaded bills
   - Uses OpenAI when API key is provided
   - Falls back to local deterministic analysis if no key is set

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal (usually `http://localhost:8501`).

## Optional AI setup

- Set an API key via sidebar field in the app, or:

```bash
export OPENAI_API_KEY="your_key_here"
export OPENAI_MODEL="gpt-4.1-mini"
```

## Notes

- PDF text extraction uses `pypdf`.
- Image OCR uses `pytesseract` when available.
- If OCR dependencies are missing in your environment, PDF/text uploads still work.

