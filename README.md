# Legal AI — Contract Intelligence

A full-stack legal contract extraction app with a premium dark UI. Upload a PDF contract and get a structured profile with 9 extracted fields — parties, dates, obligations, governing law, and penalties.

---

## What It Does

Upload any commercial contract PDF and the app extracts:
- Party names
- Effective date and expiration date
- Renewal terms
- Governing law
- Payment terms
- Termination conditions
- Penalties and liquidated damages

Choose between three extraction modes:
- **Hybrid** — RoBERTa for dates/law, Gemini for parties/clauses (recommended)
- **RoBERTa only** — fully offline, no API key needed
- **Gemini only** — all fields through Gemini

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- Gemini API key from [aistudio.google.com](https://aistudio.google.com)
- 2 GB free RAM (for RoBERTa model)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/BharatGoyal28/legal-ai.git
cd legal-ai

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install frontend dependencies
cd frontend
npm install
cd ..

# 4. Add your Gemini API key
echo "GEMINI_API_KEY=your_key_here" > backend/.env
```

---

## How To Run

**Terminal 1 — Backend:**
```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Project Structure

```
legal-ai/
├── backend/
│   ├── main.py           API server (FastAPI, port 8001)
│   ├── pipeline.py       Field routing and extraction
│   ├── cuad_qa.py        RoBERTa-base-CUAD interface
│   ├── gemini_client.py  Gemini 2.5 Flash interface
│   ├── postprocess.py    Clean RoBERTa output
│   └── pdf_reader.py     PDF text extraction
└── frontend/
    └── src/
        ├── App.jsx
        └── components/
            ├── UploadZone.jsx
            ├── ProcessingState.jsx
            ├── ContractProfile.jsx
            └── ModelSwitcher.jsx
```

---

## Notes

- First run downloads RoBERTa model (~500 MB) — takes ~2 minutes
- Gemini free tier: 20 requests/day (5 fields × 4 contracts)
- RoBERTa only mode works completely offline
