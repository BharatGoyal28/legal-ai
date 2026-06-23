"""
FastAPI backend for the Legal Contract Profiler.

Endpoints:
  POST /extract        — Upload a PDF → returns full contract profile JSON
  POST /extract-text   — Send raw text → returns full contract profile JSON
  GET  /health         — Liveness check
"""

import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv(Path(__file__).parent / ".env")

import pdf_reader
import pipeline

app = FastAPI(title="Legal Contract Profiler", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    key = os.environ.get("GEMINI_API_KEY", "")
    gemini_ready = bool(key) and "your_gemini" not in key
    return {"status": "ok", "gemini_key_set": gemini_ready}


@app.post("/extract")
async def extract_from_pdf(
    file: UploadFile = File(...),
    mode: str = Form("hybrid"),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        text = pdf_reader.extract_text(tmp_path)
        if not text.strip():
            raise HTTPException(422, "Could not extract text from PDF.")
        profile = pipeline.extract(text, contract_name=file.filename, mode=mode)
        return JSONResponse(pipeline.profile_to_dict(profile))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/extract-text")
async def extract_from_text(
    text: str = Form(...),
    name: str = Form(""),
    mode: str = Form("hybrid"),
):
    if not text.strip():
        raise HTTPException(422, "Empty text supplied.")
    profile = pipeline.extract(text, contract_name=name, mode=mode)
    return JSONResponse(pipeline.profile_to_dict(profile))
