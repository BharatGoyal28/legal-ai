"""
Gemini generative extraction with:
  - exponential backoff retry on rate-limit (429) errors
  - per-contract result caching (disk-based, keyed by text hash)
  - a simple in-process lock so only one Gemini call runs at a time

Uses the new google-genai SDK (google.generativeai is deprecated).

Gemini is used for fields that need interpretation across multiple paragraphs
or where the answer is not a single literal span (parties, renewal terms,
termination clauses, payment terms, penalties).
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types as gtypes

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

_lock = threading.Lock()
_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key or "your_gemini" in api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Add it to backend/.env before starting the server."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _cache_key(text: str, field: str) -> str:
    digest = hashlib.sha256((text + field).encode()).hexdigest()[:16]
    return str(CACHE_DIR / f"{field}_{digest}.json")


def _read_cache(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


FIELD_PROMPTS = {
    "party_1": (
        "From the following contract extract the full legal name of the FIRST party "
        "(often called Party A, the Client, Licensor, Seller, or similar). "
        "Return ONLY the exact name — no explanation.\n\nCONTRACT:\n{text}"
    ),
    "party_2": (
        "From the following contract extract the full legal name of the SECOND party "
        "(often called Party B, the Contractor, Licensee, Buyer, or similar). "
        "Return ONLY the exact name — no explanation.\n\nCONTRACT:\n{text}"
    ),
    "renewal": (
        "From the following contract extract the renewal/extension terms. Include: "
        "whether renewal is automatic, the renewal period length, and any notice required. "
        "Return one concise sentence. If none: 'Not found in contract'.\n\nCONTRACT:\n{text}"
    ),
    "payment_terms": (
        "From the following contract extract the payment terms. Include amounts, "
        "due dates, and conditions. One concise sentence. "
        "If none: 'Not found in contract'.\n\nCONTRACT:\n{text}"
    ),
    "termination_for_cause": (
        "From the following contract extract the termination-for-cause clause: "
        "what constitutes cause, notice period, cure period. One concise sentence. "
        "If none: 'Not found in contract'.\n\nCONTRACT:\n{text}"
    ),
    "penalties": (
        "From the following contract extract penalty, liquidated damages, or "
        "late-payment interest clauses. One concise sentence. "
        "If none: 'Not found in contract'.\n\nCONTRACT:\n{text}"
    ),
}

# Free-tier Gemini-flash: ~15 req/min — enforce a minimum gap
_last_call_time: float = 0.0
MIN_GAP_S = 6.0
MAX_RETRIES = 8
MODEL = "gemini-2.5-flash"


def extract_field(field: str, contract_text: str) -> dict:
    """
    Generative extraction for one field.
    Returns {"answer": str, "source": "gemini", "latency_ms": int, "cached": bool}
    """
    if field not in FIELD_PROMPTS:
        return {"answer": "Not found in contract", "source": "gemini", "latency_ms": 0, "cached": False}

    cache_path = _cache_key(contract_text, field)
    cached = _read_cache(cache_path)
    if cached:
        cached["cached"] = True
        return cached

    prompt = FIELD_PROMPTS[field].format(text=contract_text[:30000])

    with _lock:
        global _last_call_time
        wait = MIN_GAP_S - (time.time() - _last_call_time)
        if wait > 0:
            time.sleep(wait)

        client = _get_client()
        delay = 2.0
        answer = "Not found in contract"
        t0 = time.perf_counter()

        for attempt in range(MAX_RETRIES):
            try:
                resp = client.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config=gtypes.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=256,
                    ),
                )
                _last_call_time = time.time()
                answer = resp.text.strip() if resp.text else "Not found in contract"
                break
            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                    print(f"[Gemini] Rate limit (attempt {attempt+1}), sleeping {delay:.1f}s")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    _last_call_time = time.time()
                else:
                    answer = f"Gemini error: {err[:100]}"
                    break
        else:
            answer = "Not found in contract (rate limit exhausted)"

        latency_ms = int((time.perf_counter() - t0) * 1000)

    result = {"answer": answer, "source": "gemini", "latency_ms": latency_ms, "cached": False}
    # Only cache successful extractions — never cache rate-limit errors
    if "rate limit" not in answer.lower() and "error" not in answer.lower():
        _write_cache(cache_path, result)
    return result


def extract_fields(fields: list[str], contract_text: str) -> dict:
    return {f: extract_field(f, contract_text) for f in fields}
