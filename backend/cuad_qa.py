"""
Extractive QA wrapper for CUAD-fine-tuned RoBERTa-base.

Uses AutoModelForQuestionAnswering directly (bypasses the pipeline API, which
renamed the 'question-answering' task in transformers v5).

Long contracts are handled with a sliding window: the text is chunked into
MAX_LEN-token segments with STRIDE overlap, and we pick the span with the
highest combined start+end logit across all chunks.
"""

import time
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Official Atticus checkpoint (theatticusproject/cuad-roberta-base) requires
# HuggingFace authentication. We use the equivalent public fine-tune instead.
MODEL_ID = "akdeniz27/roberta-base-cuad"

MAX_LEN = 512
STRIDE = 128

CUAD_QUESTIONS = {
    "effective_date":        "What is the exact effective date or signing date of this contract?",
    "expiration_date":       "What is the exact expiration date or end date of this contract?",
    "governing_law":         "Which state or country law governs this contract?",
    "renewal":               "What are the renewal terms of this contract?",
    "termination_for_cause": "What are the conditions under which a party may terminate this contract for cause?",
    "penalties":             "What are the penalties or liquidated damages for breach of this contract?",
    "party_1":               "What is the full legal name of the first party in this contract?",
    "party_2":               "What is the full legal name of the second party in this contract?",
    "payment_terms":         "What are the payment terms and amounts in this contract?",
}

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _tokenizer is None:
        print(f"[CUAD] Loading {MODEL_ID} (first run ~500 MB download) …")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForQuestionAnswering.from_pretrained(MODEL_ID)
        _model.eval()
        print("[CUAD] Model ready.")
    return _tokenizer, _model


@dataclass
class QAResult:
    answer: str
    score: float
    latency_ms: int
    field: str


def _run_qa(question: str, context: str) -> tuple[str, float]:
    """Sliding-window extractive QA. Returns (answer_text, score)."""
    tok, mdl = _load()

    # Encode with overflow (sliding window)
    enc = tok(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=MAX_LEN,
        stride=STRIDE,
        return_overflowing_tokens=True,
        padding="max_length",
    )

    best_text = ""
    best_score = -1e9

    for i in range(enc["input_ids"].shape[0]):
        chunk = {k: v[i].unsqueeze(0) for k, v in enc.items() if k != "overflow_to_sample_mapping"}
        with torch.no_grad():
            out = mdl(**chunk)

        start_logits = out.start_logits[0]
        end_logits = out.end_logits[0]

        # Mask question tokens + special tokens (only search in context part)
        seq_ids = enc.sequence_ids(i)
        ctx_start = next((j for j, s in enumerate(seq_ids) if s == 1), 0)
        ctx_end   = next((len(seq_ids) - 1 - j for j, s in enumerate(reversed(seq_ids)) if s == 1), len(seq_ids) - 1)

        mask = torch.full_like(start_logits, -1e9)
        mask[ctx_start:ctx_end + 1] = 0
        start_logits = start_logits + mask
        end_logits   = end_logits   + mask

        # Best valid (start <= end) span within a reasonable window
        start_idx = int(start_logits.argmax())
        # limit answer to 30 tokens — longer spans are almost always wrong
        window_end = min(start_idx + 30, ctx_end)
        end_candidates = end_logits[start_idx:window_end + 1]
        if end_candidates.numel() == 0:
            continue
        end_idx = start_idx + int(end_candidates.argmax())

        score = float(start_logits[start_idx]) + float(end_logits[end_idx])
        if score > best_score:
            best_score = score
            ids = enc["input_ids"][i][start_idx:end_idx + 1]
            best_text = tok.decode(ids, skip_special_tokens=True).strip()

    # Normalise score to [0,1] with a soft sigmoid
    import math
    norm_score = 1 / (1 + math.exp(-best_score / 10)) if best_score > -1e8 else 0.0

    return best_text, norm_score


def extract_field(field: str, contract_text: str) -> Optional[QAResult]:
    if field not in CUAD_QUESTIONS:
        return None

    question = CUAD_QUESTIONS[field]
    t0 = time.perf_counter()
    answer, score = _run_qa(question, contract_text[:50000])
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    if not answer:
        return QAResult("Not found in contract", 0.0, elapsed_ms, field)
    return QAResult(answer, round(score, 3), elapsed_ms, field)


def extract_fields(fields: list[str], contract_text: str) -> dict[str, QAResult]:
    return {f: extract_field(f, contract_text) for f in fields if f in CUAD_QUESTIONS}
