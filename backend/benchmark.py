"""
Benchmark: Atticus RoBERTa-base vs Gemini on CUAD contracts.

Downloads 4 CUAD contracts + ground-truth from HuggingFace datasets,
runs both models, prints accuracy table and timing comparison.

Run:  python benchmark.py
"""

import json
import os
import time
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import cuad_qa
import gemini_client
import pipeline

# CUAD ground-truth field → dataset column mapping
FIELD_TO_CUAD = {
    "effective_date":       "Effective Date",
    "expiration_date":      "Expiration Date",
    "governing_law":        "Governing Law",
    "renewal":              "Renewal Term",
}

# Fields where we compare Gemini (no direct CUAD GT mapping — we judge manually)
GEMINI_ONLY_FIELDS = ["party_1", "party_2", "termination_for_cause", "payment_terms", "penalties"]

# How many contracts to benchmark (keep low for free-tier Gemini)
N_CONTRACTS = 4


def token_overlap(pred: str, gold: str) -> float:
    """Simple token-level F1 — standard for SQuAD/CUAD evaluation."""
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main():
    print("Loading CUAD dataset (first run may download ~150 MB)…")
    ds = load_dataset("theatticusproject/cuad", split="test", trust_remote_code=True)
    print(f"  Dataset loaded: {len(ds)} examples")

    # Pick N_CONTRACTS diverse examples (one per unique contract title)
    seen_titles = set()
    samples = []
    for ex in ds:
        title = ex.get("title", "")
        if title not in seen_titles:
            seen_titles.add(title)
            samples.append(ex)
        if len(samples) == N_CONTRACTS:
            break

    print(f"\nBenchmarking on {len(samples)} contracts…\n")
    print("=" * 80)

    roberta_scores = {f: [] for f in FIELD_TO_CUAD}
    gemini_scores = {f: [] for f in FIELD_TO_CUAD}
    roberta_times = []
    gemini_times = []

    rows = []   # for final accuracy table

    for i, sample in enumerate(samples):
        contract_text = sample["context"]
        title = sample.get("title", f"Contract_{i+1}")
        print(f"\n[{i+1}/{N_CONTRACTS}] {title[:60]}")
        print(f"  Text length: {len(contract_text):,} chars")

        row = {"contract": title[:40]}

        for field, cuad_col in FIELD_TO_CUAD.items():
            # Ground truth — CUAD stores answers as list of spans
            gt_answers = sample.get("answers", {}).get("text", [])
            # Match by question
            q_idx = None
            for qi, q in enumerate(sample.get("question", [])):
                if cuad_col.lower() in q.lower():
                    q_idx = qi
                    break
            gold = gt_answers[q_idx] if (q_idx is not None and q_idx < len(gt_answers)) else ""

            # RoBERTa
            t0 = time.perf_counter()
            rb_res = cuad_qa.extract_field(field, contract_text)
            rb_ms = int((time.perf_counter() - t0) * 1000)
            rb_ans = rb_res.answer if rb_res else "Not found in contract"
            rb_f1 = token_overlap(rb_ans, gold) if gold else None
            roberta_times.append(rb_ms)

            # Gemini (skip on missing key)
            g_ans = "N/A"
            g_f1 = None
            g_ms = 0
            if os.environ.get("GEMINI_API_KEY") and "your_gemini" not in os.environ.get("GEMINI_API_KEY", ""):
                t0 = time.perf_counter()
                g_res = gemini_client.extract_field(field, contract_text)
                g_ms = int((time.perf_counter() - t0) * 1000)
                g_ans = g_res.get("answer", "Not found in contract")
                g_f1 = token_overlap(g_ans, gold) if gold else None
                gemini_times.append(g_ms)

            if rb_f1 is not None:
                roberta_scores[field].append(rb_f1)
            if g_f1 is not None:
                gemini_scores[field].append(g_f1)

            print(f"  {field:<25} GT: {gold[:50]!r}")
            print(f"    RoBERTa ({rb_ms:>5}ms, score={rb_res.score:.3f}): {rb_ans[:70]!r}")
            if g_ans != "N/A":
                print(f"    Gemini  ({g_ms:>5}ms): {g_ans[:70]!r}")

            row[f"{field}_rb"] = f"{rb_f1:.2f}" if rb_f1 is not None else "—"
            row[f"{field}_gem"] = f"{g_f1:.2f}" if g_f1 is not None else "—"

        rows.append(row)

    # Print accuracy table
    print("\n" + "=" * 80)
    print("ACCURACY TABLE (token-F1, 0–1)  [RB = RoBERTa-base | GM = Gemini-flash]")
    print("=" * 80)
    header = f"{'Contract':<42}"
    for f in FIELD_TO_CUAD:
        header += f" {f[:8]:>9}RB {f[:8]:>8}GM"
    print(header)
    print("-" * 80)
    for row in rows:
        line = f"{row['contract']:<42}"
        for f in FIELD_TO_CUAD:
            line += f" {row.get(f+'_rb','—'):>11} {row.get(f+'_gem','—'):>10}"
        print(line)

    print("\nMean F1 per field:")
    for f in FIELD_TO_CUAD:
        rb_mean = sum(roberta_scores[f]) / max(len(roberta_scores[f]), 1)
        g_mean = sum(gemini_scores[f]) / max(len(gemini_scores[f]), 1)
        print(f"  {f:<25} RoBERTa: {rb_mean:.3f}   Gemini: {g_mean:.3f}")

    if roberta_times:
        print(f"\nAvg RoBERTa latency: {sum(roberta_times)//len(roberta_times)} ms/field")
    if gemini_times:
        print(f"Avg Gemini latency:  {sum(gemini_times)//len(gemini_times)} ms/field")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
