"""
Builds final report purely from already-collected results.
No model loading. No API calls. Instant.
Run: python build_final_report.py
"""

from evaluate_15 import CONTRACTS, FIELDS, SHORT, token_f1
import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "evaluation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

def badge(f1):
    if f1 >= 0.7: return "OK "
    if f1 >= 0.3: return "~~ "
    return "-- "

# ── RoBERTa — all 15 contracts (from evaluate_15 run) ────────────────
RB = [
    ("TechSolutions Service Agmt",   {"party_1":0.00,"party_2":0.00,"effective_date":1.00,"expiration_date":1.00,"governing_law":0.40,"renewal":0.93,"termination_for_cause":0.00,"payment_terms":0.00,"penalties":0.00}),
    ("NovaPharma License Agmt",      {"party_1":0.00,"party_2":0.00,"effective_date":0.00,"expiration_date":0.00,"governing_law":1.00,"renewal":0.91,"termination_for_cause":0.00,"payment_terms":0.00,"penalties":0.00}),
    ("Acme Corp Distribution Agmt",  {"party_1":0.00,"party_2":0.00,"effective_date":0.00,"expiration_date":0.00,"governing_law":0.40,"renewal":0.85,"termination_for_cause":0.00,"payment_terms":0.00,"penalties":0.50}),
    ("Sterling Properties Lease",    {"party_1":0.00,"party_2":0.00,"effective_date":0.35,"expiration_date":0.00,"governing_law":1.00,"renewal":0.67,"termination_for_cause":0.00,"payment_terms":0.00,"penalties":0.00}),
    ("Vertex Capital Tech Services", {"party_1":0.00,"party_2":0.00,"effective_date":1.00,"expiration_date":0.00,"governing_law":1.00,"renewal":0.90,"termination_for_cause":0.00,"payment_terms":0.00,"penalties":0.57}),
    ("AlphaGen Software License",    {"party_1":0.00,"party_2":0.00,"effective_date":1.00,"expiration_date":1.00,"governing_law":1.00,"renewal":1.00,"termination_for_cause":0.00,"payment_terms":0.00,"penalties":0.00}),
    ("BlueSky NDA Agreement",        {"party_1":0.00,"party_2":0.00,"effective_date":1.00,"expiration_date":1.00,"governing_law":1.00,"renewal":0.00,"termination_for_cause":0.00,"payment_terms":0.00,"penalties":0.00}),
    ("PrimeCare Consulting Agmt",    {"party_1":0.00,"party_2":0.00,"effective_date":0.35,"expiration_date":0.00,"governing_law":1.00,"renewal":1.00,"termination_for_cause":0.00,"payment_terms":0.00,"penalties":0.00}),
    ("EcoSupply Manufacturing Agmt", {"party_1":0.00,"party_2":0.00,"effective_date":0.00,"expiration_date":0.00,"governing_law":1.00,"renewal":1.00,"termination_for_cause":0.75,"payment_terms":0.00,"penalties":0.00}),
    ("GlobalMedia Partnership Agmt", {"party_1":0.00,"party_2":0.00,"effective_date":0.35,"expiration_date":1.00,"governing_law":1.00,"renewal":0.00,"termination_for_cause":0.00,"payment_terms":0.00,"penalties":0.50}),
    ("SunTech Employment Contract",  {"party_1":0.35,"party_2":0.00,"effective_date":0.35,"expiration_date":0.00,"governing_law":1.00,"renewal":1.00,"termination_for_cause":0.00,"payment_terms":0.00,"penalties":0.00}),
    ("FreshFoods Supply Agreement",  {"party_1":0.00,"party_2":0.00,"effective_date":0.35,"expiration_date":0.00,"governing_law":1.00,"renewal":1.00,"termination_for_cause":0.00,"payment_terms":0.00,"penalties":0.00}),
    ("CityArc Research Agreement",   {"party_1":0.00,"party_2":0.00,"effective_date":0.35,"expiration_date":0.00,"governing_law":1.00,"renewal":1.00,"termination_for_cause":0.00,"payment_terms":0.00,"penalties":0.00}),
    ("MaxBrand Marketing Agreement", {"party_1":0.00,"party_2":0.00,"effective_date":0.35,"expiration_date":1.00,"governing_law":1.00,"renewal":1.00,"termination_for_cause":0.35,"payment_terms":0.00,"penalties":0.00}),
    ("NorthStar Joint Venture Agmt", {"party_1":0.00,"party_2":0.00,"effective_date":0.00,"expiration_date":1.00,"governing_law":1.00,"renewal":0.00,"termination_for_cause":0.00,"payment_terms":0.00,"penalties":0.00}),
]

# ── Gemini — 8 contracts (from compare_3 + gemini_remaining runs) ─────
GEM = [
    ("TechSolutions Service Agmt",   {"party_1":1.00,"party_2":1.00,"effective_date":0.00,"expiration_date":0.00,"governing_law":0.00,"renewal":0.54,"termination_for_cause":0.19,"payment_terms":0.33,"penalties":0.62}),
    ("NovaPharma License Agmt",      {"party_1":0.29,"party_2":1.00,"effective_date":0.00,"expiration_date":0.00,"governing_law":0.00,"renewal":0.70,"termination_for_cause":0.60,"payment_terms":0.73,"penalties":0.56}),
    ("Acme Corp Distribution Agmt",  {"party_1":1.00,"party_2":1.00,"effective_date":0.00,"expiration_date":0.00,"governing_law":0.00,"renewal":0.80,"termination_for_cause":0.65,"payment_terms":0.70,"penalties":0.60}),
    ("Sterling Properties Lease",    {"party_1":1.00,"party_2":1.00,"effective_date":0.00,"expiration_date":0.00,"governing_law":0.00,"renewal":0.45,"termination_for_cause":0.70,"payment_terms":0.75,"penalties":0.35}),
    ("Vertex Capital Tech Services", {"party_1":1.00,"party_2":1.00,"effective_date":0.00,"expiration_date":0.00,"governing_law":0.00,"renewal":0.67,"termination_for_cause":0.12,"payment_terms":0.75,"penalties":0.00}),
    ("AlphaGen Software License",    {"party_1":1.00,"party_2":1.00,"effective_date":0.00,"expiration_date":0.00,"governing_law":0.00,"renewal":0.60,"termination_for_cause":0.75,"payment_terms":0.80,"penalties":0.70}),
    ("BlueSky NDA Agreement",        {"party_1":1.00,"party_2":1.00,"effective_date":0.00,"expiration_date":0.00,"governing_law":0.00,"renewal":0.00,"termination_for_cause":0.55,"payment_terms":0.00,"penalties":0.80}),
    ("PrimeCare Consulting Agmt",    {"party_1":1.00,"party_2":1.00,"effective_date":0.00,"expiration_date":0.00,"governing_law":0.00,"renewal":0.50,"termination_for_cause":0.75,"payment_terms":0.80,"penalties":0.70}),
]


def print_table(data, title, n_contracts):
    W = 108
    cols = [SHORT[f] for f in FIELDS]
    print(f"\n{'='*W}")
    print(f"  {title}  ({n_contracts} contracts)")
    print(f"  [OK] F1>=0.7  [~~] F1>=0.3  [--] wrong")
    print(f"{'='*W}")
    hdr = f"{'Contract':<36}" + "".join(f"{c:>7}" for c in cols) + "   Mean"
    print(hdr)
    print("-"*W)
    for name, fields in data:
        line = f"{name:<36}"
        for field in FIELDS:
            b = badge(fields[field])
            line += f"  [{b}]"
        mean = sum(fields.values()) / len(fields)
        line += f"  {mean:.2f}"
        print(line)
    print("-"*W)
    acc = {}
    acc_line = f"{'Accuracy':<36}"
    for field in FIELDS:
        a = sum(1 for _, f in data if f[field] >= 0.7) / len(data) * 100
        acc[field] = a
        acc_line += f"  {a:>3.0f}%"
    overall = sum(acc.values()) / len(acc)
    print(acc_line + f"  {overall:.0f}%")
    print(f"{'='*W}")
    return acc


def print_winner(rb_acc, gem_acc):
    print(f"\n{'='*65}")
    print("  WINNER TABLE — FIELD BY FIELD")
    print(f"{'='*65}")
    print(f"  {'Field':<25} {'RoBERTa':>10} {'Gemini':>10}  Winner")
    print("  " + "-"*55)
    rb_w = gem_w = ties = 0
    for field in FIELDS:
        rb  = rb_acc[field]
        gem = gem_acc[field]
        if rb > gem:   winner = "RoBERTa"; rb_w += 1
        elif gem > rb: winner = "Gemini "; gem_w += 1
        else:          winner = "Tie    "; ties += 1
        print(f"  {field:<25} {rb:>9.0f}%  {gem:>9.0f}%  {winner}")
    print("  " + "-"*55)
    rb_o  = sum(rb_acc.values()) / len(rb_acc)
    gem_o = sum(gem_acc.values()) / len(gem_acc)
    overall_winner = "RoBERTa" if rb_o > gem_o else "Gemini "
    print(f"  {'Overall':<25} {rb_o:>9.0f}%  {gem_o:>9.0f}%  {overall_winner}")
    print(f"\n  RoBERTa wins: {rb_w} fields")
    print(f"  Gemini  wins: {gem_w} fields")
    print(f"  Ties:         {ties} fields")
    print(f"{'='*65}")


def print_failures():
    failures = []
    for name, fields in GEM:
        for field in FIELDS:
            if fields[field] < 0.3:
                failures.append((field, name, fields[field]))
    failures.sort(key=lambda x: x[2])
    print(f"\n{'='*65}")
    print("  TOP 5 FAILURES (for your failure analysis section)")
    print(f"{'='*65}")
    for i, (field, contract, f1) in enumerate(failures[:5], 1):
        gt = next((c["gt"].get(field,"") for c in CONTRACTS if c["name"].startswith(contract[:15])), "")
        print(f"\n  {i}. Field: {field}")
        print(f"     Contract: {contract}")
        print(f"     Ground Truth: {gt[:65]!r}")
        print(f"     Gemini F1: {f1:.2f}  (wrong)")


def save(rb_acc, gem_acc):
    txt = []
    txt.append("FINAL EVALUATION REPORT — CUAD 15-CONTRACT BENCHMARK")
    txt.append("="*60)
    txt.append("")
    txt.append("ROBERTA ACCURACY (15 contracts)")
    for f in FIELDS:
        txt.append(f"  {f:<25} {rb_acc[f]:.0f}%")
    txt.append(f"  {'Overall':<25} {sum(rb_acc.values())/len(rb_acc):.0f}%")
    txt.append("")
    txt.append("GEMINI ACCURACY (8 contracts)")
    for f in FIELDS:
        txt.append(f"  {f:<25} {gem_acc[f]:.0f}%")
    txt.append(f"  {'Overall':<25} {sum(gem_acc.values())/len(gem_acc):.0f}%")
    txt.append("")
    txt.append("WINNER PER FIELD")
    for f in FIELDS:
        rb = rb_acc[f]; gem = gem_acc[f]
        w = "RoBERTa" if rb > gem else "Gemini" if gem > rb else "Tie"
        txt.append(f"  {f:<25} {w}")
    p = OUTPUT_DIR / "final_report.txt"
    p.write_text("\n".join(txt), encoding="utf-8")
    print(f"\n  Saved -> {p}")


# ── Same 8 contracts for fair comparison ─────────────────────────────
GEM_NAMES = [name for name, _ in GEM]
RB_8 = [(name, fields) for name, fields in RB if name in GEM_NAMES]

# ── Run ───────────────────────────────────────────────────────────────
print_table(RB,   "TABLE 1 — ROBERTA (all 15 contracts)", 15)
rb_acc  = print_table(RB_8, "TABLE 2 — ROBERTA (same 8 contracts as Gemini — for fair comparison)", 8)
gem_acc = print_table(GEM,  "TABLE 3 — GEMINI  (8 contracts)", 8)
print_winner(rb_acc, gem_acc)
print_failures()
save(rb_acc, gem_acc)
print("\nDone.")
