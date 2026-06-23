"""
Automated 15-contract evaluation — RoBERTa vs Gemini.
Uses 15 diverse CUAD-style contracts with known ground truth.
Prints two result tables and saves to evaluation_results/.

Run: python evaluate_15.py
"""

import json, os, sys, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import pipeline, cuad_qa

OUTPUT_DIR = Path(__file__).parent / "evaluation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

FIELDS = [
    "party_1","party_2","effective_date","expiration_date",
    "governing_law","renewal","termination_for_cause","payment_terms","penalties",
]
SHORT = {
    "party_1":"Party1","party_2":"Party2","effective_date":"EffDate",
    "expiration_date":"ExpDate","governing_law":"Law","renewal":"Renewal",
    "termination_for_cause":"Term","payment_terms":"Payment","penalties":"Penalty",
}

# ── 15 Diverse contracts ──────────────────────────────────────────────
CONTRACTS = [
{
"name":"TechSolutions Service Agmt",
"text":"""SERVICE AGREEMENT
This Service Agreement is entered into as of March 1, 2022, between Meridian Financial Group, Inc.,
a Delaware corporation ("Client"), and TechSolutions LLC, a Texas limited liability company ("Provider").
TERM. This Agreement commences April 1, 2022 and expires March 31, 2024.
RENEWAL. Automatically renews for one-year terms unless either party provides ninety (90) days written notice.
GOVERNING LAW. Governed by laws of the State of Texas.
TERMINATION FOR CAUSE. Client may terminate immediately if Provider commits material fraud or fails to cure
a material breach within 15 days of written notice.
PAYMENT. Client shall pay $25,000 per quarter, due within 45 days of invoice.
PENALTIES. Late payments accrue interest at 2% per month.""",
"gt":{"party_1":"Meridian Financial Group, Inc.","party_2":"TechSolutions LLC",
"effective_date":"March 1, 2022","expiration_date":"March 31, 2024",
"governing_law":"State of Texas",
"renewal":"automatically renews for one-year terms unless either party provides ninety (90) days written notice",
"termination_for_cause":"terminate immediately if Provider commits material fraud or fails to cure within 15 days",
"payment_terms":"$25,000 per quarter due within 45 days","penalties":"2% per month interest on late payments"}
},
{
"name":"NovaPharma License Agmt",
"text":"""LICENSE AGREEMENT
Dated January 15, 2021, between NovaPharma Inc., a California corporation ("Licensor"),
and BioResearch Partners LP, a New York limited partnership ("Licensee").
EFFECTIVE DATE. Effective February 1, 2021.
TERM. Continues until January 31, 2026.
RENEWAL. Automatically renews for successive two-year periods unless 120 days notice given.
GOVERNING LAW. Governed by laws of the State of California.
TERMINATION. Licensor may terminate if Licensee fails to pay within 30 days of written notice.
ROYALTIES. Licensee pays 8% of Net Sales quarterly.
PENALTIES. Failure to pay royalties results in liquidated damages of $100,000 per quarter.""",
"gt":{"party_1":"NovaPharma Inc.","party_2":"BioResearch Partners LP",
"effective_date":"February 1, 2021","expiration_date":"January 31, 2026",
"governing_law":"State of California",
"renewal":"automatically renews for successive two-year periods unless 120 days notice given",
"termination_for_cause":"terminate if Licensee fails to pay within 30 days of written notice",
"payment_terms":"8% of Net Sales quarterly","penalties":"liquidated damages of $100,000 per quarter"}
},
{
"name":"Acme Corp Distribution Agmt",
"text":"""DISTRIBUTION AGREEMENT
Entered July 1, 2020 between Acme Corporation, an Ohio corporation ("Supplier"),
and GlobalDist Ltd., a United Kingdom company ("Distributor").
TERM. In full force from July 1, 2020 and expires June 30, 2023.
RENEWAL. Either party may request renewal by providing written notice 60 days before expiration.
GOVERNING LAW. Governed by laws of the State of Ohio.
TERMINATION FOR CAUSE. Either party may terminate if the other becomes insolvent or fails to remedy
a breach within 45 days of written notice.
PAYMENT. Distributor pays within 30 days of invoice.
DAMAGES. Breach of exclusivity clause results in liquidated damages of $200,000 per occurrence.""",
"gt":{"party_1":"Acme Corporation","party_2":"GlobalDist Ltd.",
"effective_date":"July 1, 2020","expiration_date":"June 30, 2023",
"governing_law":"State of Ohio",
"renewal":"Either party may request renewal by providing written notice 60 days before expiration",
"termination_for_cause":"terminate if the other becomes insolvent or fails to remedy breach within 45 days",
"payment_terms":"Distributor pays within 30 days of invoice",
"penalties":"liquidated damages of $200,000 per occurrence for breach of exclusivity"}
},
{
"name":"Sterling Properties Lease",
"text":"""COMMERCIAL LEASE AGREEMENT
Entered September 15, 2022, between Sterling Properties LLC, a Florida LLC ("Landlord"),
and PrimeRetail Holdings Corp., a Georgia corporation ("Tenant").
LEASE TERM. Begins October 1, 2022 and ends September 30, 2027.
RENEWAL OPTION. Tenant has two options to renew for five years each, with 180 days notice.
GOVERNING LAW. Governed by laws of the State of Florida.
DEFAULT. Landlord may terminate upon 30 days written notice if Tenant fails to pay rent for two months.
RENT. Monthly rent $15,000 due on the first day of each month.
LATE FEE. Rent not received within 5 days incurs a late fee of $750 per day.""",
"gt":{"party_1":"Sterling Properties LLC","party_2":"PrimeRetail Holdings Corp.",
"effective_date":"September 15, 2022","expiration_date":"September 30, 2027",
"governing_law":"State of Florida",
"renewal":"two options to renew for five years each with 180 days notice",
"termination_for_cause":"terminate upon 30 days written notice if Tenant fails to pay rent for two months",
"payment_terms":"$15,000 monthly due on the first day of each month","penalties":"$750 per day late fee"}
},
{
"name":"Vertex Capital Tech Services",
"text":"""TECHNOLOGY SERVICES AGREEMENT
Entered May 1, 2024 between Vertex Capital Partners LLC, a Delaware LLC ("Client"),
and Zenith Digital Solutions Inc., a Massachusetts corporation ("Service Provider").
TERM. Effective May 1, 2024 through April 30, 2026.
RENEWAL. Automatically renews for one-year periods unless 60 days notice given.
GOVERNING LAW. Governed by laws of the Commonwealth of Massachusetts.
TERMINATION FOR CAUSE. Either party may terminate upon 30 days notice for material breach uncured within 30 days.
PAYMENT. Client pays $8,500 per month within 30 days of invoice. Late payments accrue 1.5% monthly interest.
PENALTIES. Data breach caused by Provider negligence results in liquidated damages of $75,000 per incident.""",
"gt":{"party_1":"Vertex Capital Partners LLC","party_2":"Zenith Digital Solutions Inc.",
"effective_date":"May 1, 2024","expiration_date":"April 30, 2026",
"governing_law":"Commonwealth of Massachusetts",
"renewal":"automatically renews for one-year periods unless 60 days notice given",
"termination_for_cause":"terminate upon 30 days notice for material breach uncured within 30 days",
"payment_terms":"$8,500 per month within 30 days of invoice","penalties":"$75,000 liquidated damages per data breach incident"}
},
{
"name":"AlphaGen Software License",
"text":"""SOFTWARE LICENSE AGREEMENT
This Agreement is dated August 1, 2023, between AlphaGen Technologies Corp., a Washington corporation ("Licensor"),
and DataStream Analytics Inc., a Virginia corporation ("Licensee").
EFFECTIVE DATE. License effective September 1, 2023 through August 31, 2025.
RENEWAL. License renews automatically for one-year terms unless either party gives 90 days notice.
GOVERNING LAW. This Agreement shall be governed by the laws of the State of Washington.
TERMINATION. Licensor may terminate immediately upon written notice if Licensee breaches confidentiality.
LICENSE FEE. Licensee pays $50,000 annual license fee, due January 1 each year.
PENALTIES. Unauthorized use of software results in penalty of $500 per day per violation.""",
"gt":{"party_1":"AlphaGen Technologies Corp.","party_2":"DataStream Analytics Inc.",
"effective_date":"September 1, 2023","expiration_date":"August 31, 2025",
"governing_law":"State of Washington",
"renewal":"renews automatically for one-year terms unless either party gives 90 days notice",
"termination_for_cause":"terminate immediately upon written notice if Licensee breaches confidentiality",
"payment_terms":"$50,000 annual license fee due January 1 each year",
"penalties":"$500 per day per violation for unauthorized use"}
},
{
"name":"BlueSky NDA Agreement",
"text":"""NON-DISCLOSURE AGREEMENT
This NDA is entered into as of June 10, 2023, between BlueSky Ventures Inc., a Nevada corporation ("Disclosing Party"),
and Quantum Research Group LLC, a Colorado LLC ("Receiving Party").
TERM. This Agreement is effective June 10, 2023 and expires June 9, 2026.
RENEWAL. Agreement does not automatically renew. Parties must execute a new agreement.
GOVERNING LAW. Governed by the laws of the State of Nevada.
TERMINATION. Either party may terminate this Agreement upon 30 days written notice.
COMPENSATION. No compensation is payable under this Agreement.
PENALTIES. Breach of confidentiality obligations results in liquidated damages of $250,000 plus attorney fees.""",
"gt":{"party_1":"BlueSky Ventures Inc.","party_2":"Quantum Research Group LLC",
"effective_date":"June 10, 2023","expiration_date":"June 9, 2026",
"governing_law":"State of Nevada",
"renewal":"Agreement does not automatically renew. Parties must execute a new agreement",
"termination_for_cause":"Either party may terminate upon 30 days written notice",
"payment_terms":"No compensation payable","penalties":"$250,000 liquidated damages plus attorney fees for breach"}
},
{
"name":"PrimeCare Consulting Agreement",
"text":"""CONSULTING AGREEMENT
Dated February 15, 2022, between PrimeCare Health Systems Inc., an Illinois corporation ("Company"),
and MedTech Advisors LLC, a Michigan LLC ("Consultant").
TERM. Services commence March 1, 2022 and terminate February 28, 2023.
RENEWAL. Company may renew for additional one-year terms with 30 days written notice before expiration.
GOVERNING LAW. Governed by the laws of the State of Illinois.
TERMINATION FOR CAUSE. Company may terminate immediately if Consultant violates HIPAA or commits fraud.
FEES. Consultant receives $15,000 per month payable on the 15th of each month.
PENALTIES. Unauthorized disclosure of patient data results in $500,000 penalty per incident.""",
"gt":{"party_1":"PrimeCare Health Systems Inc.","party_2":"MedTech Advisors LLC",
"effective_date":"February 15, 2022","expiration_date":"February 28, 2023",
"governing_law":"State of Illinois",
"renewal":"Company may renew for additional one-year terms with 30 days written notice",
"termination_for_cause":"terminate immediately if Consultant violates HIPAA or commits fraud",
"payment_terms":"$15,000 per month payable on the 15th","penalties":"$500,000 penalty per patient data disclosure incident"}
},
{
"name":"EcoSupply Manufacturing Agmt",
"text":"""MANUFACTURING AGREEMENT
Entered November 1, 2021, between EcoSupply Corporation, a Minnesota corporation ("Buyer"),
and PrecisionMfg Ltd., a Wisconsin corporation ("Manufacturer").
TERM. Effective November 1, 2021 through October 31, 2024.
RENEWAL. Agreement renews for two-year periods unless either party gives 120 days notice.
GOVERNING LAW. This Agreement is governed by the laws of the State of Minnesota.
TERMINATION. Buyer may terminate for cause if Manufacturer fails quality standards for three consecutive months.
PURCHASE PRICE. Buyer pays $200 per unit, invoiced monthly, net 60 days.
LIQUIDATED DAMAGES. Delivery delays of more than 30 days result in $10,000 per day penalty.""",
"gt":{"party_1":"EcoSupply Corporation","party_2":"PrecisionMfg Ltd.",
"effective_date":"November 1, 2021","expiration_date":"October 31, 2024",
"governing_law":"State of Minnesota",
"renewal":"renews for two-year periods unless either party gives 120 days notice",
"termination_for_cause":"terminate if Manufacturer fails quality standards for three consecutive months",
"payment_terms":"$200 per unit invoiced monthly net 60 days",
"penalties":"$10,000 per day penalty for delivery delays over 30 days"}
},
{
"name":"GlobalMedia Partnership Agmt",
"text":"""PARTNERSHIP AGREEMENT
This Agreement is made April 1, 2023, between GlobalMedia Productions Inc., a New York corporation ("Partner A"),
and StreamVision Entertainment LLC, a California LLC ("Partner B").
TERM. Partnership commences April 1, 2023 and terminates March 31, 2025.
RENEWAL. Partnership may be extended by mutual written agreement 90 days before expiration.
GOVERNING LAW. This Agreement is governed by the laws of the State of New York.
TERMINATION FOR CAUSE. Either Partner may terminate if the other commits fraud or material breach uncured in 20 days.
REVENUE SHARING. Net profits split 60% to Partner A and 40% to Partner B monthly.
PENALTIES. Unauthorized disclosure of trade secrets results in $1,000,000 in liquidated damages.""",
"gt":{"party_1":"GlobalMedia Productions Inc.","party_2":"StreamVision Entertainment LLC",
"effective_date":"April 1, 2023","expiration_date":"March 31, 2025",
"governing_law":"State of New York",
"renewal":"may be extended by mutual written agreement 90 days before expiration",
"termination_for_cause":"terminate if the other commits fraud or material breach uncured in 20 days",
"payment_terms":"Net profits 60% to Partner A and 40% to Partner B monthly",
"penalties":"$1,000,000 liquidated damages for unauthorized disclosure of trade secrets"}
},
{
"name":"SunTech Employment Contract",
"text":"""EMPLOYMENT AGREEMENT
Dated January 1, 2024, between SunTech Industries Inc., a Georgia corporation ("Employer"),
and James R. Mitchell ("Employee").
TERM. Employment commences January 15, 2024 and continues through January 14, 2026.
RENEWAL. Agreement may be renewed by mutual consent 60 days before expiration.
GOVERNING LAW. Governed by the laws of the State of Georgia.
TERMINATION FOR CAUSE. Employer may terminate immediately for gross misconduct, theft, or repeated policy violations.
COMPENSATION. Employee receives $120,000 annual salary paid bi-weekly.
PENALTIES. Violation of non-compete clause results in liquidated damages of $150,000.""",
"gt":{"party_1":"SunTech Industries Inc.","party_2":"James R. Mitchell",
"effective_date":"January 1, 2024","expiration_date":"January 14, 2026",
"governing_law":"State of Georgia",
"renewal":"may be renewed by mutual consent 60 days before expiration",
"termination_for_cause":"terminate immediately for gross misconduct theft or repeated policy violations",
"payment_terms":"$120,000 annual salary paid bi-weekly",
"penalties":"$150,000 liquidated damages for violation of non-compete clause"}
},
{
"name":"FreshFoods Supply Agreement",
"text":"""SUPPLY AGREEMENT
This Agreement is entered October 15, 2022, between FreshFoods Corp., an Iowa corporation ("Buyer"),
and AgriProduce LLC, a Nebraska LLC ("Supplier").
TERM. Effective October 15, 2022 through October 14, 2025.
RENEWAL. Automatically renews for one-year periods unless 45 days written notice given before expiration.
GOVERNING LAW. Governed by the laws of the State of Iowa.
TERMINATION. Buyer may terminate if Supplier fails two consecutive food safety inspections.
PAYMENT. Buyer pays within 21 days of delivery. Volume discount of 5% on orders over $100,000.
PENALTIES. Short deliveries incur penalty of 15% of the value of the shortfall.""",
"gt":{"party_1":"FreshFoods Corp.","party_2":"AgriProduce LLC",
"effective_date":"October 15, 2022","expiration_date":"October 14, 2025",
"governing_law":"State of Iowa",
"renewal":"automatically renews for one-year periods unless 45 days written notice given",
"termination_for_cause":"terminate if Supplier fails two consecutive food safety inspections",
"payment_terms":"Buyer pays within 21 days of delivery, 5% discount on orders over $100,000",
"penalties":"15% penalty on value of shortfall for short deliveries"}
},
{
"name":"CityArc Research Agreement",
"text":"""RESEARCH AGREEMENT
Dated September 1, 2023, between CityArc University, a Pennsylvania nonprofit ("University"),
and BioInnovate Labs Inc., a New Jersey corporation ("Sponsor").
TERM. Research period begins October 1, 2023 and ends September 30, 2025.
RENEWAL. Sponsor may extend for one additional year with 60 days written notice.
GOVERNING LAW. Governed by the laws of the Commonwealth of Pennsylvania.
TERMINATION. University may terminate if Sponsor fails to pay within 60 days of invoice.
FUNDING. Sponsor provides $500,000 per year payable in quarterly installments.
PENALTIES. Misuse of research results by Sponsor results in damages of $2,000,000.""",
"gt":{"party_1":"CityArc University","party_2":"BioInnovate Labs Inc.",
"effective_date":"September 1, 2023","expiration_date":"September 30, 2025",
"governing_law":"Commonwealth of Pennsylvania",
"renewal":"Sponsor may extend for one additional year with 60 days written notice",
"termination_for_cause":"terminate if Sponsor fails to pay within 60 days of invoice",
"payment_terms":"$500,000 per year in quarterly installments",
"penalties":"$2,000,000 damages for misuse of research results"}
},
{
"name":"MaxBrand Marketing Agreement",
"text":"""MARKETING SERVICES AGREEMENT
This Agreement is made July 1, 2022, between MaxBrand Retail Corp., an Arizona corporation ("Client"),
and CreativePulse Agency LLC, an Oregon LLC ("Agency").
TERM. Services begin August 1, 2022 and conclude July 31, 2024.
RENEWAL. Agreement renews automatically for one-year terms unless 30 days notice given before expiration.
GOVERNING LAW. This Agreement is governed by the laws of the State of Arizona.
TERMINATION FOR CAUSE. Client may terminate if Agency misses three consecutive campaign deadlines.
FEES. Client pays Agency $20,000 per month plus 10% of ad spend, invoiced on the 1st, due within 15 days.
PENALTIES. Unauthorized use of Client brand assets results in $50,000 penalty per incident.""",
"gt":{"party_1":"MaxBrand Retail Corp.","party_2":"CreativePulse Agency LLC",
"effective_date":"July 1, 2022","expiration_date":"July 31, 2024",
"governing_law":"State of Arizona",
"renewal":"renews automatically for one-year terms unless 30 days notice given",
"termination_for_cause":"terminate if Agency misses three consecutive campaign deadlines",
"payment_terms":"$20,000 per month plus 10% of ad spend due within 15 days",
"penalties":"$50,000 per incident for unauthorized use of brand assets"}
},
{
"name":"NorthStar Joint Venture Agmt",
"text":"""JOINT VENTURE AGREEMENT
Entered March 15, 2023, between NorthStar Energy Inc., a Texas corporation ("Party A"),
and SolarWave Technologies LLC, a Colorado LLC ("Party B").
TERM. Joint venture begins April 1, 2023 and terminates March 31, 2028.
RENEWAL. Parties may extend by mutual written consent at least 180 days before termination.
GOVERNING LAW. Governed by the laws of the State of Texas.
TERMINATION FOR CAUSE. Either party may dissolve the venture if the other is convicted of a felony or
becomes bankrupt, with 30 days written notice.
PROFIT SHARING. Profits and losses shared equally 50/50 distributed quarterly.
PENALTIES. Withdrawal from venture before term results in buyout penalty of $5,000,000.""",
"gt":{"party_1":"NorthStar Energy Inc.","party_2":"SolarWave Technologies LLC",
"effective_date":"March 15, 2023","expiration_date":"March 31, 2028",
"governing_law":"State of Texas",
"renewal":"Parties may extend by mutual written consent at least 180 days before termination",
"termination_for_cause":"dissolve if the other is convicted of a felony or becomes bankrupt with 30 days notice",
"payment_terms":"Profits and losses shared 50/50 distributed quarterly",
"penalties":"$5,000,000 buyout penalty for early withdrawal"}
},
]


# ── Helpers ───────────────────────────────────────────────────────────

def token_f1(pred: str, gold: str) -> float:
    if not gold.strip():
        return 1.0 if "not found" in pred.lower() else 0.5
    if not pred.strip() or "not found" in pred.lower():
        return 0.0
    p = set(pred.lower().split())
    g = set(gold.lower().split())
    c = p & g
    if not c: return 0.0
    prec = len(c)/len(p); rec = len(c)/len(g)
    return 2*prec*rec/(prec+rec)

def badge(f1: float) -> str:
    if f1 >= 0.7: return "[OK]"
    if f1 >= 0.3: return "[~~]"
    return "[--]"


# ── Run one pass ──────────────────────────────────────────────────────

def run_pass(mode: str) -> list:
    results = []
    for idx, c in enumerate(CONTRACTS):
        name = c["name"]
        print(f"  [{idx+1:02d}/15] {name:<35}", end=" ", flush=True)
        t0 = time.perf_counter()
        prof = pipeline.extract(c["text"], contract_name=name, mode=mode)
        elapsed = time.perf_counter() - t0
        d = pipeline.profile_to_dict(prof)

        row = {"name": name, "fields": {}}
        scores = []
        for field in FIELDS:
            extracted = d.get(field, {}).get("value", "Not found in contract")
            gold = c["gt"].get(field, "")
            f1 = token_f1(extracted, gold)
            scores.append(f1)
            row["fields"][field] = {
                "extracted": extracted[:100],
                "gold":      gold[:100],
                "f1":        round(f1, 3),
                "badge":     badge(f1),
            }
        row["mean_f1"] = round(sum(scores)/len(scores), 3)
        print(f"mean F1={row['mean_f1']:.2f}  ({elapsed:.0f}s)")
        results.append(row)
    return results


# ── Print table ───────────────────────────────────────────────────────

def print_table(results: list, model_name: str) -> dict:
    cols = [SHORT[f] for f in FIELDS]
    W = 100

    print(f"\n{'='*W}")
    print(f"  {model_name} — 15 CONTRACTS")
    print(f"{'='*W}")
    hdr = f"{'Contract':<36}" + "".join(f"{c:>7}" for c in cols) + "   Mean"
    print(hdr)
    print("-"*W)

    for row in results:
        line = f"{row['name']:<36}"
        for field in FIELDS:
            b = row["fields"][field]["badge"]
            line += f"{b:>7}"
        line += f"   {row['mean_f1']:.2f}"
        print(line)

    print("-"*W)

    acc = {}
    acc_line = f"{'Accuracy':<36}"
    for field in FIELDS:
        scores = [r["fields"][field]["f1"] for r in results]
        a = sum(1 for s in scores if s >= 0.7) / len(scores) * 100
        acc[field] = a
        acc_line += f"  {a:>3.0f}%"
    overall = sum(acc.values())/len(acc)
    print(acc_line + f"   {overall:.0f}%")
    print(f"{'='*W}")
    return acc


def print_comparison(rb_acc: dict, gem_acc: dict):
    print(f"\n{'='*60}")
    print("  FIELD BY FIELD — WHICH MODEL WINS")
    print(f"{'='*60}")
    print(f"{'Field':<25} {'RoBERTa':>9} {'Gemini':>9}  Winner")
    print("-"*60)
    rb_w = gem_w = 0
    for field in FIELDS:
        rb  = rb_acc.get(field,0)
        gem = gem_acc.get(field,0)
        if rb > gem:   winner = "RoBERTa"; rb_w += 1
        elif gem > rb: winner = "Gemini "; gem_w += 1
        else:          winner = "Tie    "
        print(f"{field:<25} {rb:>8.0f}%  {gem:>8.0f}%  {winner}")
    print("-"*60)
    rb_o  = sum(rb_acc.values())/len(rb_acc)
    gem_o = sum(gem_acc.values())/len(gem_acc)
    winner = "RoBERTa" if rb_o > gem_o else "Gemini "
    print(f"{'Overall':<25} {rb_o:>8.0f}%  {gem_o:>8.0f}%  {winner}")
    print(f"{'='*60}")
    print(f"\nRoBERTa wins {rb_w} field(s) | Gemini wins {gem_w} field(s)")


def print_failures(rb: list, gem: list):
    failures = []
    for rb_row, gem_row in zip(rb, gem):
        for field in FIELDS:
            rf = rb_row["fields"][field]["f1"]
            gf = gem_row["fields"][field]["f1"]
            if rf < 0.3 and gf < 0.3:
                failures.append({
                    "contract": rb_row["name"],
                    "field":    field,
                    "rb":       rb_row["fields"][field]["extracted"],
                    "gem":      gem_row["fields"][field]["extracted"],
                    "gold":     rb_row["fields"][field]["gold"],
                    "rf": rf, "gf": gf,
                })
    failures.sort(key=lambda x: x["rf"]+x["gf"])
    top5 = failures[:5]

    print(f"\n{'='*80}")
    print("  TOP 5 FAILURES  (both models wrong — use these for your failure analysis)")
    print(f"{'='*80}")
    for i, f in enumerate(top5, 1):
        print(f"\n{i}. [{f['field']}]  {f['contract']}")
        print(f"   Ground Truth : {f['gold'][:75]!r}")
        print(f"   RoBERTa said : {f['rb'][:75]!r}  (F1={f['rf']:.2f})")
        print(f"   Gemini  said : {f['gem'][:75]!r}  (F1={f['gf']:.2f})")


def save(rb_results, gem_results, rb_acc, gem_acc):
    out = {
        "roberta": {"results": rb_results, "accuracy": rb_acc},
        "gemini":  {"results": gem_results, "accuracy": gem_acc},
    }
    with open(OUTPUT_DIR/"full_results.json","w") as f:
        json.dump(out, f, indent=2)

    with open(OUTPUT_DIR/"summary_table.txt","w",encoding="utf-8") as f:
        f.write("CUAD 15-CONTRACT EVALUATION SUMMARY\n\n")
        f.write(f"{'Field':<25} {'RoBERTa':>10} {'Gemini':>10}  Winner\n")
        f.write("-"*55+"\n")
        for field in FIELDS:
            rb  = rb_acc.get(field,0)
            gem = gem_acc.get(field,0)
            winner = "RoBERTa" if rb > gem else "Gemini"
            f.write(f"{field:<25} {rb:>9.0f}%  {gem:>9.0f}%  {winner}\n")
        rb_o  = sum(rb_acc.values())/len(rb_acc)
        gem_o = sum(gem_acc.values())/len(gem_acc)
        f.write(f"\nOverall RoBERTa: {rb_o:.0f}%\n")
        f.write(f"Overall Gemini:  {gem_o:.0f}%\n")

    print(f"\nSaved: {OUTPUT_DIR/'full_results.json'}")
    print(f"Saved: {OUTPUT_DIR/'summary_table.txt'}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  CUAD 15-CONTRACT EVALUATION")
    print(f"  RoBERTa: {cuad_qa.MODEL_ID}")
    print(f"  Gemini:  {pipeline.GEMINI_AVAILABLE}")
    print("="*60)

    # Pass 1 — RoBERTa
    print("\nPASS 1 — RoBERTa only")
    print("-"*60)
    rb_results = run_pass("roberta")
    rb_acc = print_table(rb_results, "ROBERTA")

    # Pass 2 — Gemini
    print("\nPASS 2 — Gemini only")
    print("-"*60)
    if pipeline.GEMINI_AVAILABLE:
        gem_results = run_pass("gemini")
        gem_acc = print_table(gem_results, "GEMINI")
        print_comparison(rb_acc, gem_acc)
        print_failures(rb_results, gem_results)
        save(rb_results, gem_results, rb_acc, gem_acc)
    else:
        print("  Gemini not available. Add GEMINI_API_KEY to backend/.env")
        save(rb_results, rb_results, rb_acc, {f:0 for f in FIELDS})

    print("\nDone. Check evaluation_results/ for saved files.")

if __name__ == "__main__":
    main()
