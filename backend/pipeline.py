"""
Multi-field contract extraction pipeline.

Field -> model assignment (with Gemini key set):
+-------------------------+--------------+----------------------------------------+
| Field                   | Model        | Reason                                 |
+-------------------------+--------------+----------------------------------------+
| party_1 / party_2       | Gemini       | Names in preamble, need context        |
| effective_date          | RoBERTa      | Concrete date span in text             |
| expiration_date         | RoBERTa      | Concrete date span in text             |
| governing_law           | RoBERTa      | Short phrase, direct span              |
| renewal                 | RoBERTa      | Best CUAD field for RoBERTa (F1=0.56)  |
| termination_for_cause   | Gemini       | Multi-sentence clause                  |
| payment_terms           | Gemini       | Spread across sections                 |
| penalties               | Gemini       | Context-heavy, conditional             |
+-------------------------+--------------+----------------------------------------+

Without Gemini key all 9 fields use RoBERTa-base-CUAD.
"""

import os
import time
from dataclasses import asdict, dataclass

import cuad_qa
import gemini_client
import postprocess

GEMINI_AVAILABLE = (
    bool(os.environ.get("GEMINI_API_KEY", "")) and
    "your_gemini" not in os.environ.get("GEMINI_API_KEY", "")
)

# Fields that should use Gemini when available
PREFER_GEMINI = {"party_1", "party_2", "termination_for_cause", "payment_terms", "penalties"}

ALL_FIELDS = [
    "party_1", "party_2",
    "effective_date", "expiration_date", "governing_law",
    "renewal", "termination_for_cause", "payment_terms", "penalties",
]


@dataclass
class FieldResult:
    value: str
    confidence: float
    source: str
    latency_ms: int
    cached: bool = False


@dataclass
class ContractProfile:
    party_1: FieldResult
    party_2: FieldResult
    effective_date: FieldResult
    expiration_date: FieldResult
    renewal: FieldResult
    governing_law: FieldResult
    termination_for_cause: FieldResult
    payment_terms: FieldResult
    penalties: FieldResult
    total_latency_ms: int
    contract_name: str = ""


def _roberta(field: str, text: str) -> FieldResult:
    res = cuad_qa.extract_field(field, text)
    if res is None:
        return FieldResult("Not found in contract", 0.0, "roberta", 0)
    cleaned = postprocess.clean_field(field, res.answer)
    return FieldResult(cleaned, round(res.score, 3), "roberta", res.latency_ms)


def _gemini(field: str, text: str) -> FieldResult:
    res = gemini_client.extract_field(field, text)
    ans = res.get("answer", "Not found in contract")
    conf = 0.85 if "Not found" not in ans and "error" not in ans.lower() else 0.0
    return FieldResult(ans, conf, "gemini", res.get("latency_ms", 0), res.get("cached", False))


def _extract_field(field: str, text: str, mode: str = "hybrid") -> FieldResult:
    if mode == "roberta":
        return _roberta(field, text)
    if mode == "gemini":
        if not GEMINI_AVAILABLE:
            return FieldResult("Gemini API key not set", 0.0, "gemini(unavailable)", 0)
        return _gemini(field, text)
    # hybrid — smart assignment per field
    if GEMINI_AVAILABLE and field in PREFER_GEMINI:
        return _gemini(field, text)
    return _roberta(field, text)


def extract(contract_text: str, contract_name: str = "", mode: str = "hybrid") -> ContractProfile:
    t_start = time.perf_counter()
    results: dict[str, FieldResult] = {}
    for field in ALL_FIELDS:
        results[field] = _extract_field(field, contract_text, mode)

    total_ms = int((time.perf_counter() - t_start) * 1000)

    return ContractProfile(
        party_1=results["party_1"],
        party_2=results["party_2"],
        effective_date=results["effective_date"],
        expiration_date=results["expiration_date"],
        renewal=results["renewal"],
        governing_law=results["governing_law"],
        termination_for_cause=results["termination_for_cause"],
        payment_terms=results["payment_terms"],
        penalties=results["penalties"],
        total_latency_ms=total_ms,
        contract_name=contract_name,
    )


def profile_to_dict(profile: ContractProfile) -> dict:
    return asdict(profile)
