"""
Post-processing for RoBERTa span output.
Strips verbose spans down to clean core values.
"""

import re

# Date patterns
_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}\b"
    r"|"
    r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b",
    re.IGNORECASE,
)

# State/country law patterns
_STATE_RE = re.compile(
    r"(?:State|Commonwealth|Province|Republic|Country)\s+of\s+[\w\s]+?(?=,|\.|$|\s+without|\s+law)",
    re.IGNORECASE,
)

# Company/party name — ends at comma + legal suffix or bracket
_PARTY_RE = re.compile(
    r"([A-Z][A-Za-z0-9\s\.\,\-&']+?"
    r"(?:Inc\.|LLC|Ltd\.|Corp\.|Corporation|Limited|L\.P\.|LLP|LP|Company|Co\.))",
    re.IGNORECASE,
)

# Money amount patterns
_MONEY_RE = re.compile(
    r"\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand))?",
    re.IGNORECASE,
)

# Remove leading legal boilerplate phrases
_BOILERPLATE = re.compile(
    r"^(?:this agreement|the agreement|agreement|this contract|the contract|"
    r"this lease|the lease|this license|pursuant to|subject to|"
    r"in accordance with|as set forth in|as described in)\s+",
    re.IGNORECASE,
)


def _shorten(text: str, max_len: int = 150) -> str:
    text = text.strip()
    if len(text) > max_len:
        return text[:max_len].rstrip() + "…"
    return text


def clean_date(raw: str, prefer_last: bool = False) -> str:
    if not raw or "Not found" in raw:
        return raw
    if len(raw) <= 40:
        return raw
    matches = _DATE_RE.findall(raw)
    if not matches:
        return _shorten(raw, 80)
    return matches[-1] if prefer_last else matches[0]


def clean_governing_law(raw: str) -> str:
    if not raw or "Not found" in raw:
        return raw
    if len(raw) <= 40:
        return raw
    m = _STATE_RE.search(raw)
    return m.group(0).strip() if m else _shorten(raw, 80)


def clean_party(raw: str) -> str:
    if not raw or "Not found" in raw:
        return raw
    if len(raw) <= 60:
        return raw.strip()
    # try to extract a company name
    matches = _PARTY_RE.findall(raw)
    if matches:
        return matches[0].strip().rstrip(",")
    # strip boilerplate from front
    cleaned = _BOILERPLATE.sub("", raw).strip()
    # take up to first comma or bracket
    for sep in [" (", ", a ", ", an "]:
        if sep in cleaned:
            return cleaned[:cleaned.index(sep)].strip()
    return _shorten(cleaned, 80)


def clean_payment(raw: str) -> str:
    if not raw or "Not found" in raw:
        return raw
    # strip boilerplate
    cleaned = _BOILERPLATE.sub("", raw).strip()
    return _shorten(cleaned, 150)


def clean_penalties(raw: str) -> str:
    if not raw or "Not found" in raw:
        return raw
    cleaned = _BOILERPLATE.sub("", raw).strip()
    return _shorten(cleaned, 150)


def clean_renewal(raw: str) -> str:
    if not raw or "Not found" in raw:
        return raw
    cleaned = _BOILERPLATE.sub("", raw).strip()
    return _shorten(cleaned, 160)


def clean_termination(raw: str) -> str:
    if not raw or "Not found" in raw:
        return raw
    cleaned = _BOILERPLATE.sub("", raw).strip()
    return _shorten(cleaned, 160)


FIELD_CLEANERS = {
    "effective_date":        lambda s: clean_date(s, prefer_last=False),
    "expiration_date":       lambda s: clean_date(s, prefer_last=True),
    "governing_law":         clean_governing_law,
    "party_1":               clean_party,
    "party_2":               clean_party,
    "renewal":               clean_renewal,
    "termination_for_cause": clean_termination,
    "payment_terms":         clean_payment,
    "penalties":             clean_penalties,
}


def clean_field(field: str, raw_value: str) -> str:
    cleaner = FIELD_CLEANERS.get(field, lambda s: _shorten(s, 150))
    return cleaner(raw_value)
