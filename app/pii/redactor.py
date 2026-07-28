import re
from dataclasses import dataclass, field

# PII pattern definitions
_PATTERNS: dict[str, str] = {
    "EMIRATES_ID": r"\b784-?\d{4}-?\d{7}-?\d\b",
    "IBAN": r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,4}\d{7,19}\b",
    "ACCOUNT_NUMBER": r"\b\d{10,16}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "UAE_PHONE": r"(?:\+971|00971|0)(?:5[024568]|[234679])\d{7}\b",
    "PERSON_NAME": r"(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Sheikh|Shaikh)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}",
}

_REPLACEMENTS: dict[str, str] = {
    "EMIRATES_ID": "[EMIRATES_ID_REDACTED]",
    "IBAN": "[IBAN_REDACTED]",
    "ACCOUNT_NUMBER": "[ACCOUNT_NUM_REDACTED]",
    "EMAIL": "[EMAIL_REDACTED]",
    "UAE_PHONE": "[PHONE_REDACTED]",
    "PERSON_NAME": "[NAME_REDACTED]",
}


@dataclass
class RedactionResult:
    redacted_text: str
    detected_types: list[str] = field(default_factory=list)


def redact(text: str) -> RedactionResult:
    """Detect and mask PII in text. Returns redacted string and list of PII types found."""
    detected: list[str] = []
    result = text

    # Order matters: EMIRATES_ID and IBAN before ACCOUNT_NUMBER to avoid partial matches
    for pii_type in ["EMIRATES_ID", "IBAN", "PERSON_NAME", "EMAIL", "UAE_PHONE", "ACCOUNT_NUMBER"]:
        pattern = _PATTERNS[pii_type]
        if re.search(pattern, result, flags=re.IGNORECASE):
            detected.append(pii_type)
            result = re.sub(pattern, _REPLACEMENTS[pii_type], result, flags=re.IGNORECASE)

    return RedactionResult(redacted_text=result, detected_types=detected)
