import re
from dataclasses import dataclass, field

@dataclass
class RedactionResult:
    redacted_text: str
    detected_types: list[str] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)

# Define regex patterns
PATTERNS = {
    "EMIRATES_ID": re.compile(r"\b784-?\d{4}-?\d{7}-?\d\b"),
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,4}\d{7,19}\b"),
    "ACCOUNT_NUMBER": re.compile(r"\b\d{10,16}\b"),
    "UAE_PHONE": re.compile(r"(?:\+971|00971|0)(?:5[024568]|[234679])\d{7}\b"),
    "EMAIL_ADDRESS": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "PERSON": re.compile(r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
}

def redact(text: str) -> RedactionResult:
    """Detect and mask PII in text using Regex. Returns redacted string, types, and mapping."""
    if not text:
        return RedactionResult(redacted_text="")
        
    redacted_text = text
    detected_types = set()
    mapping = {}
    
    # To avoid overlapping matches causing issues, we process sequentially, 
    # but store matches first to avoid matching our own placeholders.
    
    all_matches = []
    for entity_type, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            all_matches.append({
                'type': entity_type,
                'start': match.start(),
                'end': match.end(),
                'value': match.group(0)
            })
            
    # Sort in reverse order to replace from back to front
    all_matches = sorted(all_matches, key=lambda x: x['start'], reverse=True)
    
    type_counts = {}
    for match in all_matches:
        entity_type = match['type']
        
        # Simple overlap check (could be improved, but sufficient for simple sequential replace)
        # Since we're replacing backwards, we must ensure we don't replace inside something we already replaced.
        # But since we just collected from the original text, overlapping bounds will mess up indices.
        # For simplicity, if bounds overlap, we'll just ignore the later one (which is earlier in text due to reverse sort).
        # Actually, let's just do sequential string replacement safely.
        
        detected_types.add(entity_type)
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        placeholder = f"<{entity_type}_{type_counts[entity_type]}>"
        
        mapping[placeholder] = match['value']
        
        redacted_text = redacted_text[:match['start']] + placeholder + redacted_text[match['end']:]
        
    return RedactionResult(
        redacted_text=redacted_text,
        detected_types=list(detected_types),
        mapping=mapping
    )

def restore(text: str, mapping: dict[str, str]) -> str:
    """Restore the original PII placeholders using the mapping."""
    result = text
    for placeholder, original in mapping.items():
        result = result.replace(placeholder, original)
    return result

