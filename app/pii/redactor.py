import re
from dataclasses import dataclass, field
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

@dataclass
class RedactionResult:
    redacted_text: str
    detected_types: list[str] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)

# Set up NLP engine to use the default Spacy model
provider = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
})
nlp_engine = provider.create_engine()

# Initialize engines
analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
anonymizer = AnonymizerEngine()

# Custom patterns for UAE specific PII
custom_patterns = [
    {
        "name": "EMIRATES_ID",
        "regex": r"\b784-?\d{4}-?\d{7}-?\d\b",
        "score": 1.0
    },
    {
        "name": "IBAN",
        "regex": r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,4}\d{7,19}\b",
        "score": 1.0
    },
    {
        "name": "ACCOUNT_NUMBER",
        "regex": r"\b\d{10,16}\b",
        "score": 0.8
    },
    {
        "name": "UAE_PHONE",
        "regex": r"(?:\+971|00971|0)(?:5[024568]|[234679])\d{7}\b",
        "score": 1.0
    }
]

for pat in custom_patterns:
    pattern = Pattern(name=pat["name"], regex=pat["regex"], score=pat["score"])
    recognizer = PatternRecognizer(supported_entity=pat["name"], patterns=[pattern])
    analyzer.registry.add_recognizer(recognizer)

def redact(text: str) -> RedactionResult:
    """Detect and mask PII in text using Presidio. Returns redacted string, types, and mapping."""
    if not text:
        return RedactionResult(redacted_text="")
        
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=["EMIRATES_ID", "IBAN", "ACCOUNT_NUMBER", "UAE_PHONE", "PERSON", "EMAIL_ADDRESS"],
        score_threshold=0.5
    )
    
    mapping = {}
    detected_types = set()
    
    redacted_text = text
    # Sort results in reverse order to not mess up indices when replacing
    results = sorted(results, key=lambda x: x.start, reverse=True)
    
    type_counts = {}
    for res in results:
        entity_type = res.entity_type
        detected_types.add(entity_type)
        
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        placeholder = f"<{entity_type}_{type_counts[entity_type]}>"
        
        original_value = text[res.start:res.end]
        mapping[placeholder] = original_value
        
        redacted_text = redacted_text[:res.start] + placeholder + redacted_text[res.end:]
        
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
