import json
from pathlib import Path
from typing import Optional, Dict

REGISTRY_PATH = Path("data/sharia_docs/registry.json")

def load_registry() -> Dict[str, dict]:
    """Load the PDF registry mapping from disk."""
    if not REGISTRY_PATH.exists():
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_registry(registry: Dict[str, dict]) -> None:
    """Save the PDF registry mapping to disk."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")

def register_pdf(source_name: str, filename: str, cloudinary_url: str) -> None:
    """Register a new PDF mapping in the registry."""
    registry = load_registry()
    registry[source_name] = {
        "filename": filename,
        "cloudinary_url": cloudinary_url,
    }
    save_registry(registry)

def get_pdf_url(source_name: str) -> Optional[str]:
    """Retrieve the Cloudinary PDF URL for a given source name."""
    registry = load_registry()
    item = registry.get(source_name)
    if item:
        return item.get("cloudinary_url")
    return None

def unregister_pdf(source_name: str) -> bool:
    """Remove a PDF from the registry."""
    registry = load_registry()
    if source_name in registry:
        del registry[source_name]
        save_registry(registry)
        return True
    return False
