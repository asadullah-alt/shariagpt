"""
Test fixtures and shared mocks for ShariaGPT eval tests.
LLM calls are mocked to avoid API costs in CI.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Shared LLM mock factory ──────────────────────────────────────────────────

def make_llm_mock(answer: str, prompt_tokens: int = 120, completion_tokens: int = 80):
    """Create an OpenAI-compatible mock completion response."""
    mock_choice = MagicMock()
    mock_choice.message.content = answer

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = prompt_tokens
    mock_usage.completion_tokens = completion_tokens
    mock_usage.total_tokens = prompt_tokens + completion_tokens

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage = mock_usage
    return mock_completion


# ── App client fixture ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    """
    TestClient with mocked Qdrant, Redis, and sentence-transformers so no
    external services are needed during CI testing.
    """
    import os
    os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
    os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
    os.environ.setdefault("QDRANT_API_KEY", "test-key")
    os.environ.setdefault("REDIS_URL", "")

    # Patch Qdrant client to avoid real network calls
    with patch("app.rag.vector_store.QdrantClient") as MockQdrant:
        mock_qdrant = MagicMock()
        mock_qdrant.get_collections.return_value.collections = []
        mock_qdrant.search.return_value = []
        MockQdrant.return_value = mock_qdrant

        # Patch embedder so no external API call is made
        with patch("app.rag.embedder.requests.post") as mock_post:
            def side_effect(*args, **kwargs):
                inp = kwargs.get("json", {}).get("input", "")
                n = 1 if isinstance(inp, str) else len(inp)
                mock_resp = MagicMock()
                mock_resp.json.return_value = {"data": [{"embedding": [0.1]*384, "index": i} for i in range(n)]}
                return mock_resp
            mock_post.side_effect = side_effect

            from app.main import app
            from app.auth.jwt_handler import require_auth
            app.dependency_overrides[require_auth] = lambda: {"sub": "testuser@example.com", "2fa_complete": True}
            yield TestClient(app, raise_server_exceptions=True)
            app.dependency_overrides.clear()


@pytest.fixture
def mock_retrieve_murabaha():
    """Return realistic Murabaha chunks as if retrieved from Qdrant."""
    return [
        {
            "chunk_id": "1000",
            "text": (
                "Murabaha is a cost-plus-profit sale contract. The financier purchases an asset "
                "at cost price and sells it to the customer at a disclosed markup. The total price "
                "and profit are agreed upfront. No floating or variable profit rates are permitted. "
                "The bank cannot charge additional profit if the customer is late — that would be riba."
            ),
            "source": "murabaha_overview",
            "score": 0.91,
        },
        {
            "chunk_id": "1001",
            "text": (
                "In Murabaha, the financier must first own the asset before selling it. "
                "Selling something not yet owned (bay al-ma'dum) is prohibited. "
                "Mal Murabaha profit rate is fixed and disclosed at contract signing."
            ),
            "source": "murabaha_overview",
            "score": 0.87,
        },
    ]


@pytest.fixture
def mock_retrieve_irrelevant():
    """Low-relevance chunks for off-topic queries."""
    return [
        {
            "chunk_id": "9999",
            "text": "Zakat is the third pillar of Islam — an obligatory annual payment.",
            "source": "zakat_calculation",
            "score": 0.22,
        }
    ]
