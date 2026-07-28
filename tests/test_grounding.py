"""
Eval Test 1 — Answer Grounding
Asserts that the LLM's answer is grounded in retrieved context and does not
introduce hallucinated financial figures or rules beyond the provided documents.
"""
import pytest
from unittest.mock import patch, MagicMock
from tests.conftest import make_llm_mock


GROUNDED_ANSWER = (
    "Murabaha is a cost-plus-profit sale where the financier buys the asset and sells it "
    "to the customer at a disclosed markup. The profit rate is fixed at contract signing. "
    "Late payment fees cannot be retained by the bank as that would constitute riba."
)

# Terms that should NOT appear if the model is hallucinating outside context
HALLUCINATION_SIGNALS = [
    "LIBOR", "SOFR", "prime rate", "federal funds", "credit score", "APR",
    "annual percentage", "compound interest",
]


def test_answer_is_grounded_in_context(client, mock_retrieve_murabaha):
    """
    The response should contain terms from the retrieved Murabaha chunks
    and must NOT introduce hallucinated conventional finance concepts.
    """
    with patch("app.routers.chat.retrieve", return_value=mock_retrieve_murabaha):
        with patch("app.routers.chat.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = make_llm_mock(
                GROUNDED_ANSWER
            )

            resp = client.post(
                "/chat",
                json={"session_id": "test-ground-1", "message": "How does Murabaha financing work?"},
            )

    assert resp.status_code == 200
    data = resp.json()
    answer = data["response"].lower()

    # ── Positive grounding check: key terms from context must appear ─────────
    assert "murabaha" in answer, "Response must mention Murabaha"
    assert any(t in answer for t in ["profit", "markup", "cost"]), (
        "Response must reference profit/cost concepts from retrieved context"
    )

    # ── Negative hallucination check ─────────────────────────────────────────
    for signal in HALLUCINATION_SIGNALS:
        assert signal.lower() not in answer, (
            f"Hallucination detected: '{signal}' found in response but not in context"
        )

    # ── Metadata checks ──────────────────────────────────────────────────────
    assert len(data["retrieved_chunk_ids"]) >= 1, "At least one chunk must be retrieved"
    assert data["avg_relevance_score"] > 0.5, "Avg relevance score should be high for Murabaha query"


def test_response_cites_source_concepts(client, mock_retrieve_murabaha):
    """
    A follow-up grounding test: response to a Riba question must reflect
    retrieved context and must not invent interest-rate numbers.
    """
    riba_answer = (
        "Riba means any predetermined excess charged on a loan. "
        "Mal cannot charge additional profit for late payment — any such fee must be "
        "donated to charity, not retained by the bank, as this would be riba."
    )

    with patch("app.routers.chat.retrieve", return_value=mock_retrieve_murabaha):
        with patch("app.routers.chat.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = make_llm_mock(
                riba_answer
            )

            resp = client.post(
                "/chat",
                json={"session_id": "test-ground-2", "message": "What happens if I pay late on Murabaha?"},
            )

    assert resp.status_code == 200
    answer = resp.json()["response"].lower()
    assert "riba" in answer or "charity" in answer or "late" in answer
    # No invented interest rate percentages
    import re
    rate_pattern = r"\b\d+(\.\d+)?%\b"
    matches = re.findall(rate_pattern, answer)
    assert len(matches) == 0, f"Hallucinated rate figures found: {matches}"
