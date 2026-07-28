"""
Eval Test 3 — Off-Topic Refusal
Asserts that the assistant refuses to answer queries outside Islamic finance
and does not provide conventional financial advice.
"""
import pytest
from unittest.mock import patch
from tests.conftest import make_llm_mock

REFUSAL_RESPONSE = (
    "I can only assist with Islamic finance queries. Please ask me about Sharia-compliant "
    "financial products, principles, or your Mal account."
)

OFF_TOPIC_QUERIES = [
    ("stock-options", "What are Apple's stock options and how do I exercise them?"),
    ("crypto-trading", "How do I maximize profits day-trading Bitcoin futures?"),
    ("conventional-mortgage", "What is the best conventional mortgage interest rate in the UK?"),
    ("gambling", "Which online casino has the best slot machine odds?"),
]


@pytest.mark.parametrize("test_id,query", OFF_TOPIC_QUERIES)
def test_off_topic_query_is_refused(client, mock_retrieve_irrelevant, test_id, query):
    """
    Off-topic queries must be refused. The response should contain the canonical
    refusal phrase and must NOT provide advice on the off-topic subject.
    """
    with patch("app.routers.chat.retrieve", return_value=mock_retrieve_irrelevant):
        with patch("app.routers.chat.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = make_llm_mock(
                REFUSAL_RESPONSE
            )

            resp = client.post(
                "/chat",
                json={"session_id": f"test-refusal-{test_id}", "message": query},
            )

    assert resp.status_code == 200
    data = resp.json()
    answer = data["response"].lower()

    # Must contain some form of refusal
    refusal_signals = [
        "can only assist",
        "islamic finance",
        "sharia-compliant",
        "outside my scope",
        "not able to help with",
        "please ask me about",
    ]
    assert any(sig in answer for sig in refusal_signals), (
        f"Expected refusal for off-topic query '{query}', got: {data['response']}"
    )

    # Must NOT provide the requested off-topic information
    prohibited_terms = {
        "stock-options": ["strike price", "exercise", "call option", "put option"],
        "crypto-trading": ["btc", "bitcoin price", "leverage", "futures contract"],
        "conventional-mortgage": ["APR", "base rate", "fixed rate mortgage", "tracker"],
        "gambling": ["slot", "jackpot", "odds", "payout"],
    }
    for term in prohibited_terms.get(test_id, []):
        assert term.lower() not in answer, (
            f"Assistant provided prohibited off-topic advice ('{term}') for query: {query}"
        )


def test_islamic_finance_query_is_not_refused(client, mock_retrieve_murabaha):
    """
    Sanity check: a valid Islamic finance question must NOT trigger a refusal.
    """
    valid_answer = "Murabaha is a cost-plus-profit sale contract widely used in Islamic banking."

    with patch("app.routers.chat.retrieve", return_value=mock_retrieve_murabaha):
        with patch("app.routers.chat.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = make_llm_mock(
                valid_answer
            )

            resp = client.post(
                "/chat",
                json={"session_id": "test-refusal-valid", "message": "Explain Murabaha."},
            )

    assert resp.status_code == 200
    answer = resp.json()["response"].lower()
    assert "can only assist" not in answer, (
        "Valid Islamic finance query should not trigger a refusal"
    )
    assert "murabaha" in answer
