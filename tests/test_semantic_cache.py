import pytest
from unittest.mock import patch, MagicMock
from app.routers.chat import ChatResponse

def test_semantic_cache_hit_skips_llm(client, mock_retrieve_murabaha):
    """
    Test that a cache hit returns immediately without calling OpenAI.
    """
    cached_answer = "This is a cached Murabaha response."

    with patch("app.routers.chat.get_cached_response", return_value=cached_answer):
        with patch("app.routers.chat.OpenAI") as MockOpenAI:
            # We don't even configure the mock because it shouldn't be called
            
            resp = client.post(
                "/chat",
                json={"session_id": "test-cache", "message": "Tell me about Murabaha"},
            )

            assert resp.status_code == 200
            data = resp.json()
            assert data["response"] == cached_answer
            assert data["cache_hit"] is True
            assert len(data["retrieved_chunk_ids"]) == 0
            MockOpenAI.assert_not_called()


def test_semantic_cache_miss_calls_llm(client, mock_retrieve_murabaha):
    """
    Test that a cache miss calls the LLM and then saves to cache.
    """
    from tests.conftest import make_llm_mock
    
    new_answer = "This is a fresh LLM response."

    with patch("app.routers.chat.get_cached_response", return_value=None):
        with patch("app.routers.chat.set_cached_response") as mock_set_cache:
            with patch("app.routers.chat.retrieve", return_value=mock_retrieve_murabaha):
                with patch("app.routers.chat.OpenAI") as MockOpenAI:
                    MockOpenAI.return_value.chat.completions.create.return_value = make_llm_mock(new_answer)
                    
                    resp = client.post(
                        "/chat",
                        json={"session_id": "test-cache", "message": "Tell me about Murabaha again"},
                    )

                    assert resp.status_code == 200
                    data = resp.json()
                    assert data["response"] == new_answer
                    assert data["cache_hit"] is False
                    assert len(data["retrieved_chunk_ids"]) > 0
                    
                    # Ensure the LLM was called
                    MockOpenAI.return_value.chat.completions.create.assert_called_once()
                    
                    # Ensure we tried to save to cache
                    mock_set_cache.assert_called_once()
