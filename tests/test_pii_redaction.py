"""
Eval Test 2 — PII Detection & Redaction Correctness
Tests that Emirates IDs, account numbers, email addresses, UAE phone numbers,
and titled names are all detected and masked before reaching the LLM.
"""
import pytest
from unittest.mock import patch
from tests.conftest import make_llm_mock
from app.pii.redactor import redact


# ── Unit tests for redactor (no HTTP) ───────────────────────────────────────

class TestRedactorUnit:
    def test_emirates_id_redacted(self):
        text = "My Emirates ID is 784-1990-1234567-1 please verify."
        result = redact(text)
        assert "784-1990-1234567-1" not in result.redacted_text
        assert "<EMIRATES_ID_1>" in result.redacted_text
        assert "EMIRATES_ID" in result.detected_types

    def test_emirates_id_without_dashes_redacted(self):
        text = "Emirates ID: 784199012345671"
        result = redact(text)
        assert "784199012345671" not in result.redacted_text
        assert "EMIRATES_ID" in result.detected_types

    def test_account_number_redacted(self):
        text = "My account number is 1234567890 and I need a statement."
        result = redact(text)
        assert "1234567890" not in result.redacted_text
        assert "<ACCOUNT_NUMBER_1>" in result.redacted_text
        assert "ACCOUNT_NUMBER" in result.detected_types

    def test_email_redacted(self):
        text = "Contact me at ahmed.ali@example.ae for more info."
        result = redact(text)
        assert "ahmed.ali@example.ae" not in result.redacted_text
        assert "<EMAIL_ADDRESS_1>" in result.redacted_text
        assert "EMAIL_ADDRESS" in result.detected_types

    def test_uae_phone_redacted(self):
        text = "Call me on +971501234567 about my Murabaha."
        result = redact(text)
        assert "+971501234567" not in result.redacted_text
        assert "<UAE_PHONE_1>" in result.redacted_text
        assert "UAE_PHONE" in result.detected_types

    def test_titled_name_redacted(self):
        text = "Dr. Mohammed Al-Rashid wants to open a Mudaraba account."
        result = redact(text)
        assert "Mohammed" not in result.redacted_text
        assert "<PERSON_1>" in result.redacted_text
        assert "PERSON" in result.detected_types

    def test_no_false_positives_for_normal_text(self):
        text = "What is the profit rate for Murabaha financing at the bank?"
        result = redact(text)
        assert result.redacted_text == text
        assert result.detected_types == []

    def test_multiple_pii_types_in_one_message(self):
        text = "I am Mr. Ahmed Hassan, ID 784-1985-9876543-2, account 9876543210."
        result = redact(text)
        assert "784-1985-9876543-2" not in result.redacted_text
        assert "9876543210" not in result.redacted_text
        assert "EMIRATES_ID" in result.detected_types
        assert "ACCOUNT_NUMBER" in result.detected_types


# ── Integration test: PII flagged in trace via API ───────────────────────────

def test_pii_flagged_in_api_response(client, mock_retrieve_murabaha):
    """
    End-to-end: send a message with Emirates ID; verify API reports PII detected
    and the response does not echo back the raw ID.
    """
    with patch("app.routers.chat.retrieve", return_value=mock_retrieve_murabaha):
        with patch("app.routers.chat.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = make_llm_mock(
                "Your Murabaha account details are being processed."
            )

            resp = client.post(
                "/chat",
                json={
                    "session_id": "test-pii-1",
                    "message": "My Emirates ID is 784-1990-1234567-1, what is my Murabaha balance?",
                },
            )

    assert resp.status_code == 200
    data = resp.json()

    # API should report PII was detected
    assert "EMIRATES_ID" in data["pii_detected"], (
        "EMIRATES_ID should be listed in pii_detected"
    )

    # Raw ID must not appear in the response
    assert "784-1990-1234567-1" not in data["response"], (
        "Raw Emirates ID must not appear in LLM response"
    )
