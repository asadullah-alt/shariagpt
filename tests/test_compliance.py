import os
import json
from pathlib import Path

def test_security_headers_present(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "Strict-Transport-Security" in response.headers
    assert "X-Content-Type-Options" in response.headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "X-Frame-Options" in response.headers
    assert "Content-Security-Policy" in response.headers

def test_audit_logging(client):
    log_file = Path("logs/audit.jsonl")
    initial_count = 0
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            initial_count = len(f.readlines())
            
    # Trigger a request that will be logged
    response = client.get("/health")
    assert response.status_code == 200
    
    assert log_file.exists()
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) > initial_count
        
        # Check the most recent log
        last_log = json.loads(lines[-1])
        assert last_log["type"] == "http_request"
        assert "/health" in last_log["url"]
        assert last_log["status_code"] == 200
        assert "latency_ms" in last_log
        assert "user_id" in last_log
