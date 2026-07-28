import json
import time
from pathlib import Path
from datetime import datetime, timezone
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
from app.config import get_settings

class AuditLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.audit_file = self.log_dir / "audit.jsonl"

    def log(self, event: dict):
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(event)
        with open(self.audit_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class AuditMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, logger: AuditLogger):
        super().__init__(app)
        self.logger = logger
        self.settings = get_settings()

    def _extract_user_id(self, request: Request) -> str:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                # We decode without verification just to extract the 'sub' for auditing purposes.
                # Actual validation happens in the route dependencies.
                payload = jwt.decode(token, options={"verify_signature": False})
                return payload.get("sub", "guest")
            except Exception:
                return "invalid_token"
        return "guest"

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Determine client IP (handles standard proxies)
        client_ip = request.client.host if request.client else "unknown"
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()

        user_id = self._extract_user_id(request)

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            self._log_event(request, user_id, client_ip, start_time, status_code, str(e))
            raise e

        self._log_event(request, user_id, client_ip, start_time, status_code)
        return response

    def _log_event(self, request: Request, user_id: str, client_ip: str, start_time: float, status_code: int, error: str = None):
        latency_ms = round((time.time() - start_time) * 1000, 2)
        event = {
            "type": "http_request",
            "method": request.method,
            "url": str(request.url),
            "client_ip": client_ip,
            "user_id": user_id,
            "status_code": status_code,
            "latency_ms": latency_ms,
        }
        if error:
            event["error"] = error
            
        self.logger.log(event)
