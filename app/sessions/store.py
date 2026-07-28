"""
Session store with Upstash Redis backend and in-memory fallback.
Conversation history persists across restarts when Redis is configured.
"""
import json
import time
from typing import Optional

from app.config import get_settings

# In-memory fallbacks
_memory: dict[str, list[dict]] = {}
_user_chats: dict[str, dict[str, dict]] = {}

try:
    import upstash_redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class SessionStore:
    def __init__(self) -> None:
        self._r: Optional[object] = None
        s = get_settings()
        self._ttl = s.session_ttl_seconds

        if _REDIS_AVAILABLE and s.upstash_redis_rest_url and s.upstash_redis_rest_token:
            try:
                from upstash_redis import Redis
                self._r = Redis(url=s.upstash_redis_rest_url, token=s.upstash_redis_rest_token)
                self._r.ping()
                print("[SessionStore] Connected to Upstash Redis ✓")
            except Exception as exc:
                print(f"[SessionStore] Upstash Redis unavailable ({exc}), using in-memory fallback")
                self._r = None

    def _key(self, session_id: str) -> str:
        return f"shariagpt:session:{session_id}:history"
        
    def _user_chats_key(self, user_id: str) -> str:
        return f"shariagpt:user:{user_id}:chats"

    def get_history(self, session_id: str) -> list[dict]:
        if self._r:
            try:
                raw = self._r.get(self._key(session_id))
                return json.loads(raw) if raw else []
            except Exception:
                return []
        return list(_memory.get(session_id, []))

    def get_user_chats(self, user_id: str) -> list[dict]:
        if self._r:
            try:
                chats = self._r.hgetall(self._user_chats_key(user_id))
                if not chats:
                    return []
                # Handle Upstash REST hgetall return format which is often a dict or flat list
                parsed = []
                if isinstance(chats, dict):
                    for k, v in chats.items():
                        c = json.loads(v)
                        c["session_id"] = k
                        parsed.append(c)
                elif isinstance(chats, list):
                    # Some clients return [k1, v1, k2, v2]
                    for i in range(0, len(chats), 2):
                        k = chats[i]
                        v = chats[i+1]
                        c = json.loads(v)
                        c["session_id"] = k
                        parsed.append(c)
                return sorted(parsed, key=lambda x: x.get("timestamp", 0), reverse=True)
            except Exception as e:
                print(f"Error fetching user chats: {e}")
                return []
        
        # In-memory fallback
        user_c = _user_chats.get(user_id, {})
        parsed = []
        for k, v in user_c.items():
            c = dict(v)
            c["session_id"] = k
            parsed.append(c)
        return sorted(parsed, key=lambda x: x.get("timestamp", 0), reverse=True)

    def append_turn(self, session_id: str, user_msg: str, assistant_msg: str, user_id: Optional[str] = None) -> None:
        history = self.get_history(session_id)
        is_first_turn = len(history) == 0
        
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})
        history = history[-40:]  # keep last 20 turns (40 messages)
        
        if self._r:
            try:
                self._r.setex(self._key(session_id), self._ttl, json.dumps(history))
                if is_first_turn and user_id and user_id != "guest":
                    title = user_msg[:40] + "..." if len(user_msg) > 40 else user_msg
                    chat_meta = json.dumps({"title": title, "timestamp": time.time()})
                    self._r.hset(self._user_chats_key(user_id), values={session_id: chat_meta})
            except Exception as e:
                print(f"[SessionStore] Failed to save to redis: {e}")
        else:
            _memory[session_id] = history
            if is_first_turn and user_id and user_id != "guest":
                if user_id not in _user_chats:
                    _user_chats[user_id] = {}
                title = user_msg[:40] + "..." if len(user_msg) > 40 else user_msg
                _user_chats[user_id][session_id] = {"title": title, "timestamp": time.time()}

    def delete_user_data(self, user_id: str) -> bool:
        if self._r:
            try:
                chats = self._r.hgetall(self._user_chats_key(user_id))
                if isinstance(chats, dict):
                    session_ids = list(chats.keys())
                elif isinstance(chats, list):
                    session_ids = [chats[i] for i in range(0, len(chats), 2)]
                else:
                    session_ids = []
                
                for sid in session_ids:
                    self._r.delete(self._key(sid))
                self._r.delete(self._user_chats_key(user_id))
                return True
            except Exception as e:
                print(f"[SessionStore] Failed to delete user data: {e}")
                return False
        else:
            if user_id in _user_chats:
                session_ids = list(_user_chats[user_id].keys())
                for sid in session_ids:
                    _memory.pop(sid, None)
                _user_chats.pop(user_id, None)
            return True

    def clear(self, session_id: str) -> None:
        if self._r:
            try:
                self._r.delete(self._key(session_id))
            except Exception:
                pass
        else:
            _memory.pop(session_id, None)

    def active_sessions(self) -> int:
        if self._r:
            try:
                keys = self._r.keys("shariagpt:session:*:history")
                return len(keys) if keys else 0
            except Exception:
                return 0
        return len(_memory)


_instance: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    global _instance
    if _instance is None:
        _instance = SessionStore()
    return _instance
