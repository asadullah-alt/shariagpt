"""
User Store — Qdrant-backed user CRUD
─────────────────────────────────────
Stores user accounts in a dedicated Qdrant collection 'sharia_users'.
Each user is a point with a dummy zero-vector and all data in the payload.
Lookup is done via scroll with payload filter on email.
"""
import uuid
import time
from typing import Optional
import bcrypt
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue,
)

from app.rag.vector_store import get_client
from app.rag.embedder import VECTOR_DIM

USERS_COLLECTION = "sharia_users"



def _ensure_users_collection():
    client = get_client()
    try:
        names = [c.name for c in client.get_collections().collections]
        if USERS_COLLECTION not in names:
            client.create_collection(
                collection_name=USERS_COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            # Create payload index on email for fast lookups
            client.create_payload_index(
                collection_name=USERS_COLLECTION,
                field_name="email",
                field_schema="keyword",
            )
            print(f"[UserStore] Created collection '{USERS_COLLECTION}'")
    except Exception as e:
        print(f"[UserStore] Collection check error: {e}")


def find_user_by_email(email: str) -> Optional[dict]:
    """Look up a user by email. Returns the payload dict or None."""
    _ensure_users_collection()
    client = get_client()
    try:
        results, _ = client.scroll(
            collection_name=USERS_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="email", match=MatchValue(value=email.lower()))]
            ),
            limit=1,
            with_payload=True,
        )
        if results:
            payload = results[0].payload
            payload["_point_id"] = results[0].id
            return payload
        return None
    except Exception as e:
        print(f"[UserStore] Lookup error: {e}")
        return None


def create_user(
    email: str,
    password: str,
    name: str,
    emirates_id: str,
    account_number: str,
    account_type: str,
    balance: str,
    totp_secret: str,
) -> dict:
    """Create a new user in Qdrant. Returns the user payload."""
    _ensure_users_collection()
    client = get_client()

    # Check uniqueness
    existing = find_user_by_email(email)
    if existing:
        raise ValueError(f"User with email '{email}' already exists.")

    user_payload = {
        "email": email.lower(),
        "password_hash": bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
        "name": name,
        "emirates_id": emirates_id,
        "account_number": account_number,
        "account_type": account_type,
        "balance": balance,
        "totp_secret": totp_secret,
        "is_2fa_enabled": True,
        "consent_version": "v1.0",
        "created_at": time.time(),
    }

    # Use a dummy zero-vector since we only need payload storage
    dummy_vector = [0.0] * VECTOR_DIM
    point_id = str(uuid.uuid4())

    client.upsert(
        collection_name=USERS_COLLECTION,
        points=[
            PointStruct(id=point_id, vector=dummy_vector, payload=user_payload)
        ],
        wait=True,
    )

    user_payload["_point_id"] = point_id
    return user_payload


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def update_user_field(email: str, field: str, value) -> bool:
    """Update a single field on a user's payload."""
    _ensure_users_collection()
    client = get_client()
    user = find_user_by_email(email)
    if not user:
        return False
    try:
        client.set_payload(
            collection_name=USERS_COLLECTION,
            payload={field: value},
            points=[user["_point_id"]],
        )
        return True
    except Exception as e:
        print(f"[UserStore] Update error: {e}")
        return False


def delete_user(email: str) -> bool:
    """Delete a user from Qdrant by email."""
    _ensure_users_collection()
    client = get_client()
    user = find_user_by_email(email)
    if not user:
        return False
    try:
        client.delete(
            collection_name=USERS_COLLECTION,
            points_selector=[user["_point_id"]],
        )
        return True
    except Exception as e:
        print(f"[UserStore] Delete error: {e}")
        return False
