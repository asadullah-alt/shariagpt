from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv

# Explicitly load .env file so the environment variables are populated
load_dotenv()


class Settings(BaseSettings):
    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o-mini"

    # Qdrant Cloud
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "sharia_knowledge"

    # Redis (optional — falls back to in-memory if not set)
    redis_url: str = ""

    # Upstash Redis REST (for semantic caching)
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""

    # LlamaCloud (for PDF → Markdown parsing via admin upload)
    llama_cloud_api_key: str = ""

    # Admin endpoint protection
    admin_api_key: str = "change-me-in-production"

    # Cloudinary configuration
    cloudinary_url: str = ""

    # App
    app_version: str = "1.0.0"
    log_dir: str = "logs"
    top_k_chunks: int = 5
    session_ttl_seconds: int = 86400  # 24 hours

    # JWT Auth
    jwt_secret: str = "change-me-shariagpt-jwt-secret-2024"
    jwt_expire_minutes: int = 1440  # 24 hours

    # LangSmith Observability
    langchain_tracing_v2: str = "false"
    langchain_api_key: str = ""
    langchain_project: str = "shariagpt"

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
