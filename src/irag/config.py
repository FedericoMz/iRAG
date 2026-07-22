import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / "config.env")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", str(ROOT / "experiment data")))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", str(ROOT / "outputs")))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b")
    model_provider: str = os.getenv("MODEL_PROVIDER", "ollama")
    ollama_generation_model: str = os.getenv(
        "OLLAMA_GENERATION_MODEL", os.getenv("GENERATION_MODEL", "qwen3.5:9b")
    )
    ollama_auxiliary_model: str = os.getenv(
        "OLLAMA_AUXILIARY_MODEL", os.getenv("AUXILIARY_MODEL", "qwen3.5:4b")
    )
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    openrouter_generation_model: str = os.getenv(
        "OPENROUTER_GENERATION_MODEL", "openai/gpt-5.2"
    )
    openrouter_auxiliary_model: str = os.getenv(
        "OPENROUTER_AUXILIARY_MODEL", "openai/gpt-5.2"
    )
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    openrouter_http_referer: str | None = os.getenv("OPENROUTER_HTTP_REFERER")
    openrouter_app_title: str = os.getenv("OPENROUTER_APP_TITLE", "iRAG experiments")
    bedrock_region: str = os.getenv(
        "BEDROCK_REGION",
        os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "eu-west-1")),
    )
    bedrock_profile: str | None = os.getenv("BEDROCK_PROFILE") or None
    bedrock_generation_model: str = os.getenv(
        "BEDROCK_GENERATION_MODEL",
        "eu.amazon.nova-2-lite-v1:0",
    )
    bedrock_auxiliary_model: str = os.getenv(
        "BEDROCK_AUXILIARY_MODEL",
        "eu.amazon.nova-2-lite-v1:0",
    )
    model_timeout: int = int(
        os.getenv("MODEL_TIMEOUT", os.getenv("OLLAMA_TIMEOUT", "300"))
    )
    model_retries: int = int(
        os.getenv("MODEL_RETRIES", os.getenv("OLLAMA_RETRIES", "3"))
    )
    bedrock_retries: int = int(os.getenv("BEDROCK_RETRIES", "10"))
    bedrock_throttle_retries: int = int(
        os.getenv("BEDROCK_THROTTLE_RETRIES", "100")
    )
    bedrock_throttle_max_delay: float = float(
        os.getenv("BEDROCK_THROTTLE_MAX_DELAY", "60")
    )


settings = Settings()
