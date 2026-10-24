from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    allowed_origins: str = "http://localhost:3000"
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    # Key pool — comma-separated; read directly by LLMClient via os.getenv
    llm_api_keys: str = ""
    llm_model: str = "openai/gpt-oss-20b"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    upload_dir: str = "./data/uploads"
    max_file_size_mb: int = 5
    investigations_enabled: bool = False
    legacy_execution_enabled: bool = False
    dataset_bucket: str = "lumiq-datasets"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_rpm: int = 2
    gemini_rpd: int = 20
    gemini_tpm: int = 12000
    groq_rpm: int = 5
    groq_rpd: int = 50
    groq_tpm: int = 12000
    execution_timeout_seconds: int = 10
    max_retry_count: int = 2
    # Qdrant Cloud — replaces ChromaDB local storage
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    cohere_api_key: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
