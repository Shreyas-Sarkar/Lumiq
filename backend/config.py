from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    # Key pool — comma-separated; read directly by LLMClient via os.getenv
    llm_api_keys: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    upload_dir: str = "./data/uploads"
    max_file_size_mb: int = 50
    execution_timeout_seconds: int = 10
    max_retry_count: int = 2
    # Qdrant Cloud — replaces ChromaDB local storage
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
