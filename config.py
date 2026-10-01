import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if available
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

class Config:
    """Application Configuration Settings."""
    
    # Gemini API Credentials
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # AI Models
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemini-3-flash-preview")
    REASONING_MODEL: str = os.getenv("REASONING_MODEL", "gemini-2.5-pro")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    
    # RAG Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "600"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    MAX_RAG_RESULTS: int = int(os.getenv("MAX_RAG_RESULTS", "4"))
    
    # Network / Server Settings
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / "data_store"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate critical configuration."""
        if not cls.GEMINI_API_KEY:
            print("⚠️ WARNING: GEMINI_API_KEY is not set in environment or .env file.")
            return False
        return True

# Ensure data directory exists
Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
