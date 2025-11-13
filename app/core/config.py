from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application configuration sourced from environment variables or defaults."""

    data_dir: Path = Path("data/pdfs")
    persist_dir: Path = Path("storage/index")
    model_name: str = "Qwen/Qwen3-8B"
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    load_in_4bit: bool = True
    device_map: Optional[str] = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.1

    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__")

    @property
    def resolved_data_dir(self) -> Path:
        """Ensure the data directory exists and return an absolute path."""
        path = self.data_dir if self.data_dir.is_absolute() else Path.cwd() / self.data_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def resolved_persist_dir(self) -> Path:
        """Ensure the index persistence directory exists and return an absolute path."""
        path = self.persist_dir if self.persist_dir.is_absolute() else Path.cwd() / self.persist_dir
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a cached instance of application settings."""
    return AppSettings()

