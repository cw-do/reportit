"""Application settings — loads LLM/OpenRouter config from .env.

Adapted from eqsanstools-cli config/settings.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "openai/gpt-5-mini"
FALLBACK_MODEL = "google/gemini-3-flash-preview"


@dataclass
class LLMSettings:
    """LLM configuration for OpenRouter (OpenAI-compatible API)."""

    api_key: str = ""
    model: str = DEFAULT_MODEL
    fallback_model: str = FALLBACK_MODEL
    base_url: str = "https://openrouter.ai/api/v1"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass
class AppSettings:
    """Top-level application settings."""

    llm: LLMSettings = field(default_factory=LLMSettings)
    # Maximum agentic strategy steps (LLM tool-calling turns).
    max_llm_steps: int = 40

    @classmethod
    def load(cls) -> "AppSettings":
        """Load settings from .env file and environment variables."""
        try:
            from dotenv import load_dotenv

            env_locations = [
                Path.cwd() / ".env",
                Path(__file__).resolve().parent.parent.parent / ".env",  # repo root
                Path.home() / ".reportit" / ".env",
            ]
            for env_path in env_locations:
                if env_path.is_file():
                    load_dotenv(env_path)
                    break
        except ImportError:
            pass

        settings = cls()
        settings.llm = LLMSettings(
            # tolerate trailing space in key name ("OPENROUTER_API_KEY ")
            api_key=(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY ") or "").strip(),
            model=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL).strip(),
            fallback_model=os.getenv("OPENROUTER_FALLBACK_MODEL", FALLBACK_MODEL).strip(),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip(),
        )
        try:
            settings.max_llm_steps = int(os.getenv("REPORTIT_MAX_LLM_STEPS", "40"))
        except ValueError:
            settings.max_llm_steps = 30
        return settings
