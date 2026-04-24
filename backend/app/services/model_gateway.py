from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import litellm

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelRoute:
    provider: str
    model: str


class ModelGateway:
    def __init__(
        self,
        primary_model: str,
        fallback_model: str,
        allowed_models: list[str] | None = None,
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None,
        gemini_api_key: str | None = None,
    ) -> None:
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.allowed_models = allowed_models or [primary_model, fallback_model]

        # Configure provider keys for litellm if values are provided from settings.
        if anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if gemini_api_key:
            os.environ["GEMINI_API_KEY"] = gemini_api_key
            # Some Google SDK paths use GOOGLE_API_KEY.
            os.environ.setdefault("GOOGLE_API_KEY", gemini_api_key)

    def route(self, task_type: str) -> ModelRoute:
        if task_type in {"qa", "edit"}:
            return ModelRoute(provider="local", model=self.primary_model)
        return ModelRoute(provider="local", model=self.fallback_model)

    def get_allowed_models(self) -> list[str]:
        """Return list of allowed model identifiers for frontend selection."""
        return self.allowed_models

    def validate_model(self, model: str) -> bool:
        """Check if requested model is in allowed list."""
        return model in self.allowed_models

    def _build_model_candidates(self, requested_model: str | None) -> list[str]:
        """Build ordered unique model candidates for robust fallback behavior."""
        candidates: list[str] = []

        if requested_model and self.validate_model(requested_model):
            candidates.append(requested_model)

        # Keep backward compatibility with explicit primary/fallback settings.
        if self.primary_model in self.allowed_models:
            candidates.append(self.primary_model)
        if self.fallback_model in self.allowed_models:
            candidates.append(self.fallback_model)

        # Add remaining allowed models to broaden fallback coverage.
        candidates.extend(self.allowed_models)

        # Deduplicate while preserving order.
        seen: set[str] = set()
        ordered_unique: list[str] = []
        for model in candidates:
            if model not in seen:
                seen.add(model)
                ordered_unique.append(model)
        return ordered_unique

    def complete(self, prompt: str, task_type: str = "qa", requested_model: str | None = None) -> tuple[str, str, bool]:
        """Complete a prompt with litellm, respecting requested_model if provided.

        Implements fallback chain across ordered model candidates.

        Returns: (completion_text, model_used, fallback_used)
        """
        candidates = self._build_model_candidates(requested_model)
        first_model = candidates[0] if candidates else self.primary_model
        last_error: Exception | None = None

        for idx, model in enumerate(candidates):
            try:
                response = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0.7,
                )
                completion_text = response.choices[0].message.content or ""
                fallback_used = idx > 0
                return completion_text, model, fallback_used
            except Exception as e:
                last_error = e
                if idx == 0:
                    logger.warning(f"Primary model {model} failed: {e}. Attempting fallback chain...")
                else:
                    logger.warning(f"Fallback model {model} failed: {e}")

        logger.error(f"All model candidates failed. Last error: {last_error}")
        error_msg = (
            f"Failed to get response from any configured model. "
            f"Tried: {', '.join(candidates)}. Last error: {last_error}"
        )
        return error_msg, first_model, True
