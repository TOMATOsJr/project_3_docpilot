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


@dataclass(slots=True)
class RoutingContext:
    task_type: str
    estimated_tokens: int


class ModelSelectionStrategy:
    """Task-aware model selection strategy used when the user does not pin a model."""

    _FAST_HINTS = ("flash-lite", "haiku", "8b-instant", "mini")
    _QUALITY_HINTS = ("1.5-pro", "sonnet", "gpt-4o", "opus")

    def __init__(self, primary_model: str, fallback_model: str, allowed_models: list[str]) -> None:
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self._allowed_models = allowed_models

    def select(self, context: RoutingContext) -> tuple[str, str]:
        task = context.task_type.lower()

        # Long-context tasks prioritize quality/larger-context models.
        if task in {"synthesis", "multi_doc", "multi_document", "generate"}:
            selected = self._pick_quality_model()
            return selected, "Selected a quality/long-context model for synthesis-style task."

        # Edit tasks escalate to a quality model when prompt size is large.
        if task == "edit" and context.estimated_tokens > 3_000:
            selected = self._pick_quality_model()
            return selected, "Selected a quality model because the edit prompt is large."

        # Routine QA and short edits default to low-latency models.
        if task in {"qa", "edit"}:
            selected = self._pick_fast_model()
            return selected, "Selected a fast model for low-latency interactive response."

        selected = self._primary_model if self._primary_model in self._allowed_models else self._allowed_models[0]
        return selected, "Selected the default primary model for this task type."

    def _pick_fast_model(self) -> str:
        for model in self._allowed_models:
            lowered = model.lower()
            if any(hint in lowered for hint in self._FAST_HINTS):
                return model
        return self._primary_model if self._primary_model in self._allowed_models else self._allowed_models[0]

    def _pick_quality_model(self) -> str:
        if self._fallback_model in self._allowed_models:
            return self._fallback_model
        for model in self._allowed_models:
            lowered = model.lower()
            if any(hint in lowered for hint in self._QUALITY_HINTS):
                return model
        return self._primary_model if self._primary_model in self._allowed_models else self._allowed_models[0]


class ModelGateway:
    def __init__(
        self,
        primary_model: str,
        fallback_model: str,
        allowed_models: list[str] | None = None,
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None,
        gemini_api_key: str | None = None,
        groq_api_key: str | None = None,
    ) -> None:
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.allowed_models = allowed_models or [primary_model, fallback_model]
        self._selection_strategy = ModelSelectionStrategy(
            primary_model=self.primary_model,
            fallback_model=self.fallback_model,
            allowed_models=self.allowed_models,
        )

        # Configure provider keys for litellm if values are provided from settings.
        if anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if gemini_api_key:
            os.environ["GEMINI_API_KEY"] = gemini_api_key
            # Some Google SDK paths use GOOGLE_API_KEY.
            os.environ.setdefault("GOOGLE_API_KEY", gemini_api_key)
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key

    def route(self, task_type: str) -> ModelRoute:
        selected = self.select_model(task_type=task_type, prompt="", requested_model=None)
        return ModelRoute(provider="local", model=selected)

    def estimate_tokens(self, prompt: str, model: str | None = None) -> int:
        selected_model = model or self.primary_model
        try:
            return int(litellm.token_counter(model=selected_model, text=prompt))
        except Exception:
            return max(1, len(prompt) // 4)

    def select_model(self, task_type: str, prompt: str, requested_model: str | None = None) -> str:
        """Select an execution model using explicit preference or runtime strategy."""
        selected, _reason = self.select_model_with_reason(task_type=task_type, prompt=prompt, requested_model=requested_model)
        return selected

    def select_model_with_reason(self, task_type: str, prompt: str, requested_model: str | None = None) -> tuple[str, str]:
        """Select an execution model and include a short human-readable reason."""
        if requested_model and self.validate_model(requested_model):
            return requested_model, "User explicitly selected this model."

        estimated_tokens = self.estimate_tokens(prompt)
        context = RoutingContext(task_type=task_type, estimated_tokens=estimated_tokens)
        selected, reason = self._selection_strategy.select(context)
        logger.info(
            "Dynamic model selection: task=%s estimated_tokens=%s selected=%s",
            task_type,
            estimated_tokens,
            selected,
        )
        return selected, reason

    def get_allowed_models(self) -> list[str]:
        """Return list of allowed model identifiers for frontend selection."""
        return self.allowed_models

    def validate_model(self, model: str) -> bool:
        """Check if requested model is in allowed list."""
        return model in self.allowed_models

    def _build_model_candidates(self, requested_model: str | None, task_type: str, prompt: str) -> list[str]:
        """Build ordered unique model candidates for robust fallback behavior."""
        candidates: list[str] = []

        selected, _reason = self.select_model_with_reason(task_type=task_type, prompt=prompt, requested_model=requested_model)
        candidates.append(selected)

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

    def complete(self, prompt: str, task_type: str = "qa", requested_model: str | None = None) -> tuple[str, str, bool, str]:
        """Complete a prompt with litellm, respecting requested_model if provided.

        Implements fallback chain across ordered model candidates.

        Returns: (completion_text, model_used, fallback_used, model_selection_reason)
        """
        selected_model, selection_reason = self.select_model_with_reason(
            task_type=task_type,
            prompt=prompt,
            requested_model=requested_model,
        )
        candidates = self._build_model_candidates(requested_model, task_type=task_type, prompt=prompt)
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
                reason = selection_reason
                if fallback_used:
                    reason = f"{selection_reason} Primary attempt used {selected_model} but fallback switched to {model}."
                return completion_text, model, fallback_used, reason
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
        return error_msg, first_model, True, f"{selection_reason} All candidate models failed in fallback chain."
