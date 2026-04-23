from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelRoute:
    provider: str
    model: str


class ModelGateway:
    def __init__(self, primary_model: str, fallback_model: str) -> None:
        self.primary_model = primary_model
        self.fallback_model = fallback_model

    def route(self, task_type: str) -> ModelRoute:
        if task_type in {"qa", "edit"}:
            return ModelRoute(provider="local", model=self.primary_model)
        return ModelRoute(provider="local", model=self.fallback_model)

    def complete(self, prompt: str, task_type: str = "qa") -> str:
        route = self.route(task_type)
        return f"[{route.model}] {prompt}"
