"""One reusable LM Studio interface; no connections are made during import."""
from __future__ import annotations

import os
from typing import Protocol, Literal
from pydantic import BaseModel, Field, field_validator
from .schema import StrictModel
from .reasoning import ReasoningPolicy, canonical_role, InferenceError

API_TIMEOUT_SECONDS = 3000


class ModelConfig(StrictModel):
    model: str = Field(default_factory=lambda: os.getenv("LMSTUDIO_MODEL", "google/gemma-4-e4b"), min_length=1)
    planner_model: str | None = None
    ux_model: str | None = Field(default=None, min_length=1)
    architect_model: str | None = Field(default=None, min_length=1)
    coordinator_model: str | None = Field(default=None, min_length=1)
    qa_model: str | None = Field(default=None, min_length=1)
    review_max_tokens: int | None = Field(default=None, ge=1)
    base_url: str = Field(default_factory=lambda: os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"))
    temperature: float = Field(default=0.4, ge=0, le=2)
    top_p: float = Field(default=0.95, gt=0, le=1)
    seed: int | None = 1024
    planner_max_tokens: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    timeout_seconds: float = Field(default=API_TIMEOUT_SECONDS, gt=0)
    reasoning_policy: ReasoningPolicy | None = Field(default=None, exclude_if=lambda value: value is None)
    inference_role: str | None = Field(default=None, exclude_if=lambda value: value is None)
    inference_artifact_dir: str | None = Field(default=None, exclude=True)
    # Omitted fields preserve historical plans and their one-pass behavior.
    manual_verification: Literal['on', 'off'] | None = Field(default=None, exclude_if=lambda v: v is None)
    max_repair_attempts: int | None = Field(default=None, ge=0, le=10, strict=True, exclude_if=lambda v: v is None)

    @field_validator('inference_role')
    @classmethod
    def valid_role(cls, value):
        return canonical_role(value) if value is not None else None

    def model_for(self, role: str) -> str:
        role = canonical_role(role)
        role = 'architect' if role == 'architecture' else 'planner' if role in {'intake', 'decomposition'} else role
        return self.model if role == "implementer" else getattr(self, role + "_model") or self.model

    def for_role(self, role: str):
        canonical = canonical_role(role)
        role = 'architect' if canonical == 'architecture' else canonical
        if role in {"planner", "implementer", "intake", "decomposition"}:
            return self.model_copy(update={'inference_role': canonical}), role != "implementer"
        offset = {"ux": 2, "architect": 3, "coordinator": 4, "qa": 5}[role]
        return self.model_copy(update={"model": self.model_for(role), "planner_model": None,
                                      "seed": self.seed + offset - 1 if self.seed is not None else None,
                                      'inference_role': canonical, "max_tokens": self.review_max_tokens}), False


class ModelClient(Protocol):
    def complete(self, system: str, user: str, config: ModelConfig, *, planner: bool = False,
                 response_schema: type[BaseModel] | None = None) -> tuple[str, dict]: ...
    def models(self, config: ModelConfig) -> list[str]: ...


def json_schema_response_format(schema: type[BaseModel]) -> dict:
    """Schema name is protocol metadata, never a wrapper around response data."""
    return {"type": "json_schema", "json_schema": {
        "name": schema.__name__, "strict": True, "schema": schema.model_json_schema(),
    }}


class StructuredOutputRequestError(InferenceError):
    pass


class LMStudioClient:
    def _client(self, config):
        from openai import OpenAI
        return OpenAI(base_url=config.base_url, api_key="lm-studio", timeout=config.timeout_seconds, max_retries=0)

    def models(self, config):
        with self._client(config) as client:
            return [item.id for item in client.models.list().data]

    def _async_client(self, config):
        from openai import AsyncOpenAI
        return AsyncOpenAI(base_url=config.base_url, api_key="lm-studio",
                           timeout=config.timeout_seconds, max_retries=0)

    def complete(self, system, user, config, *, planner=False, response_schema=None):
        from .inference import complete
        return complete(self, system, user, config, planner, response_schema)
