"""Anthropic client for the agent, judges and fixer. All calls go to the local
gateway (sfb.gateway), which holds the key, meters spend and enforces caps."""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic

from .agent import ModelResponse

AGENT_MODEL = "claude-haiku-5-5"


def _manifest_hash() -> str:
    from .manifest import compute

    return compute()["manifest_hash"][:16]


class AnthropicModel:
    def __init__(self, base_url: str, *, model: str = AGENT_MODEL, run: str = "",
                 effort: str | None = None, timeout: float = 480.0) -> None:
        # The gateway token, not an API key: the gateway attaches the real key.
        # No SDK retries: the gateway retries unbilled errors itself and logs
        # every attempt. The timeout outlasts the gateway's upstream timeout.
        self.client = anthropic.Anthropic(
            base_url=base_url, api_key=os.environ["SFB_GATEWAY_TOKEN"], max_retries=0, timeout=timeout,
            default_headers={"x-sfb-run": run, "x-sfb-manifest": _manifest_hash()},
        )
        self.model = model
        self.effort = effort

    def _output_config(self, extra: dict[str, Any] | None = None) -> dict[str, Any] | None:
        config = dict(extra or {})
        if self.effort:
            config["effort"] = self.effort
        return config or None

    def create(self, *, system: str, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]], max_tokens: int) -> ModelResponse:
        kwargs: dict[str, Any] = dict(model=self.model, max_tokens=max_tokens, system=system,
                                      messages=messages, tools=tools)
        output_config = self._output_config()
        if output_config:
            kwargs["output_config"] = output_config
        response = self.client.messages.create(**kwargs)
        return ModelResponse(
            content=[block.model_dump(mode="json", exclude_none=True) for block in response.content],
            usage=response.usage.model_dump(mode="json"),
            stop_reason=str(response.stop_reason),
            model=response.model,
        )

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any],
                      max_tokens: int = 2048) -> tuple[dict[str, Any] | None, dict[str, Any], str]:
        """One structured-output call. Returns (answer or None, usage, stop_reason)."""
        response = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
            output_config=self._output_config({"format": {"type": "json_schema", "schema": schema}}),
        )
        usage = response.usage.model_dump(mode="json")
        if response.stop_reason in ("refusal", "max_tokens"):
            return None, usage, str(response.stop_reason)
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            return json.loads(text), usage, str(response.stop_reason)
        except json.JSONDecodeError:
            return None, usage, "unparseable"
