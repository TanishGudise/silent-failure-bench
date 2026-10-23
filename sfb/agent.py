"""Agent loop, traced with the uselemma-tracing SDK and captured locally.

The SDK's `transport=` hook receives the exact ingest payload, so traces are in
the vendor's wire format without any account or network call. Generation spans
follow the vendor's direct-SDK example: the full prompt as `input` and
`llm_input_messages`, the visible reply as `output`. Thinking blocks stay in
the conversation (the API needs them back) but are not written to the trace.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from uselemma_tracing import Lemma

from .env import Env
from .labels import Labels, label
from .tasks import Task

CAPTURE_PROJECT_ID = "00000000-0000-4000-8000-000000000001"
AGENT_NAME = "support-desk-agent"

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "find_customer": {"type": "object", "properties": {"email": {"type": "string"}},
                      "required": ["email"]},
    "list_reservations": {"type": "object", "properties": {"customer_id": {"type": "string"}},
                          "required": ["customer_id"]},
    "get_reservation": {"type": "object", "properties": {"reservation_id": {"type": "string"}},
                        "required": ["reservation_id"]},
    "change_reservation": {"type": "object", "properties": {
        "reservation_id": {"type": "string"},
        "new_date": {"type": "string", "description": "YYYY-MM-DD"}},
        "required": ["reservation_id", "new_date"]},
    "cancel_reservation": {"type": "object", "properties": {"reservation_id": {"type": "string"}},
                           "required": ["reservation_id"]},
    "issue_refund": {"type": "object", "properties": {
        "reservation_id": {"type": "string"},
        "amount": {"type": "number", "description": "Dollars"},
        "method": {"type": "string", "enum": ["original_payment", "travel_credit"]}},
        "required": ["reservation_id", "amount"]},
    "update_contact": {"type": "object", "properties": {
        "customer_id": {"type": "string"}, "phone": {"type": "string"}},
        "required": ["customer_id", "phone"]},
    "resolve_ticket": {"type": "object", "properties": {
        "summary": {"type": "string", "description": "Message shown to the customer"},
        "actions": {"type": "array", "items": {"type": "object", "properties": {
            "type": {"type": "string", "enum": ["cancel", "change_date", "refund", "update_contact"]},
            "reservation_id": {"type": "string", "description": "Required for cancel, change_date and refund"},
            "status": {"type": "string", "enum": ["done", "pending", "failed"]},
            "detail": {"type": "string"}},
            "required": ["type", "status"], "additionalProperties": False}}},
        "required": ["summary", "actions"], "additionalProperties": False},
}
# resolve_ticket is the report of record, so its enum is enforced by the API.
STRICT_TOOLS = {"resolve_ticket"}

# Stop reasons that end an episode as aborted rather than as agent behavior.
ABORT_STOPS = {"max_tokens", "refusal", "model_context_window_exceeded", "pause_turn"}

BASELINE_SYSTEM_PROMPT = """You are the support agent for Skyway Travel. You help customers with their flight reservations using the tools provided.

- Identify the customer with find_customer, using the email they give.
- Act only on what the customer asked for. Don't issue a refund unless the customer asks for one.
- Act on clear requests without asking the customer to confirm.
- When you're done, call resolve_ticket with a short summary for the customer and one entry per requested action, with status done, pending, or failed."""

BASELINE_TOOL_DESCRIPTIONS = {
    "find_customer": "Look up a customer by email. Returns customer_id and name.",
    "list_reservations": "List a customer's reservations: id, route, departure date and status.",
    "get_reservation": "Get full details for one reservation, including fare class, price and fare rules.",
    "change_reservation": "Move an active reservation to a new departure date.",
    "cancel_reservation": "Cancel an active reservation.",
    "issue_refund": "Refund a cancelled, refundable reservation. method is original_payment (the default) or travel_credit.",
    "update_contact": "Update a customer's phone number.",
    "resolve_ticket": "Close the ticket. summary is shown to the customer; actions lists each requested action with its status.",
}


@dataclass(frozen=True)
class AgentConfig:
    system_prompt: str = BASELINE_SYSTEM_PROMPT
    tool_descriptions: tuple[tuple[str, str], ...] = tuple(sorted(BASELINE_TOOL_DESCRIPTIONS.items()))

    def tools(self) -> list[dict[str, Any]]:
        descriptions = dict(self.tool_descriptions)
        return [{"name": name, "description": descriptions[name], "input_schema": schema,
                 **({"strict": True} if name in STRICT_TOOLS else {})}
                for name, schema in TOOL_SCHEMAS.items()]

    def with_edits(self, system_prompt: str | None = None,
                   tool_descriptions: dict[str, str] | None = None) -> "AgentConfig":
        descriptions = dict(self.tool_descriptions)
        for name, text in (tool_descriptions or {}).items():
            if name in descriptions and isinstance(text, str) and text.strip():
                descriptions[name] = text.strip()
        return AgentConfig(system_prompt=(system_prompt or self.system_prompt).strip(),
                           tool_descriptions=tuple(sorted(descriptions.items())))

    @property
    def fingerprint(self) -> str:
        blob = json.dumps({"system": self.system_prompt, "tools": self.tool_descriptions})
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


@dataclass
class ModelResponse:
    content: list[dict[str, Any]]
    usage: dict[str, Any]
    stop_reason: str
    model: str


class ModelClient(Protocol):
    def create(self, *, system: str, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]], max_tokens: int) -> ModelResponse: ...


@dataclass
class Episode:
    task_key: str
    family: str
    seed: int
    induced: bool
    config: str
    labels: Labels
    trace: dict[str, Any]
    env: Env
    model_calls: int
    usage: dict[str, int] = field(default_factory=dict)
    final_text: str = ""
    stop_reasons: list[str] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        return {
            "task": self.task_key, "family": self.family, "seed": self.seed,
            "induced": self.induced, "config": self.config,
            "label": self.labels.primary, "labels": sorted(self.labels.all),
            "off_enum": self.labels.details.get("off_enum", []),
            "model_calls": self.model_calls, "usage": self.usage, "stop_reasons": self.stop_reasons,
            "claim": self.env.claim, "calls": [c.__dict__ for c in self.env.calls],
            "final_world": self.env.world.snapshot(),
        }


def _visible(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [b for b in content if b.get("type") in ("text", "tool_use", "tool_result")]


def _prompt_view(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    view = [{"role": "system", "content": system}]
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            content = _visible(content)
        view.append({"role": message["role"], "content": content})
    return view


def _capture(sink: list[dict[str, Any]]):
    def transport(_url: str, _headers: dict[str, str], body: bytes):
        sink.append(json.loads(body.decode()))
        return 201, "{}"
    return transport


def run_episode(task: Task, config: AgentConfig, model: ModelClient, *,
                max_model_calls: int = 10, max_tokens: int = 4096) -> Episode:
    env = task.make_env()
    captured: list[dict[str, Any]] = []
    lemma = Lemma(api_key="capture-only", project_id=CAPTURE_PROJECT_ID,
                  transport=_capture(captured))
    messages = copy.deepcopy(task.messages)
    tools = config.tools()
    state: dict[str, Any] = {"calls": 0, "final_text": "", "aborted": None, "stops": [],
                             "usage": {"input_tokens": 0, "output_tokens": 0}}

    def run(trace):
        for _ in range(max_model_calls):
            prompt = _prompt_view(config.system_prompt, messages)
            started = time.perf_counter()
            response = model.create(system=config.system_prompt, messages=messages,
                                    tools=tools, max_tokens=max_tokens)
            state["calls"] += 1
            state["stops"].append(response.stop_reason)
            for key in state["usage"]:
                state["usage"][key] += int(response.usage.get(key) or 0)
            visible = _visible(response.content)
            texts = [b["text"] for b in visible if b["type"] == "text" and b.get("text")]
            tool_uses = [b for b in visible if b["type"] == "tool_use"]
            trace.record_generation(
                name="assistant",
                model=response.model,
                input=prompt,
                output="\n".join(texts) if texts and not tool_uses else visible,
                duration_ms=int((time.perf_counter() - started) * 1000),
                llm_input_messages=prompt,
                llm_provider="anthropic",
                usage={k: response.usage[k] for k in ("input_tokens", "output_tokens")
                       if isinstance(response.usage.get(k), int)},
            )
            messages.append({"role": "assistant", "content": response.content})
            if texts:
                state["final_text"] = texts[-1]
            if response.stop_reason in ABORT_STOPS:
                # A cut-off or declined turn says nothing about what the agent
                # would have done; don't run a possibly truncated tool call.
                state["aborted"] = response.stop_reason
                return
            if not tool_uses:
                return
            results = []
            for block in tool_uses:
                tool_started = time.perf_counter()
                output = env.call(block["name"], block.get("input") or {})
                trace.record_tool(
                    name=block["name"],
                    input=block.get("input") or {},
                    output=output,
                    duration_ms=int((time.perf_counter() - tool_started) * 1000),
                )
                result = {"type": "tool_result", "tool_use_id": block["id"],
                          "content": json.dumps(output)}
                if "error" in output:
                    result["is_error"] = True
                results.append(result)
            messages.append({"role": "user", "content": results})
            if env.resolved:
                return
        state["aborted"] = "model_call_cap"

    def traced(trace):
        run(trace)
        summary = env.claim["summary"] if env.claim else state["final_text"]
        trace.output(summary or "")
        return summary

    last_user = next(m["content"] for m in reversed(task.messages) if m["role"] == "user")
    lemma.trace(AGENT_NAME, traced, input=last_user, thread_id=task.opaque_id)
    if len(captured) != 1:
        raise RuntimeError(f"expected one captured trace, got {len(captured)}")
    return Episode(
        task_key=task.key, family=task.family, seed=task.seed, induced=task.induced,
        config=config.fingerprint, labels=label(task, env, aborted=state["aborted"]),
        trace=captured[0], env=env, model_calls=state["calls"], usage=state["usage"],
        final_text=state["final_text"], stop_reasons=state["stops"],
    )
