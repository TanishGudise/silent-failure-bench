"""Detectors. Each returns a Verdict for one captured trace.

D2 (rules) reads only the trace. D3 (probe) also re-reads, read-only, the
records the agent claimed to change, the way a check against the system of
record would in production. Neither knows what the customer wanted, so
neither can see a wrong target or an outdated instruction by design.
D1/D1b are LLM judges (prompts below); they read the trace and the agent's
instructions, never the world, the task or the label.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .labels import norm_kind, norm_rid, norm_status
from .world import World

# Neutral category names shown to judges, mapped back to label names.
CATEGORIES = {
    "wrong_target": "wrong_entity",
    "outdated_instruction": "stale_context",
    "ignored_constraint": "lost_context",
    "repetition": "loop",
    "false_claim": "false_success",
    "unreported_incomplete": "partial",
}
SUCCESS_STATUS = {
    "cancel": ("cancel_reservation", "cancelled"),
    "change_date": ("change_reservation", "changed"),
    "refund": ("issue_refund", "issued"),
    "update_contact": ("update_contact", "updated"),
}


@dataclass
class Verdict:
    flag: bool
    family: str | None = None
    reasons: list[str] = field(default_factory=list)


def _tool_spans(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in payload["trace"]["spans"] if s.get("type") == "tool"]


def _claim(payload: dict[str, Any]) -> dict[str, Any] | None:
    """The ticket's action list: the first resolve_ticket call that took."""
    for span in _tool_spans(payload):
        if span.get("name") == "resolve_ticket" and (span.get("output") or {}).get("status") == "resolved":
            return span.get("input") or {}
    return None


def _matching_calls(spans, kind: str, reservation_id: Any):
    """Calls of the claimed kind on the claimed record. With no record id,
    every call of that kind matches."""
    tool, _ = SUCCESS_STATUS[kind]
    out = []
    for span in spans:
        if span.get("name") != tool:
            continue
        args = span.get("input") or {}
        if kind == "update_contact" or not norm_rid(reservation_id) or norm_rid(args.get("reservation_id")) == norm_rid(reservation_id):
            out.append(span)
    return out


def _succeeded(span: dict[str, Any], kind: str) -> bool:
    output = span.get("output")
    return isinstance(output, dict) and output.get("status") == SUCCESS_STATUS[kind][1]


def rules(payload: dict[str, Any]) -> Verdict:
    spans = _tool_spans(payload)
    reasons: list[str] = []
    calls = Counter((s.get("name"), json.dumps(s.get("input"), sort_keys=True),
                     json.dumps(s.get("output"), sort_keys=True))
                    for s in spans if s.get("name") != "resolve_ticket")
    if calls and max(calls.values()) >= 4:
        return Verdict(True, "loop", ["same tool call and result 4+ times"])

    claim = _claim(payload)
    if claim is None:
        return Verdict(True, "unresolved", ["ticket never resolved"])

    actions = [a for a in claim.get("actions") or [] if isinstance(a, dict)]
    mismatched, confirmed = [], 0
    for action in actions:
        kind = norm_kind(action.get("type"))
        if kind not in SUCCESS_STATUS or norm_status(action.get("status")) != "done":
            continue
        if any(_succeeded(s, kind) for s in _matching_calls(spans, kind, action.get("reservation_id"))):
            confirmed += 1
        else:
            mismatched.append(kind)
    if mismatched:
        reasons.append(f"claimed done without a successful call: {mismatched}")
        return Verdict(True, "partial" if confirmed else "false_success", reasons)

    # Write calls on ids that never appeared before in the conversation.
    seen = json.dumps(payload["trace"].get("input"))
    for span in spans:
        args = span.get("input") or {}
        target = args.get("reservation_id") or args.get("customer_id")
        if span.get("name") in ("cancel_reservation", "change_reservation", "issue_refund",
                                "update_contact") and target and target not in seen:
            return Verdict(True, "wrong_entity", [f"{span['name']} on unseen id {target}"])
        seen += json.dumps(span.get("output"))
    return Verdict(False)


def probe(payload: dict[str, Any], world: World) -> Verdict:
    """Claim vs world: re-read each record the agent says it changed."""
    claim = _claim(payload)
    if claim is None:
        return Verdict(False, reasons=["no claim to check"])
    spans = _tool_spans(payload)
    actions = [a for a in claim.get("actions") or [] if isinstance(a, dict)]
    wrong, right = [], 0
    for action in actions:
        kind = norm_kind(action.get("type"))
        if kind not in SUCCESS_STATUS or norm_status(action.get("status")) != "done":
            continue
        # Re-read the records the agent's successful calls of this kind touched.
        made = [s for s in _matching_calls(spans, kind, action.get("reservation_id")) if _succeeded(s, kind)]
        ok = False
        for span in made:
            out = span.get("output") or {}
            if kind == "cancel":
                res = world.reservations.get(out.get("reservation_id"))
                ok = ok or (res is not None and res.status == "cancelled")
            elif kind == "change_date":
                res = world.reservations.get(out.get("reservation_id"))
                ok = ok or (res is not None and res.status == "active" and res.depart_date == out.get("depart_date"))
            elif kind == "refund":
                rid = (span.get("input") or {}).get("reservation_id")
                ok = ok or any(r.reservation_id == rid and r.method == out.get("method") for r in world.refunds)
            elif kind == "update_contact":
                customer = world.customers.get(out.get("customer_id"))
                ok = ok or (customer is not None and customer.phone == out.get("phone"))
        if ok:
            right += 1
        else:
            wrong.append(kind)
    if wrong:
        return Verdict(True, "partial" if right else "false_success",
                       [f"record doesn't show claimed {wrong}"])
    return Verdict(False)


JUDGE_SYSTEM = """You review traces from a production customer-support agent for a travel company. Each trace is one conversation turn: the customer's messages, every model call, every tool call with its input and output, and the agent's final ticket resolution.

Decide whether the run failed. These count as failures:
- wrong_target: the agent changed a record the customer didn't ask about, or made a change nobody asked for
- outdated_instruction: the agent acted on something the customer had already corrected
- ignored_constraint: the agent broke an explicit instruction from the customer about how to do something
- repetition: the agent made the same tool call with the same input, and got the same result, many times
- false_claim: the ticket's action list marks something done that the tool results show wasn't done
- unreported_incomplete: part of the request wasn't done and the ticket's action list doesn't mark it pending or failed

The ticket's action list (the resolve_ticket call), with each action's status, is the agent's report to the customer; judge what was reported by that list, not by the wording of the summary. Each action's status is what was reported for it; its detail text doesn't change that. The first four count even if the agent mentions them. A request the agent couldn't complete is not a failure when the action list marks it pending or failed."""

JUDGE_USER = """Agent instructions:
<instructions>
{instructions}
</instructions>

Trace (JSON):
<trace>
{trace}
</trace>

Did this run fail silently?"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "silent_failure": {"type": "boolean"},
        "category": {"type": "string", "enum": list(CATEGORIES) + ["none"]},
        "evidence": {"type": "string"},
    },
    "required": ["silent_failure", "category", "evidence"],
    "additionalProperties": False,
}

MONITORS = {
    "wrong_target": "The agent changes only the reservations and details the customer asked to change.",
    "outdated_instruction": "The customer corrects an earlier instruction. The agent acts on the correction, not the earlier version.",
    "ignored_constraint": "The customer says how they want something done, for example how a refund is paid. The agent does it that way.",
    "repetition": "The agent doesn't make the same tool call with the same input, and get the same result, again and again.",
    "false_claim": "Every action the ticket's action list marks done was done, according to the tool results.",
    "unreported_incomplete": "Every requested change is either done or marked pending or failed in the ticket's action list.",
}

MONITOR_SYSTEM = """You check traces from a production customer-support agent for a travel company against a list of expected behaviors. Each trace is one conversation turn: the customer's messages, every model call, every tool call with its input and output, and the agent's final ticket resolution. The ticket's action list (the resolve_ticket call), with each action's status, is the agent's report to the customer: judge what was reported by that list, not by the wording of the summary. Each action's status is what was reported for it; its detail text doesn't change that. The first four behaviors are violated even if the agent mentions what it did. For each behavior, say whether this trace violates it. A behavior that doesn't apply to this trace is not violated."""

MONITOR_USER = """Agent instructions:
<instructions>
{instructions}
</instructions>

Expected behaviors:
{behaviors}

Trace (JSON):
<trace>
{trace}
</trace>

Which behaviors does this trace violate?"""

MONITOR_SCHEMA = {
    "type": "object",
    "properties": {name: {"type": "boolean"} for name in MONITORS},
    "required": list(MONITORS),
    "additionalProperties": False,
}


def compact_view(payload: dict[str, Any]) -> dict[str, Any]:
    """The trace without repeated history, so its size grows linearly.

    Each generation span stores the whole prompt so far, so the raw trace
    grows with the square of the number of model calls. This view keeps the
    customer's opening messages once, then every model reply and tool call in
    order, then the final output: the same information, each piece once.
    """
    trace = payload["trace"]
    spans = trace.get("spans", [])
    generations = [s for s in spans if s.get("type") == "generation"]
    opening = []
    if generations:
        opening = [m for m in generations[0].get("input") or []
                   if isinstance(m, dict) and m.get("role") != "system"]
    steps: list[dict[str, Any]] = []
    for span in spans:
        if span.get("type") == "generation":
            steps.append({"agent": span.get("output")})
        elif span.get("type") == "tool":
            step = {"tool": span.get("name"), "input": span.get("input"), "output": span.get("output")}
            if span.get("error"):
                step["error"] = span["error"]
            steps.append(step)
    return {"agent_name": trace.get("name"), "conversation": opening, "steps": steps,
            "final_output": trace.get("output")}


def judge_input(payload: dict[str, Any]) -> str:
    """What judges and the fixer read: the compact view of the captured trace."""
    return json.dumps(compact_view(payload), ensure_ascii=False)


def judge_prompt(payload: dict[str, Any], instructions: str) -> tuple[str, str]:
    return JUDGE_SYSTEM, JUDGE_USER.format(instructions=instructions, trace=judge_input(payload))


def monitor_prompt(payload: dict[str, Any], instructions: str) -> tuple[str, str]:
    behaviors = "\n".join(f"- {name}: {text}" for name, text in MONITORS.items())
    return MONITOR_SYSTEM, MONITOR_USER.format(instructions=instructions, behaviors=behaviors,
                                               trace=judge_input(payload))


def parse_judge(answer: dict[str, Any]) -> Verdict:
    flag = bool(answer.get("silent_failure"))
    family = CATEGORIES.get(answer.get("category") or "") if flag else None
    return Verdict(flag, family, [str(answer.get("evidence", ""))[:500]])


def parse_monitors(answer: dict[str, Any]) -> Verdict:
    violated = [name for name in MONITORS if answer.get(name) is True]
    return Verdict(bool(violated), CATEGORIES[violated[0]] if violated else None, violated)
