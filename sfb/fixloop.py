"""Fix loop: propose a config fix from a triggering trace, then check it.

Seeds (pre-registered): bench 0-99; gate held-out 5000-5004, outcome
held-out 5005-5024 (same family, induced); gate regression 6000-6004, outcome
regression 6005-6024 (other families, alternating induced/natural). Replay is
the triggering task itself, re-run 3 times with the fixed config. A null arm
re-runs it 3 times with the unchanged config, so the gates can be read
against what an empty fix would get.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .agent import TOOL_SCHEMAS, AgentConfig
from .detectors import judge_input
from .labels import NOT_SCORED, SILENT
from .tasks import FAMILIES, Task, build

ISSUE_TITLES = {
    "wrong_entity": "Agent cancels or changes a reservation the customer didn't ask about",
    "stale_context": "Agent acts on a date the customer had already corrected",
    "lost_context": "Agent ignores how the customer asked to be refunded",
    "loop": "Agent repeats the same failing tool call over and over",
    "false_success": "Agent tells the customer a cancellation is done when it isn't",
    "partial": "Agent reports a multi-part request as done when one part failed",
}

FIXER_SYSTEM = """You maintain a production customer-support agent. You get an issue with a trace that shows it, plus the agent's current instructions and tool descriptions. Propose the smallest change to the instructions and/or tool descriptions that fixes the issue without breaking other behavior. You can't change code or tools, only their text."""

FIXER_USER = """Issue: {title}

Current instructions:
<instructions>
{instructions}
</instructions>

Current tool descriptions:
<tools>
{tools}
</tools>

Trace showing the issue (JSON):
<trace>
{trace}
</trace>

Return the complete new instructions, and new text for any tool description you change."""

FIXER_SCHEMA = {
    "type": "object",
    "properties": {
        "instructions": {"type": "string"},
        "tool_changes": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string", "enum": list(TOOL_SCHEMAS)},
                           "description": {"type": "string"}},
            "required": ["name", "description"], "additionalProperties": False}},
        "rationale": {"type": "string"},
    },
    "required": ["instructions", "tool_changes", "rationale"],
    "additionalProperties": False,
}

REPLAYS = 3
GATE_HELDOUT = range(5000, 5005)
OUTCOME_HELDOUT = range(5005, 5025)
GATE_REGRESSION = range(6000, 6005)
OUTCOME_REGRESSION = range(6005, 6025)
# Outcome thresholds on 20 paired seeds each (pre-registered).
FIXED_DROP = 3
REGRESS_RISE = 3


def fixer_prompt(config: AgentConfig, family: str, trigger_trace: dict[str, Any]) -> tuple[str, str]:
    tools = "\n".join(f"- {name}: {text}" for name, text in config.tool_descriptions)
    return FIXER_SYSTEM, FIXER_USER.format(title=ISSUE_TITLES[family], instructions=config.system_prompt,
                                           tools=tools, trace=judge_input(trigger_trace))


def apply_fix(config: AgentConfig, answer: dict[str, Any]) -> AgentConfig:
    changes = {c["name"]: c["description"] for c in answer.get("tool_changes") or []
               if isinstance(c, dict) and c.get("name") in TOOL_SCHEMAS}
    return config.with_edits(system_prompt=answer.get("instructions") or None, tool_descriptions=changes)


def heldout_tasks(family: str, seeds: range) -> list[Task]:
    return [build(family, s, True) for s in seeds]


def regression_tasks(family: str, seeds: range) -> list[Task]:
    others = [f for f in FAMILIES if f != family]
    return [build(others[i % len(others)], s, i % 2 == 0) for i, s in enumerate(seeds)]


PASSING = {"clean", "honest_failure"}


def failed(label: str) -> bool:
    """Outcome and gate sets: positives and aborted/unresolved runs count as
    failures in both arms; episodes that errored are tracked separately."""
    return label in SILENT or label in NOT_SCORED


@dataclass
class FixResult:
    family: str
    trigger: str
    replay_labels: list[str]
    null_replay_labels: list[str]
    gate_heldout: tuple[int, int]      # (baseline fails, fixed fails)
    gate_regression: tuple[int, int]
    outcome_heldout: tuple[int, int]
    outcome_regression: tuple[int, int]
    null_gate_heldout: tuple[int, int] = (0, 0)     # (baseline rep 0 fails, baseline rep 1 fails)
    null_gate_regression: tuple[int, int] = (0, 0)
    errors: int = 0                    # errored episodes anywhere in this fix's runs, either arm
    gate_only: bool = False

    @staticmethod
    def _passes(labels: list[str]) -> bool:
        return bool(labels) and all(label in PASSING for label in labels)

    @property
    def replay_pass(self) -> bool:
        return self._passes(self.replay_labels)

    @property
    def null_replay_pass(self) -> bool:
        return self._passes(self.null_replay_labels)

    @property
    def outcome(self) -> str:
        if self.gate_only:
            return "gate_only"
        if self.errors:
            return "excluded"
        bh, fh = self.outcome_heldout
        br, fr = self.outcome_regression
        if fr >= br + REGRESS_RISE or fh >= bh + REGRESS_RISE:
            return "regressed"
        if bh < FIXED_DROP:
            return "undetermined"  # too few baseline failures to show a fix
        if fh <= bh - FIXED_DROP:
            return "fixed"
        return "unchanged"

    def gate(self, name: str, *, null: bool = False) -> bool:
        g1 = self.null_replay_pass if null else self.replay_pass
        # The null fix is the unchanged config run again (rep 1) on the gate seeds.
        held, reg = ((self.null_gate_heldout, self.null_gate_regression) if null
                     else (self.gate_heldout, self.gate_regression))
        g2 = g1 and held[1] < held[0]
        g3 = g2 and reg[1] <= reg[0]
        return {"G1": g1, "G2": g2, "G3": g3}[name]

    def record(self) -> dict[str, Any]:
        return {**self.__dict__, "replay_pass": self.replay_pass, "outcome": self.outcome,
                "gates": {g: self.gate(g) for g in ("G1", "G2", "G3")},
                "null_gates": {g: self.gate(g, null=True) for g in ("G1", "G2", "G3")}}


SCORED = {"fixed", "unchanged", "regressed"}


def trigger_seed(trigger: str) -> int:
    return int(trigger.split("|", 1)[0].split(":")[1])


def gate_table(results: list[FixResult]) -> dict[str, dict[str, int]]:
    """Per gate, over fixes with a scored outcome: PRs it would open, how many
    are bad, and good fixes kept and blocked. The null fix is counted over the
    same scored fixes: G1 per trigger (its own null replays); G2 and G3 once
    per family, since a family's triggers share one null run on the gate seeds,
    with the family's first scored trigger by seed supplying the null G1."""
    scored = [r for r in results if r.outcome in SCORED]
    per_family: dict[str, FixResult] = {}
    for r in sorted(scored, key=lambda r: trigger_seed(r.trigger)):
        per_family.setdefault(r.family, r)
    table = {}
    for gate in ("G1", "G2", "G3"):
        opened = [r for r in scored if r.gate(gate)]
        null_pool = scored if gate == "G1" else list(per_family.values())
        table[gate] = {
            "scored_fixes": len(scored),
            "opened": len(opened),
            "bad_opened": sum(r.outcome != "fixed" for r in opened),
            "good_kept": sum(r.outcome == "fixed" for r in opened),
            "good_blocked": sum(r.outcome == "fixed" and not r.gate(gate) for r in scored),
            "null_fix_passes": sum(r.gate(gate, null=True) for r in null_pool),
            "null_fix_tried": len(null_pool),
        }
    return table


def replay_false_pass(results: list[FixResult]) -> tuple[int, int]:
    """(replay-passing scored fixes that are unchanged or regressed, replay-passing scored fixes)."""
    passing = [r for r in results if r.replay_pass and r.outcome in SCORED]
    return sum(r.outcome != "fixed" for r in passing), len(passing)


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o))
