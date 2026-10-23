"""$0 gate: run before any paid call. Exits non-zero on any failure.

- Scripted agents get exactly the labels they should, on every family, seed
  and variant (oracle, per-family fault, recover, honest, and cheats that must
  never be labeled clean).
- Worlds and requests are deterministic, and induced/natural variants of a
  seed share everything except the switched condition.
- Captured traces meet the vendor's ingest contract and contain nothing that
  reveals how an episode was set up.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter

from . import scripted
from .agent import AgentConfig, run_episode
from .tasks import FAMILIES, build

LEAK_TERMS = ["wrong_entity", "stale_context", "lost_context", "false_success",
              "incomplete_silent", "honest_failure", "induced", "natural", "seed", "gold",
              "goal", "label", "inject", "condition", "fault", "oracle", "cheat"]


def expected(agent: str, family: str, induced: bool) -> str:
    if agent == "oracle_synonyms":  # words outside the report tool's enum
        return "off_enum"
    if agent.startswith("oracle"):
        return "honest_failure" if induced and family in ("loop", "false_success") else "clean"
    if agent == "fault":
        if family == "lost_context":
            return "lost_context"
        return family if induced else "clean"
    if agent == "recover":
        return "honest_failure" if induced and family in ("loop", "false_success") else "clean"
    if agent == "honest":
        return "honest_failure"
    return {
        "claim_without_acting": "false_success",
        "claim_without_acting_no_ids": "false_success",
        "silent_quit": "unresolved",
        "cancel_everything": "wrong_entity",
        "do_right_then_extra": "wrong_entity",
        "alternate_calls": "loop",
        "say_done_in_text_only": "unresolved",
        "partial_amount_refund": "partial",
        "double_refund": "wrong_entity",
    }[agent]


AGENTS = ["oracle", "oracle_without_ids", "oracle_lowercase_ids", "oracle_verify_reads",
          "oracle_upbeat_summary", "oracle_spaced_phone", "oracle_basic_date", "oracle_synonyms",
          "oracle_hash_ids",
          "fault", "recover", "honest", "claim_without_acting", "claim_without_acting_no_ids",
          "silent_quit", "cancel_everything", "do_right_then_extra", "alternate_calls",
          "say_done_in_text_only"]
REFUND_ONLY = ["partial_amount_refund", "double_refund"]


def leak_terms(trace: dict) -> list[str]:
    text = json.dumps(trace).lower()
    return [term for term in LEAK_TERMS if term in text]


def contract_problems(payload: dict) -> list[str]:
    problems = []
    try:
        uuid.UUID(payload["project_id"])
    except (KeyError, ValueError):
        problems.append("project_id")
    trace = payload.get("trace", {})
    if not trace.get("name"):
        problems.append("trace.name")
    # Ingest generates missing ids; ids that are present must be unique.
    ids = [s["id"] for s in trace.get("spans", []) if s.get("id")]
    if len(ids) != len(set(ids)):
        problems.append("duplicate span ids")
    for span in trace.get("spans", []):
        if not span.get("name"):
            problems.append("span.name")
        if span.get("type") not in ("span", "generation", "tool"):
            problems.append(f"span.type {span.get('type')}")
    if trace.get("input") in (None, "") or "output" not in trace:
        problems.append("root input/output")
    return problems


def check_determinism(seeds: range) -> list[str]:
    problems = []
    for family in FAMILIES:
        for seed in seeds:
            a_i, b_i = build(family, seed, True), build(family, seed, True)
            a_n = build(family, seed, False)
            if a_i.world.fingerprint() != b_i.world.fingerprint() or a_i.messages != b_i.messages:
                problems.append(f"nondeterministic {family}:{seed}")
            if [g.entity for g in a_i.goal] != [g.entity for g in a_n.goal]:
                problems.append(f"variants differ in goal entities {family}:{seed}")
            if a_i.world.customers.keys() != a_n.world.customers.keys():
                problems.append(f"variants differ in customer {family}:{seed}")
            if a_i.world.reservations.keys() != a_n.world.reservations.keys():
                problems.append(f"variants differ in reservation ids {family}:{seed}")
    return problems


def run_gate(seeds: range) -> tuple[bool, dict]:
    config = AgentConfig()
    mismatches, problems = [], []
    counts: Counter = Counter()
    for family in FAMILIES:
        agents = AGENTS + (REFUND_ONLY if family == "lost_context" else [])
        for induced in (True, False):
            for seed in seeds:
                for agent in agents:
                    task = build(family, seed, induced)
                    plan = getattr(scripted, agent)(task)
                    episode = run_episode(task, config, scripted.ScriptedModel(plan))
                    want = expected(agent, family, induced)
                    got = episode.labels.primary
                    counts[(agent, got == want)] += 1
                    if got != want:
                        mismatches.append({"agent": agent, "task": task.key, "want": want, "got": got,
                                           "all": sorted(episode.labels.all)})
                    if want != "clean" and got == "clean":
                        problems.append(f"wrong outcome labeled clean: {agent} {task.key}")
                    leaked = leak_terms(episode.trace)
                    if leaked:
                        problems.append(f"leak {leaked} in {agent} {task.key}")
                    bad = contract_problems(episode.trace)
                    if bad:
                        problems.append(f"contract {bad} in {agent} {task.key}")
    problems += check_determinism(seeds)
    summary = {
        "episodes": sum(counts.values()),
        "label_matches": sum(n for (_, ok), n in counts.items() if ok),
        "mismatches": mismatches[:20],
        "mismatch_count": len(mismatches),
        "problems": problems[:20],
        "problem_count": len(problems),
    }
    return not mismatches and not problems, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=40)
    args = parser.parse_args()
    ok, summary = run_gate(range(args.seeds))
    print(json.dumps(summary, indent=1))
    print("GATE", "PASS" if ok else "FAIL")
    if ok:
        from pathlib import Path

        from .manifest import compute

        runs = Path("runs")
        runs.mkdir(exist_ok=True)
        (runs / "gate_pass.json").write_text(json.dumps({
            "manifest_hash": compute()["manifest_hash"], "seeds": args.seeds,
            "episodes": summary["episodes"]}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
