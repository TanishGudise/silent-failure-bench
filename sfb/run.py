"""Command line runner. Paid commands talk only to the local gateway.

  episodes  run agent episodes        (paid)
  detect    run detectors on a run    (D1/D1b paid, D2/D3 free)
  fixloop   propose and check fixes   (paid)
  report    tables with Wilson CIs    (free)
  sizes     trace size vs turn count  (free, scripted)
  quote     pilot cost estimate       (free, scripted)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from . import detectors, fixloop, scripted
from .agent import AgentConfig, run_episode
from .analysis import detector_table, fmt, prevalence, wilson
from .gateway import PRICES
from .tasks import FAMILIES, Task, build
from .world import Customer, Refund, Reservation, World

RUNS = Path("runs")
MAX_MODEL_CALLS = 10
AGENT_MAX_TOKENS = 4096
JUDGE_MAX_TOKENS = 2048
FIXER_MAX_TOKENS = 8192


def parse_range(text: str) -> range:
    lo, _, hi = text.partition("-")
    return range(int(lo), int(hi or lo) + 1)


class Store:
    """Append-only JSONL keyed by `key`; resumable."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: dict[str, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows[row["key"]] = row

    def add(self, row: dict[str, Any]) -> None:
        with self.lock:
            with open(self.path, "a") as handle:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            self.rows[row["key"]] = row


def world_from(snapshot: dict[str, Any]) -> World:
    return World(
        customers={k: Customer(**v) for k, v in snapshot["customers"].items()},
        reservations={k: Reservation(**v) for k, v in snapshot["reservations"].items()},
        refunds=[Refund(**r) for r in snapshot["refunds"]],
    )


class BudgetExhausted(SystemExit):
    pass


def preflight() -> str:
    """Paid commands run only on the frozen code, after the gate passed on it."""
    from .manifest import ROOT, compute

    current = compute()
    if os.environ.get("SFB_TEST_MODE") == "1":  # tests against the local fake API only
        return current["manifest_hash"]
    saved = json.loads((ROOT / "manifest.json").read_text())
    if saved["manifest_hash"] != current["manifest_hash"]:
        raise SystemExit("code or pre-registration changed since the freeze: amend, re-freeze, re-run the gate")
    marker = RUNS / "gate_pass.json"
    passed = json.loads(marker.read_text()) if marker.exists() else {}
    if passed.get("manifest_hash") != current["manifest_hash"] or passed.get("seeds", 0) < 40:
        raise SystemExit("run `python -m sfb.gate --seeds 40` on the frozen code first")
    return current["manifest_hash"]


def model_for(gateway: str, run: str, effort: str | None = None):
    from .llm import AnthropicModel

    return AnthropicModel(gateway, run=run, effort=effort)


def _is_budget_refusal(exc: Exception) -> bool:
    return "budget cap reached" in str(exc)


def run_episodes(store: Store, jobs: Iterable[tuple[Task, AgentConfig, int]], gateway: str,
                 run: str, workers: int, manifest_hash: str,
                 retry_errors: bool = False) -> list[dict[str, Any]]:
    model = model_for(gateway, run)
    stop = threading.Event()
    todo = []
    for task, config, rep in jobs:
        key = f"{task.key}|{config.fingerprint}|{rep}"
        row = store.rows.get(key)
        if row is None or (retry_errors and "error" in row):
            todo.append((key, task, config))

    def one(item):
        if stop.is_set():
            return
        key, task, config = item
        try:
            episode = run_episode(task, config, model, max_model_calls=MAX_MODEL_CALLS,
                                  max_tokens=AGENT_MAX_TOKENS)
        except Exception as exc:
            if _is_budget_refusal(exc):
                stop.set()  # not the agent's doing: stop, store nothing, resume under a new cap
                return
            store.add({"key": key, "manifest": manifest_hash, "error": f"{type(exc).__name__}: {exc}"[:500]})
            return
        store.add({"key": key, "manifest": manifest_hash, "record": episode.record(), "trace": episode.trace})

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))
    if stop.is_set():
        raise BudgetExhausted("budget cap reached: stopped; completed episodes are saved")
    return [store.rows[f"{t.key}|{c.fingerprint}|{r}"] for t, c, r in jobs]


def cmd_episodes(args) -> None:
    manifest_hash = preflight()
    store = Store(RUNS / args.run / "episodes.jsonl")
    config = AgentConfig()
    from .manifest import compute

    (RUNS / args.run).mkdir(parents=True, exist_ok=True)
    (RUNS / args.run / "config.json").write_text(json.dumps(
        {"system_prompt": config.system_prompt, "tools": config.tool_descriptions,
         "fingerprint": config.fingerprint, "manifest_hash": compute()["manifest_hash"]}, indent=1))
    variants = [v == "induced" for v in args.variants.split(",")]
    jobs = [(build(f, s, v), config, 0) for f in args.families.split(",")
            for s in parse_range(args.seeds) for v in variants]
    rows = run_episodes(store, jobs, args.gateway, args.run, args.workers, manifest_hash, args.retry_errors)
    errors = [r for r in rows if "error" in r]
    print(f"{len(rows)} episodes, {len(errors)} errors")


def cmd_detect(args) -> None:
    wanted = args.detectors.split(",")
    manifest_hash = preflight() if {"d1", "d1b"} & set(wanted) else ""
    episodes = Store(RUNS / args.run / "episodes.jsonl")
    out = Store(RUNS / args.run / f"detect.jsonl")
    instructions = json.loads((RUNS / args.run / "config.json").read_text())["system_prompt"]
    models = {name: model_for(args.gateway, f"{args.run}-{name}") for name in ("d1", "d1b") if name in wanted}
    stop = threading.Event()

    def one(row):
        if "error" in row or stop.is_set():
            return
        for name in wanted:
            key = f"{row['key']}|{name}"
            if key in out.rows:
                continue
            payload = row["trace"]
            if name == "d2":
                v, usage = detectors.rules(payload), {}
            elif name == "d3":
                v, usage = detectors.probe(payload, world_from(row["record"]["final_world"])), {}
            else:
                system, user = (detectors.judge_prompt if name == "d1" else detectors.monitor_prompt)(payload, instructions)
                schema = detectors.JUDGE_SCHEMA if name == "d1" else detectors.MONITOR_SCHEMA
                try:
                    answer, usage, stop_reason = models[name].complete_json(
                        system=system, user=user, schema=schema, max_tokens=JUDGE_MAX_TOKENS)
                except Exception as exc:
                    if _is_budget_refusal(exc):
                        stop.set()
                        return
                    out.add({"key": key, "episode": row["key"], "detector": name, "manifest": manifest_hash,
                             "error": f"{type(exc).__name__}: {exc}"[:300]})
                    continue
                if answer is None:
                    out.add({"key": key, "episode": row["key"], "detector": name, "manifest": manifest_hash,
                             "error": stop_reason, "usage": usage})
                    continue
                v = (detectors.parse_judge if name == "d1" else detectors.parse_monitors)(answer)
            out.add({"key": key, "episode": row["key"], "detector": name, "manifest": manifest_hash,
                     "flag": v.flag, "family": v.family, "reasons": v.reasons, "usage": usage})

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, list(episodes.rows.values())))
    if stop.is_set():
        raise BudgetExhausted("budget cap reached: stopped; completed verdicts are saved")
    print(f"{len(out.rows)} detector results")


def cmd_fixloop(args) -> None:
    manifest_hash = preflight()
    bench = Store(RUNS / args.run / "episodes.jsonl")
    store = Store(RUNS / args.run / "fixloop_episodes.jsonl")
    fixes = Store(RUNS / args.run / "fixes.jsonl")
    baseline = AgentConfig()
    fixer = model_for(args.gateway, args.run + "-fixer")
    tag = args.run + "-fixloop"
    summary_path = RUNS / args.run / "fixloop_summary.json"
    previous = json.loads(summary_path.read_text())["fixes"] if summary_path.exists() else []
    results: dict[str, fixloop.FixResult] = {
        r["trigger"]: fixloop.FixResult(**{k: (tuple(v) if isinstance(v, list) and k.startswith(("gate_", "outcome_")) else v)
                                           for k, v in r.items() if k in fixloop.FixResult.__dataclass_fields__})
        for r in previous}

    def label_of(row: dict[str, Any]) -> str:
        return row["record"]["label"] if "record" in row else "error"

    for family in args.families.split(","):
        triggers = sorted((r for r in bench.rows.values() if "record" in r
                           and r["record"]["family"] == family and r["record"]["induced"]
                           and r["record"]["label"] == family), key=lambda r: r["record"]["seed"])
        triggers = triggers[: args.per_family]
        if not triggers:
            print(f"{family}: no triggering trace")
            continue
        outcome_heldout = range(0) if args.gate_only else fixloop.OUTCOME_HELDOUT
        outcome_regression = range(0) if args.gate_only else fixloop.OUTCOME_REGRESSION
        sets = [fixloop.heldout_tasks(family, fixloop.GATE_HELDOUT),
                fixloop.heldout_tasks(family, outcome_heldout),
                fixloop.regression_tasks(family, fixloop.GATE_REGRESSION),
                fixloop.regression_tasks(family, outcome_regression)]
        base_rows = run_episodes(store, [(t, baseline, 0) for tasks in sets for t in tasks],
                                 args.gateway, tag, args.workers, manifest_hash, args.retry_errors)
        for trig in triggers:
            key = f"fix|{trig['key']}"
            if key not in fixes.rows:
                system, user = fixloop.fixer_prompt(baseline, family, trig["trace"])
                try:
                    answer, usage, stop = fixer.complete_json(system=system, user=user, schema=fixloop.FIXER_SCHEMA,
                                                              max_tokens=FIXER_MAX_TOKENS)
                except Exception as exc:
                    if _is_budget_refusal(exc):
                        raise BudgetExhausted("budget cap reached at the fixer call") from exc
                    answer, usage, stop = None, {}, f"{type(exc).__name__}: {exc}"[:300]
                fixes.add({"key": key, "family": family, "manifest": manifest_hash, "answer": answer,
                           "usage": usage, "stop": stop})
            answer = fixes.rows[key]["answer"]
            if not answer:
                print(f"{key}: fixer gave no usable answer ({fixes.rows[key]['stop']})")
                continue
            fixed = fixloop.apply_fix(baseline, answer)
            trigger_task = build(family, trig["record"]["seed"], True)
            # Fresh re-runs of the trigger: the fixed config, and the unchanged
            # config (reps 1-3; rep 0 is the bench episode that triggered).
            replays = run_episodes(store, [(trigger_task, fixed, rep) for rep in range(fixloop.REPLAYS)],
                                   args.gateway, tag, args.workers, manifest_hash, args.retry_errors)
            nulls = run_episodes(store, [(trigger_task, baseline, rep) for rep in range(1, fixloop.REPLAYS + 1)],
                                 args.gateway, tag, args.workers, manifest_hash, args.retry_errors)
            fix_rows = run_episodes(store, [(t, fixed, 0) for tasks in sets for t in tasks],
                                    args.gateway, tag, args.workers, manifest_hash, args.retry_errors)
            cuts = [0]
            for tasks in sets:
                cuts.append(cuts[-1] + len(tasks))

            def fails(rows, i):
                return sum(fixloop.failed(label_of(r)) for r in rows[cuts[i]:cuts[i + 1]])

            errors = sum("error" in r for r in replays + nulls + fix_rows + base_rows)
            result = fixloop.FixResult(
                family=family, trigger=trig["key"],
                replay_labels=[label_of(r) for r in replays],
                null_replay_labels=[label_of(r) for r in nulls],
                gate_heldout=(fails(base_rows, 0), fails(fix_rows, 0)),
                outcome_heldout=(fails(base_rows, 1), fails(fix_rows, 1)),
                gate_regression=(fails(base_rows, 2), fails(fix_rows, 2)),
                outcome_regression=(fails(base_rows, 3), fails(fix_rows, 3)),
                errors=errors, gate_only=args.gate_only,
            )
            results[trig["key"]] = result
            print(json.dumps(result.record()))
    final = list(results.values())
    bad, passing = fixloop.replay_false_pass(final)
    lo, hi = wilson(bad, passing)
    summary_path.write_text(json.dumps({
        "fixes": [r.record() for r in final],
        "replay_false_pass": {"bad": bad, "replay_passing_scored": passing, "wilson95": [lo, hi]},
        "gates": fixloop.gate_table(final),
    }, indent=1))


def load_rows(run: str) -> list[dict[str, Any]]:
    episodes = Store(RUNS / run / "episodes.jsonl")
    detect_path = RUNS / run / "detect.jsonl"
    verdicts: dict[str, dict[str, Any]] = {}
    if detect_path.exists():
        for row in Store(detect_path).rows.values():
            if "flag" in row:
                verdicts.setdefault(row["episode"], {})[row["detector"]] = row
    rows = []
    for key, row in episodes.rows.items():
        if "record" not in row:
            continue
        rec = row["record"]
        rows.append({"key": key, "family": rec["family"], "induced": rec["induced"],
                     "label": rec["label"], "verdicts": verdicts.get(key, {})})
    return rows


def detector_errors(run: str) -> dict[str, int]:
    path = RUNS / run / "detect.jsonl"
    counts: dict[str, int] = {}
    if path.exists():
        for row in Store(path).rows.values():
            if "error" in row:
                counts[row["detector"]] = counts.get(row["detector"], 0) + 1
    return counts


def cost_per_1k(ledger: Path, run: str, detector: str, traces: int) -> float | None:
    if not ledger.exists() or not traces:
        return None
    spent = sum(e["usd"] for e in map(json.loads, ledger.read_text().splitlines())
                if e.get("run") == f"{run}-{detector}")
    return round(spent / traces * 1000, 4)


def cascade(rows: list[dict[str, Any]], judge: str) -> list[dict[str, Any]]:
    """D2 or D3 first; the judge only on traces neither flagged."""
    out = []
    for r in rows:
        v = r["verdicts"]
        if not all(k in v for k in ("d2", "d3", judge)):
            continue
        cheap = v["d2"]["flag"] or v["d3"]["flag"]
        flag = cheap or v[judge]["flag"]
        family = (v["d2"] if v["d2"]["flag"] else v["d3"] if v["d3"]["flag"] else v[judge]).get("family")
        out.append({**r, "verdicts": {"cascade": {"flag": flag, "family": family, "judged": not cheap}}})
    return out


def cmd_report(args) -> None:
    rows = load_rows(args.run)
    print(json.dumps(prevalence(rows), indent=1))
    print("detector errors (excluded from the tables below):", detector_errors(args.run))
    for slice_name, sub in (("induced", [r for r in rows if r["induced"]]),
                            ("natural", [r for r in rows if not r["induced"]])):
        tables = {det: detector_table(sub, det) for det in ("d1", "d1b", "d2", "d3")}
        for judge in ("d1", "d1b"):
            casc = cascade(sub, judge)
            if casc:
                tables[f"d2+d3 then {judge}"] = detector_table(casc, "cascade")
                share = sum(r["verdicts"]["cascade"]["judged"] for r in casc) / len(casc)
                tables[f"d2+d3 then {judge}"]["judge_calls_share"] = round(share, 3)
        for det, table in tables.items():
            if not any(t["n"] for t in table["recall"].values()) and not any(
                    t["n"] for t in table["false_flags"].values()):
                continue
            print(f"\n[{slice_name}] {det}")
            for fam, r in table["recall"].items():
                print(f"  recall {fam:18s} {fmt(r)}")
            print(f"  precision            {fmt(table['precision'])}")
            for k, r in table["false_flags"].items():
                print(f"  false flags {k:9s} {fmt(r)}")
            if "judge_calls_share" in table:
                print(f"  judge calls needed   {table['judge_calls_share']:.0%} of traces")
    if args.ledger:
        traces = len(rows)
        for det in ("d1", "d1b"):
            print(f"{det}: ${cost_per_1k(Path(args.ledger), args.run, det, traces)} per 1,000 traces")


def _lookups(task: Task, k: int):
    def plan():
        found = yield scripted.Call("find_customer", {"email": task.customer_email})
        for _ in range(k):
            yield scripted.Call("get_reservation", {"reservation_id": task.goal[0].entity})
        yield scripted.Call("resolve_ticket", {"summary": "Checked.", "actions": []})
    return plan


def cmd_sizes(args) -> None:
    """Trace bytes vs number of model calls, through the real SDK, $0."""
    out = []
    for verbose in (False, True):
        task = build("lost_context", 0, verbose)
        for k in range(0, args.max_turns - 1):
            episode = run_episode(task, AgentConfig(), scripted.ScriptedModel(_lookups(task, k)))
            trace = episode.trace["trace"]
            spans = trace["spans"]
            gen_bytes = sum(len(json.dumps(s)) for s in spans if s["type"] == "generation")
            total = len(json.dumps(episode.trace))
            out.append({"long_tool_outputs": verbose, "model_calls": episode.model_calls,
                        "trace_bytes": total, "generation_span_bytes": gen_bytes,
                        "est_tokens": round(total / args.bytes_per_token)})
    for row in out:
        print(json.dumps(row))


def cmd_quote(args) -> None:
    """Per-request token estimates from scripted episodes, priced at list rates."""
    rate_in, rate_out = PRICES["claude-haiku-5-5"][:2]
    agent_in, judge_in = [], []
    for family in FAMILIES:
        for induced in (True, False):
            task = build(family, 0, induced)
            episode = run_episode(task, AgentConfig(), scripted.ScriptedModel(scripted.oracle(task)))
            prompts = [s for s in episode.trace["trace"]["spans"] if s["type"] == "generation"]
            tools_bytes = len(json.dumps(AgentConfig().tools()))
            agent_in.append(sum(len(json.dumps(s["input"])) + tools_bytes for s in prompts))
            judge_in.append(len(detectors.judge_input(episode.trace)) + 2500)
    bpt = args.bytes_per_token
    ep_in = max(agent_in) / bpt * args.call_factor
    ep_out = args.out_per_call * MAX_MODEL_CALLS
    judge_tokens = max(judge_in) / bpt * args.call_factor
    episode_usd = (ep_in * rate_in + ep_out * rate_out) / 1e6
    judge_usd = (judge_tokens * rate_in + 600 * rate_out) / 1e6
    print(json.dumps({"episode_usd_est": round(episode_usd, 5), "judge_call_usd_est": round(judge_usd, 5),
                      "assumptions": vars(args) | {"agent_prompt_bytes_max": max(agent_in),
                                                   "judge_input_bytes_max": max(judge_in)}}, indent=1, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("episodes")
    p.add_argument("--run", required=True)
    p.add_argument("--gateway", required=True)
    p.add_argument("--families", default=",".join(FAMILIES))
    p.add_argument("--seeds", default="0-4")
    p.add_argument("--variants", default="induced,natural")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--retry-errors", action="store_true", help="re-run episodes that errored (logged)")
    p.set_defaults(fn=cmd_episodes)
    p = sub.add_parser("detect")
    p.add_argument("--run", required=True)
    p.add_argument("--gateway", default="")
    p.add_argument("--detectors", default="d2,d3")
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=cmd_detect)
    p = sub.add_parser("fixloop")
    p.add_argument("--run", required=True)
    p.add_argument("--gateway", required=True)
    p.add_argument("--families", default=",".join(FAMILIES))
    p.add_argument("--per-family", type=int, default=2)
    p.add_argument("--gate-only", action="store_true", help="pilot: replay and gate sets, no outcome sets")
    p.add_argument("--retry-errors", action="store_true", help="re-run episodes that errored (logged)")
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=cmd_fixloop)
    p = sub.add_parser("report")
    p.add_argument("--run", required=True)
    p.add_argument("--ledger", default="", help="gateway ledger for cost per 1,000 traces")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("sizes")
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--bytes-per-token", type=float, default=3.5)
    p.set_defaults(fn=cmd_sizes)
    p = sub.add_parser("quote")
    p.add_argument("--bytes-per-token", type=float, default=3.0)
    p.add_argument("--call-factor", type=float, default=2.0, help="model episodes use up to this many times the scripted calls")
    p.add_argument("--out-per-call", type=int, default=1500)
    p.set_defaults(fn=cmd_quote)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
