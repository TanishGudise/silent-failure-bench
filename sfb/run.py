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
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from . import detectors, fixloop, scripted
from .agent import AgentConfig, run_episode
from .analysis import detector_table, fmt, prevalence
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


def model_for(gateway: str, run: str, effort: str | None = None):
    from .llm import AnthropicModel

    return AnthropicModel(gateway, run=run, effort=effort)


def run_episodes(store: Store, jobs: Iterable[tuple[Task, AgentConfig, int]], gateway: str,
                 run: str, workers: int) -> list[dict[str, Any]]:
    model = model_for(gateway, run)
    todo = []
    for task, config, rep in jobs:
        key = f"{task.key}|{config.fingerprint}|{rep}"
        if key not in store.rows:
            todo.append((key, task, config))

    def one(item):
        key, task, config = item
        try:
            episode = run_episode(task, config, model, max_model_calls=MAX_MODEL_CALLS,
                                  max_tokens=AGENT_MAX_TOKENS)
        except Exception as exc:  # recorded and reported, never retried silently
            store.add({"key": key, "error": f"{type(exc).__name__}: {exc}"[:500]})
            return
        store.add({"key": key, "record": episode.record(), "trace": episode.trace})

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))
    return [store.rows[f"{t.key}|{c.fingerprint}|{r}"] for t, c, r in jobs]


def cmd_episodes(args) -> None:
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
    rows = run_episodes(store, jobs, args.gateway, args.run, args.workers)
    errors = [r for r in rows if "error" in r]
    print(f"{len(rows)} episodes, {len(errors)} errors")


def cmd_detect(args) -> None:
    episodes = Store(RUNS / args.run / "episodes.jsonl")
    out = Store(RUNS / args.run / f"detect.jsonl")
    instructions = json.loads((RUNS / args.run / "config.json").read_text())["system_prompt"]
    wanted = args.detectors.split(",")
    model = model_for(args.gateway, args.run + "-detect") if {"d1", "d1b"} & set(wanted) else None

    def one(row):
        if "error" in row:
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
                answer, usage, stop = model.complete_json(system=system, user=user, schema=schema,
                                                          max_tokens=JUDGE_MAX_TOKENS)
                if answer is None:
                    out.add({"key": key, "episode": row["key"], "detector": name, "error": stop, "usage": usage})
                    continue
                v = (detectors.parse_judge if name == "d1" else detectors.parse_monitors)(answer)
            out.add({"key": key, "episode": row["key"], "detector": name,
                     "flag": v.flag, "family": v.family, "reasons": v.reasons, "usage": usage})

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, list(episodes.rows.values())))
    print(f"{len(out.rows)} detector results")


def cmd_fixloop(args) -> None:
    bench = Store(RUNS / args.run / "episodes.jsonl")
    store = Store(RUNS / args.run / "fixloop_episodes.jsonl")
    fixes = Store(RUNS / args.run / "fixes.jsonl")
    baseline = AgentConfig()
    fixer = model_for(args.gateway, args.run + "-fixer")
    results = []
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
        base_jobs = [(t, baseline, 0) for t in
                     fixloop.heldout_tasks(family, fixloop.GATE_HELDOUT)
                     + fixloop.heldout_tasks(family, outcome_heldout)
                     + fixloop.regression_tasks(family, fixloop.GATE_REGRESSION)
                     + fixloop.regression_tasks(family, outcome_regression)]
        base_rows = run_episodes(store, base_jobs, args.gateway, args.run + "-fixloop", args.workers)
        for trig in triggers:
            key = f"fix|{trig['key']}"
            if key not in fixes.rows:
                system, user = fixloop.fixer_prompt(baseline, family, trig["trace"])
                answer, usage, stop = fixer.complete_json(system=system, user=user, schema=fixloop.FIXER_SCHEMA,
                                                          max_tokens=FIXER_MAX_TOKENS)
                fixes.add({"key": key, "family": family, "answer": answer, "usage": usage, "stop": stop})
            answer = fixes.rows[key]["answer"]
            if not answer:
                print(f"{key}: fixer gave no usable answer")
                continue
            fixed = fixloop.apply_fix(baseline, answer)
            seed = trig["record"]["seed"]
            fix_jobs = ([(build(family, seed, True), fixed, rep) for rep in range(fixloop.REPLAYS)]
                        + [(t, fixed, 0) for t, _, _ in base_jobs])
            fix_rows = run_episodes(store, fix_jobs, args.gateway, args.run + "-fixloop", args.workers)
            labels = [r.get("record", {}).get("label", "error") for r in fix_rows]
            base_labels = [r.get("record", {}).get("label", "error") for r in base_rows]
            fails = lambda ls: sum(fixloop.failed(x) or x == "error" for x in ls)
            n_rep, g_h, o_h, g_r = fixloop.REPLAYS, len(fixloop.GATE_HELDOUT), len(outcome_heldout), len(fixloop.GATE_REGRESSION)
            fix_rest, base_rest = labels[n_rep:], base_labels
            cut = [0, g_h, g_h + o_h, g_h + o_h + g_r, len(base_rest)]
            seg = lambda ls, i: ls[cut[i]:cut[i + 1]]
            result = fixloop.FixResult(
                family=family, trigger=trig["key"], replay_labels=labels[:n_rep],
                gate_heldout=(fails(seg(base_rest, 0)), fails(seg(fix_rest, 0))),
                outcome_heldout=(fails(seg(base_rest, 1)), fails(seg(fix_rest, 1))),
                gate_regression=(fails(seg(base_rest, 2)), fails(seg(fix_rest, 2))),
                outcome_regression=(fails(seg(base_rest, 3)), fails(seg(fix_rest, 3))),
            )
            results.append(result)
            print(json.dumps(result.record()))
    (RUNS / args.run / "fixloop_summary.json").write_text(json.dumps({
        "fixes": [r.record() for r in results],
        "replay_false_pass": fixloop.replay_false_pass(results),
        "gates": fixloop.gate_table(results),
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


def cmd_report(args) -> None:
    rows = load_rows(args.run)
    print(json.dumps(prevalence(rows), indent=1))
    for slice_name, sub in (("induced", [r for r in rows if r["induced"]]),
                            ("natural", [r for r in rows if not r["induced"]])):
        for det in ("d1", "d1b", "d2", "d3"):
            table = detector_table(sub, det)
            if not any(table["recall"].values()) and not table["false_flags"]["natural"]["n"] \
                    and not table["false_flags"]["recovered"]["n"]:
                continue
            print(f"\n[{slice_name}] {det}")
            for fam, r in table["recall"].items():
                print(f"  recall {fam:18s} {fmt(r)}")
            print(f"  precision            {fmt(table['precision'])}")
            for k, r in table["false_flags"].items():
                print(f"  false flags {k:9s} {fmt(r)}")


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
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=cmd_fixloop)
    p = sub.add_parser("report")
    p.add_argument("--run", required=True)
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
