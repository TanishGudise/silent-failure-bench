"""Command line runner. Paid commands talk only to the local gateway.

  smoke     one paid episode, then reconcile its tokens with the console
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
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import detectors, fixloop, scripted
from .agent import AgentConfig, run_episode
from .analysis import detector_table, fmt, prevalence, rate, wilson
from .gateway import PRICES
from .labels import NOT_SCORED
from .tasks import FAMILIES, Task, build
from .world import Customer, Refund, Reservation, World

RUNS = Path("runs")
MAX_MODEL_CALLS = 10
AGENT_MAX_TOKENS = 4096
JUDGE_MAX_TOKENS = 2048
FIXER_MAX_TOKENS = 8192
SMOKE_TASK = ("false_success", 0, False)
TOTAL_BUDGET_USD = 20.0
# Errors that would repeat on every call stop a run instead of being stored.
STOP_STATUSES = {401, 403}
STOP_PHRASES = ("usage limit", "spend limit", "credit balance")
MAX_ERRORS_IN_A_ROW = 5


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


class RunStopped(SystemExit):
    pass


def preflight(require_smoke: bool = True) -> str:
    """Paid commands run only on the frozen code, after the gate passed on it
    and the smoke episode's tokens matched the console."""
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
    # The smoke checks the gateway's accounting against the console, once per
    # workspace and gateway: a re-freeze that leaves sfb/gateway.py alone keeps it.
    smoke_path = RUNS / "smoke.json"
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
    if require_smoke and not (smoke.get("confirmed") is True
                              and smoke.get("gateway_hash") == current["files"]["sfb/gateway.py"]):
        raise SystemExit("run `smoke` and confirm its tokens against the console with this gateway first")
    return current["manifest_hash"]


def model_for(gateway: str, run: str, effort: str | None = None):
    from .llm import AnthropicModel

    return AnthropicModel(gateway, run=run, effort=effort)


def stop_reason(exc: Exception) -> str | None:
    """Why an error should stop the run rather than be stored, if it should."""
    text = str(exc)
    if "budget cap reached" in text:
        return "budget cap reached"
    status = getattr(exc, "status_code", None)
    if status in STOP_STATUSES:
        return f"upstream refused the key ({status})"
    if isinstance(status, int) and 400 <= status < 500 and any(phrase in text.lower() for phrase in STOP_PHRASES):
        return "upstream usage or spend limit reached"
    return None


class Stopper:
    """Stops a run on a budget refusal, an upstream key or spend-limit
    refusal (nothing stored), or MAX_ERRORS_IN_A_ROW errors in a row (stored)."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.lock = threading.Lock()
        self.reason = ""
        self.streak = 0

    def _stop(self, reason: str) -> None:
        if not self.event.is_set():
            self.reason = reason
            self.event.set()

    def error(self, exc: Exception) -> bool:
        """Count an error; False if it's a stop and must not be stored."""
        reason = stop_reason(exc)
        with self.lock:
            if reason:
                self._stop(reason)
                return False
            self.streak += 1
            if self.streak >= MAX_ERRORS_IN_A_ROW:
                self._stop(f"{self.streak} errors in a row (last: {type(exc).__name__})")
            return True

    def ok(self) -> None:
        with self.lock:
            self.streak = 0

    def check(self, saved: str) -> None:
        if self.event.is_set():
            raise RunStopped(f"{self.reason}: stopped; completed {saved} are saved")


def run_episodes(store: Store, jobs: Iterable[tuple[Task, AgentConfig, int]], gateway: str,
                 run: str, workers: int, manifest_hash: str,
                 retry_errors: bool = False, stopper: Stopper | None = None) -> list[dict[str, Any]]:
    """stopper: shared across batches, so errors in a row are counted across them."""
    model = model_for(gateway, run)
    stopper = stopper or Stopper()
    todo = []
    retried = []
    for task, config, rep in jobs:
        key = f"{task.key}|{config.fingerprint}|{rep}"
        row = store.rows.get(key)
        if row is None or (retry_errors and "error" in row):
            todo.append((key, task, config))
            if row is not None:
                retried.append({"key": key, "previous_error": row["error"]})
    if retried:  # every re-run of an errored episode is logged
        with open(store.path.parent / "retries.jsonl", "a") as handle:
            for entry in retried:
                handle.write(json.dumps(entry) + "\n")

    def one(item):
        if stopper.event.is_set():
            return
        key, task, config = item
        try:
            episode = run_episode(task, config, model, max_model_calls=MAX_MODEL_CALLS,
                                  max_tokens=AGENT_MAX_TOKENS)
        except Exception as exc:
            # A budget or key refusal isn't the agent's doing: stop, store nothing.
            if stopper.error(exc):
                store.add({"key": key, "manifest": manifest_hash, "error": f"{type(exc).__name__}: {exc}"[:500]})
            return
        stopper.ok()
        store.add({"key": key, "manifest": manifest_hash, "record": episode.record(), "trace": episode.trace})

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))
    stopper.check("episodes")
    return [store.rows[f"{t.key}|{c.fingerprint}|{r}"] for t, c, r in jobs]


def episode_jobs(families: list[str], seeds: range, variants: list[bool],
                 config: AgentConfig) -> list[tuple[Task, AgentConfig, int]]:
    """Seed-major, so a run stopped at its cap leaves every family with the same seeds."""
    return [(build(f, s, v), config, 0) for s in seeds for f in families for v in variants]


def fix_order(triggers: dict[str, list[Any]], per_family: int) -> list[tuple[str, Any]]:
    """Every family's first trigger before any family's second, so a stop at
    the cap leaves the families balanced."""
    return [(family, found[rank]) for rank in range(per_family)
            for family, found in triggers.items() if rank < len(found)]


def pilot_prevalence(rows: list[dict[str, Any]], families_run: set[str]) -> dict[str, tuple[int, int, bool]]:
    """Per family in the run (errored episodes count as in it): (episodes
    labeled with the family, scored induced episodes, below 15%). A family with
    no scored induced episode is below."""
    out = {}
    for family in FAMILIES:
        if family not in families_run:
            continue
        sub = [r for r in rows if r["induced"] and r["family"] == family and r["label"] not in NOT_SCORED]
        hits = sum(r["label"] == family for r in sub)
        out[family] = (hits, len(sub), not sub or hits / len(sub) < 0.15)
    return out


def seed_major(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Episode rows in the order the bench ran them (errored rows last)."""
    def order(row: dict[str, Any]) -> tuple:
        rec = row.get("record")
        if rec is None:
            return (1, 0, 0, False)
        return (0, rec["seed"], FAMILIES.index(rec["family"]), not rec["induced"])
    return sorted(rows, key=order)


def ledger_rows(ledger: Path) -> list[dict[str, Any]]:
    if not ledger.exists():
        return []
    return [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]


UNBILLED_NOTES = {"unbilled error reply; retrying", "unbilled error reply"}
# Bound bookings whose cause lies with the workspace or the API, not one call
# (a non-standard tier or non-global region, or usage the gateway can't
# read): the smoke stops on them.
PERSISTENT_NOTES = ("unexpected ", "unreadable usage")
RUN_TAG_SUFFIXES = ("", "-smoke", "-d1", "-d1b", "-fixloop", "-fixer")


def ledger_totals(ledgers: list[Path], tags: set[str] | None = None) -> dict[str, Any]:
    """Requests, tokens and spend booked in these ledgers (under these run tags)."""
    rows = [r for path in ledgers for r in ledger_rows(path) if tags is None or r.get("run") in tags]
    tokens_in = tokens_out = 0
    notes = []
    for r in rows:
        usage = r.get("usage") or {}
        try:
            tokens_in += sum(int(usage.get(k) or 0) for k in
                             ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
            tokens_out += int(usage.get("output_tokens") or 0)
        except (TypeError, ValueError, AttributeError):
            notes.append("unreadable usage")
        if r.get("note") and r["note"] not in UNBILLED_NOTES:
            notes.append(r["note"])
    return {"requests": len(rows), "input_tokens": tokens_in, "output_tokens": tokens_out,
            "usd": round(sum(r["usd"] for r in rows), 6), "notes": notes}


def gateway_hash() -> str:
    from .manifest import compute

    return compute()["files"]["sfb/gateway.py"]


def cmd_smoke(args) -> None:
    """Reconcile the gateway's ledgers with the console, once per workspace and gateway.

    Run: one paid episode through the gateway (on a new workspace, its first
    call). Confirm: the console's input and output token counts for the
    workspace, over all dates, must equal the sums over every ledger exactly.
    A call booked at the bound because no usage came back adds no tokens to
    the sums, so the console's counts decide it; a bound booked for a
    non-standard tier or non-global region, or unreadable usage, stops the
    smoke. This command doesn't print the ledger's counts first.
    """
    path = RUNS / "smoke.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    ledgers = [Path(p) for p in args.ledger]
    current_gateway = gateway_hash()
    if not args.console_tokens:
        if saved.get("confirmed") and saved.get("gateway_hash") == current_gateway:
            raise SystemExit("the smoke is already confirmed for this workspace and gateway")
        if not args.gateway or not args.run:
            raise SystemExit("--gateway and --run are required to run the smoke episode")
        manifest_hash = preflight(require_smoke=False)
        store = Store(RUNS / args.run / f"smoke_episodes-{current_gateway[:8]}.jsonl")
        # An errored smoke episode is re-run by running `smoke` again (logged).
        (row,) = run_episodes(store, [(build(*SMOKE_TASK), AgentConfig(), 0)], args.gateway,
                              f"{args.run}-smoke", 1, manifest_hash, retry_errors=True)
        saved.update({"manifest_hash": manifest_hash, "gateway_hash": current_gateway, "run": args.run,
                      "episode_ok": "record" in row, "confirmed": False})
        path.write_text(json.dumps(saved, indent=1))
        if "record" not in row:
            raise SystemExit("the smoke episode errored; run `smoke` again to retry it")
        print("Smoke episode done. When the console's Usage page shows it, run `smoke --ledger <every ledger> "
              "--console-tokens IN,OUT --spend-limit USD` (one --ledger per ledger) with the workspace's token "
              "counts over all dates (input = uncached + cache write + cache read).")
        return
    if not saved:
        raise SystemExit("run the smoke episode first")
    totals = ledger_totals(ledgers)
    console = tuple(int(x) for x in args.console_tokens.split(","))
    problems = []
    if saved.get("gateway_hash") != current_gateway:
        problems.append("the gateway changed since the smoke episode")
    if not saved.get("episode_ok"):
        problems.append("the smoke episode errored")
    if not totals["requests"]:
        problems.append("no requests in the given ledgers")
    if console != (totals["input_tokens"], totals["output_tokens"]):
        problems.append("the console's token counts don't equal the ledgers'")
    persistent = [n for n in totals["notes"] if n.startswith(PERSISTENT_NOTES)]
    if persistent:
        problems.append(f"the gateway booked the bound for: {sorted(set(persistent))}")
    if args.spend_limit is None or not 0 < args.spend_limit <= TOTAL_BUDGET_USD:
        problems.append(f"record the workspace's console spend limit (0 < limit <= ${TOTAL_BUDGET_USD:.0f})")
    # Every attempt is kept, so a mismatch followed by a retyped match shows.
    saved.setdefault("attempts", []).append({
        "console_tokens": list(console), "ledger_tokens": [totals["input_tokens"], totals["output_tokens"]],
        "ledgers": [str(p) for p in ledgers], "gateway_hash": current_gateway, "bound_notes": totals["notes"],
        "spend_limit_usd": args.spend_limit,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "problems": problems})
    saved.update({"confirmed": not problems, "spend_limit_usd": args.spend_limit})
    path.write_text(json.dumps(saved, indent=1))
    print(f"SMOKE OK ({len(totals['notes'])} call(s) booked at the bound with no usage)" if not problems
          else "STOP: " + "; ".join(problems))
    if problems:
        raise SystemExit(1)

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
    jobs = episode_jobs(args.families.split(","), parse_range(args.seeds), variants, config)
    rows = run_episodes(store, jobs, args.gateway, args.run, args.workers, manifest_hash, args.retry_errors)
    errors = [r for r in rows if "error" in r]
    print(f"{len(rows)} episodes, {len(errors)} errors")


def cmd_detect(args) -> None:
    from .manifest import compute

    wanted = args.detectors.split(",")
    paid = {"d1", "d1b"} & set(wanted)
    if paid and not args.gateway:
        raise SystemExit("--gateway is required for d1/d1b")
    manifest_hash = preflight() if paid else compute()["manifest_hash"]
    episodes = Store(RUNS / args.run / "episodes.jsonl")
    out = Store(RUNS / args.run / "detect.jsonl")
    instructions = json.loads((RUNS / args.run / "config.json").read_text())["system_prompt"]
    models = {name: model_for(args.gateway, f"{args.run}-{name}") for name in ("d1", "d1b") if name in wanted}
    stopper = Stopper()
    if args.retry_errors:  # every re-run of an errored detector call is logged
        retried = [{"key": k, "detector": r["detector"], "previous_error": r["error"]}
                   for k, r in out.rows.items() if "error" in r and r["detector"] in wanted]
        if retried:
            with open(RUNS / args.run / "retries.jsonl", "a") as handle:
                for entry in retried:
                    handle.write(json.dumps(entry) + "\n")

    def one(row):
        if "error" in row or stopper.event.is_set():
            return
        for name in wanted:
            key = f"{row['key']}|{name}"
            if key in out.rows and not (args.retry_errors and "error" in out.rows[key]):
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
                    if not stopper.error(exc):
                        return
                    out.add({"key": key, "episode": row["key"], "detector": name, "manifest": manifest_hash,
                             "error": f"{type(exc).__name__}: {exc}"[:300]})
                    if stopper.event.is_set():
                        return
                    continue
                stopper.ok()
                if answer is None:
                    out.add({"key": key, "episode": row["key"], "detector": name, "manifest": manifest_hash,
                             "error": stop_reason, "usage": usage})
                    continue
                v = (detectors.parse_judge if name == "d1" else detectors.parse_monitors)(answer)
            out.add({"key": key, "episode": row["key"], "detector": name, "manifest": manifest_hash,
                     "flag": v.flag, "family": v.family, "reasons": v.reasons, "usage": usage})

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, seed_major(episodes.rows.values())))
    stopper.check("verdicts")
    print(f"{len(out.rows)} detector results")


def cmd_fixloop(args) -> None:
    manifest_hash = preflight()
    bench = Store(RUNS / args.run / "episodes.jsonl")
    store = Store(RUNS / args.run / "fixloop_episodes.jsonl")
    fixes = Store(RUNS / args.run / "fixes.jsonl")
    baseline = AgentConfig()
    fixer = model_for(args.gateway, args.run + "-fixer")
    stopper = Stopper()  # one for the whole loop: fixer calls and every episode batch
    tag = args.run + "-fixloop"
    summary_path = RUNS / args.run / "fixloop_summary.json"
    saved = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    results: dict[str, fixloop.FixResult] = {
        r["trigger"]: fixloop.FixResult(**{k: (tuple(v) if isinstance(v, list) and k.startswith(("gate_", "outcome_")) else v)
                                           for k, v in r.items() if k in fixloop.FixResult.__dataclass_fields__})
        for r in saved.get("fixes", [])}
    fixer_failures: dict[str, str] = saved.get("fixer_failures", {})

    def label_of(row: dict[str, Any]) -> str:
        return row["record"]["label"] if "record" in row else "error"

    families = args.families.split(",")
    triggers: dict[str, list[dict[str, Any]]] = {}
    for family in families:
        triggers[family] = sorted((r for r in bench.rows.values() if "record" in r
                                   and r["record"]["family"] == family and r["record"]["induced"]
                                   and r["record"]["label"] == family),
                                  key=lambda r: r["record"]["seed"])[: args.per_family]
        if not triggers[family]:
            print(f"{family}: no triggering trace")
    outcome_heldout = range(0) if args.gate_only else fixloop.OUTCOME_HELDOUT
    outcome_regression = range(0) if args.gate_only else fixloop.OUTCOME_REGRESSION

    def propose(family: str, trig: dict[str, Any]) -> dict[str, Any] | None:
        key = f"fix|{trig['key']}"
        if key not in fixes.rows:
            system, user = fixloop.fixer_prompt(baseline, family, trig["trace"])
            try:
                answer, usage, stop = fixer.complete_json(system=system, user=user, schema=fixloop.FIXER_SCHEMA,
                                                          max_tokens=FIXER_MAX_TOKENS)
            except Exception as exc:
                # Nothing stored, so the next run retries it; a budget or key
                # refusal (not the fixer's failure), or MAX_ERRORS_IN_A_ROW in a
                # row, stops the loop.
                if stopper.error(exc):
                    fixer_failures[trig["key"]] = f"call failed: {type(exc).__name__}"
                    print(f"{key}: fixer call failed ({type(exc).__name__}); re-run to retry")
                stopper.check("fixes")
                return None
            stopper.ok()
            fixes.add({"key": key, "family": family, "manifest": manifest_hash, "answer": answer,
                       "usage": usage, "stop": stop})
        answer = fixes.rows[key]["answer"]
        if not answer:
            # The model's own refusal, cutoff or unparseable answer: a real outcome, reported.
            fixer_failures[trig["key"]] = f"no usable answer: {fixes.rows[key]['stop']}"
            print(f"{key}: fixer gave no usable answer ({fixes.rows[key]['stop']})")
            return None
        fixer_failures.pop(trig["key"], None)
        return answer

    def evaluate(family: str, trig: dict[str, Any], answer: dict[str, Any]) -> fixloop.FixResult:
        sets = [fixloop.heldout_tasks(family, fixloop.GATE_HELDOUT),
                fixloop.heldout_tasks(family, outcome_heldout),
                fixloop.regression_tasks(family, fixloop.GATE_REGRESSION),
                fixloop.regression_tasks(family, outcome_regression)]
        # The family's baseline runs (stored, so later triggers reuse them).
        base_rows = run_episodes(store, [(t, baseline, 0) for tasks in sets for t in tasks],
                                 args.gateway, tag, args.workers, manifest_hash, args.retry_errors, stopper)
        # The null fix: the unchanged config again (rep 1) on the gate seeds.
        null_gate_rows = run_episodes(store, [(t, baseline, 1) for t in sets[0] + sets[2]],
                                      args.gateway, tag, args.workers, manifest_hash, args.retry_errors, stopper)
        null_held_fails = sum(fixloop.failed(label_of(r)) for r in null_gate_rows[:len(sets[0])])
        null_reg_fails = sum(fixloop.failed(label_of(r)) for r in null_gate_rows[len(sets[0]):])
        fixed = fixloop.apply_fix(baseline, answer)
        trigger_task = build(family, trig["record"]["seed"], True)
        # Fresh re-runs of the trigger: the fixed config, and the unchanged
        # config (reps 1-3; rep 0 is the bench episode that triggered).
        replays = run_episodes(store, [(trigger_task, fixed, rep) for rep in range(fixloop.REPLAYS)],
                               args.gateway, tag, args.workers, manifest_hash, args.retry_errors, stopper)
        nulls = run_episodes(store, [(trigger_task, baseline, rep) for rep in range(1, fixloop.REPLAYS + 1)],
                             args.gateway, tag, args.workers, manifest_hash, args.retry_errors, stopper)
        fix_rows = run_episodes(store, [(t, fixed, 0) for tasks in sets for t in tasks],
                                args.gateway, tag, args.workers, manifest_hash, args.retry_errors, stopper)
        cuts = [0]
        for tasks in sets:
            cuts.append(cuts[-1] + len(tasks))

        def fails(rows, i):
            return sum(fixloop.failed(label_of(r)) for r in rows[cuts[i]:cuts[i + 1]])

        errors = sum("error" in r for r in replays + nulls + fix_rows + base_rows + null_gate_rows)
        return fixloop.FixResult(
            family=family, trigger=trig["key"],
            replay_labels=[label_of(r) for r in replays],
            null_replay_labels=[label_of(r) for r in nulls],
            gate_heldout=(fails(base_rows, 0), fails(fix_rows, 0)),
            outcome_heldout=(fails(base_rows, 1), fails(fix_rows, 1)),
            gate_regression=(fails(base_rows, 2), fails(fix_rows, 2)),
            outcome_regression=(fails(base_rows, 3), fails(fix_rows, 3)),
            null_gate_heldout=(fails(base_rows, 0), null_held_fails),
            null_gate_regression=(fails(base_rows, 2), null_reg_fails),
            errors=errors, gate_only=args.gate_only,
        )

    stopped = None
    try:
        # Each fix is proposed before its family's baseline runs, so a failing
        # fixer costs only its own calls.
        for family, trig in fix_order(triggers, args.per_family):
            answer = propose(family, trig)
            if answer is None:
                continue
            result = evaluate(family, trig, answer)
            results[trig["key"]] = result
            print(json.dumps(result.record()))
    except BaseException as exc:
        stopped = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Written on every exit, so a run stopped at its cap still has one.
        final = list(results.values())
        bad, passing = fixloop.replay_false_pass(final)
        lo, hi = wilson(bad, passing)
        summary_path.write_text(json.dumps({
            "manifest_hash": manifest_hash,
            "stopped": stopped,
            "fixes": [r.record() for r in final],
            "replay_false_pass": {"bad": bad, "replay_passing_scored": passing, "wilson95": [lo, hi]},
            "gates": fixloop.gate_table(final),
            "fixer_failures": fixer_failures,
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
                     "label": rec["label"], "off_enum": rec.get("off_enum", []),
                     "verdicts": verdicts.get(key, {})})
    return rows


def detector_errors(run: str) -> dict[str, dict[str, int]]:
    """Errored detector calls, per detector and episode family."""
    path = RUNS / run / "detect.jsonl"
    counts: dict[str, dict[str, int]] = {}
    if path.exists():
        for row in Store(path).rows.values():
            if "error" in row:
                family = row["episode"].split(":", 1)[0]
                per = counts.setdefault(row["detector"], {})
                per[family] = per.get(family, 0) + 1
    return counts


def union(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """D2 or D3 flags: the $0 detectors together, for H1."""
    out = []
    for r in rows:
        v = r["verdicts"]
        if "d2" in v and "d3" in v:
            pick = v["d2"] if v["d2"]["flag"] else v["d3"]
            out.append({**r, "verdicts": {"d2|d3": {"flag": v["d2"]["flag"] or v["d3"]["flag"],
                                                     "family": pick.get("family")}}})
    return out


def cost_per_1k(ledger: Path, run: str, detector: str, traces: int) -> float | None:
    if not ledger.exists() or not traces:
        return None
    spent = sum(e["usd"] for e in ledger_rows(ledger) if e.get("run") == f"{run}-{detector}")
    return round(spent / traces * 1000, 4)


def spend_report(ledger: Path, run: str, agent_episodes: int, console_usd: float | None) -> None:
    """Booked spend per run tag, the mean agent cost per episode (the CP5
    cut), and, given the console's cost, the post-pilot dollar check."""
    tags = {run + suffix for suffix in RUN_TAG_SUFFIXES}
    rows = [e for e in ledger_rows(ledger) if e.get("run") in tags]
    for tag in sorted({e["run"] for e in rows}):
        t = ledger_totals([ledger], {tag})
        print(f"spend {tag:22s} ${t['usd']:.4f} over {t['requests']} ledger rows; "
              f"tokens in {t['input_tokens']:,}, out {t['output_tokens']:,}")
    if agent_episodes:
        mean = ledger_totals([ledger], {run})["usd"] / agent_episodes
        print(f"mean agent cost per episode: ${mean:.5f} (CP5 uses one trigger per family above $0.0055)")
    total = sum(e["usd"] for e in rows)
    print(f"booked total: ${total:.4f}")
    if console_usd is not None:
        ok = console_usd <= total * 1.10
        print(f"console ${console_usd:.2f} vs ledger ${total:.4f}: " + ("OK" if ok else "STOP: console above 1.10x the ledger"))


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
    errored = [k for k, r in Store(RUNS / args.run / "episodes.jsonl").rows.items() if "record" not in r]
    print(json.dumps(prevalence(rows), indent=1))
    print("induced prevalence (primary label = family, of scored induced episodes):")
    families_run = {key.split(":", 1)[0] for key in Store(RUNS / args.run / "episodes.jsonl").rows}
    for family, (hits, scored, below) in pilot_prevalence(rows, families_run).items():
        print(f"  {family:14s} {fmt(rate(hits, scored))}{'  BELOW 15%' if below else ''}")
    smoke_path = RUNS / "smoke.json"
    if smoke_path.exists():
        smoke = json.loads(smoke_path.read_text())
        attempts = smoke.get("attempts", [])
        print(f"smoke: {'confirmed' if smoke.get('confirmed') else 'NOT confirmed'}; {len(attempts)} attempt(s), "
              f"{sum(bool(a['problems']) for a in attempts)} with problems")
    print("positives per family (primary label = family, both variants):",
          {f: sum(r["label"] == f for r in rows) for f in FAMILIES})
    print("errored episodes (no label):", len(errored))
    retries = RUNS / args.run / "retries.jsonl"
    if retries.exists():
        entries = [json.loads(line) for line in retries.read_text().splitlines() if line.strip()]
        print("re-runs after an error (logged): episodes", sum("detector" not in e for e in entries),
              "| detector calls", sum("detector" in e for e in entries))
    off_enum: dict[str, dict[str, int]] = {}
    for r in rows:
        if r["off_enum"]:
            variant = "induced" if r["induced"] else "natural"
            per = off_enum.setdefault(variant, {})
            per[r["family"]] = per.get(r["family"], 0) + 1
    total_off = sum(sum(v.values()) for v in off_enum.values())
    print(f"episodes with words outside the report tool's enum (off_enum, not scored): {total_off}", off_enum)
    if total_off:
        print("STOP: off-enum words in the action list; a dated amendment decides how to read them")
    print("detector errors by family (excluded from the tables below):", detector_errors(args.run))
    for slice_name, sub in (("induced", [r for r in rows if r["induced"]]),
                            ("natural", [r for r in rows if not r["induced"]])):
        tables = {det: detector_table(sub, det) for det in ("d1", "d1b", "d2", "d3")}
        tables["d2|d3"] = detector_table(union(sub), "d2|d3")
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
            for fam, r in table["family_named_correctly_when_flagged"].items():
                print(f"  named right {fam:13s} {fmt(r)}")
            print(f"  not scored (aborted/unresolved/off_enum): {table['not_scored']}")
            for k, r in table["false_flags"].items():
                print(f"  false flags {k:9s} {fmt(r)}")
            if "judge_calls_share" in table:
                print(f"  judge calls needed   {table['judge_calls_share']:.0%} of traces")
    if args.ledger:
        traces = len(rows)
        for det in ("d1", "d1b"):
            print(f"{det}: ${cost_per_1k(Path(args.ledger), args.run, det, traces)} per 1,000 traces")
        agent_episodes = len(Store(RUNS / args.run / "episodes.jsonl").rows)
        spend_report(Path(args.ledger), args.run, agent_episodes, args.console_usd)


def _lookups(task: Task, k: int):
    def plan():
        yield scripted.Call("find_customer", {"email": task.customer_email})
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
    p = sub.add_parser("smoke")
    p.add_argument("--run", default="", help="required to run the smoke episode")
    p.add_argument("--gateway", default="")
    p.add_argument("--ledger", action="append", default=[], help="a gateway ledger (confirm: give every one)")
    p.add_argument("--console-tokens", default="", help="IN,OUT for the workspace from the console's Usage page")
    p.add_argument("--spend-limit", type=float, default=None, help="the workspace's console spend limit, USD")
    p.set_defaults(fn=cmd_smoke)
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
    p.add_argument("--retry-errors", action="store_true", help="re-run detector calls that errored (logged)")
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
    p.add_argument("--ledger", default="", help="gateway ledger: spend per tag, cost per 1,000 traces")
    p.add_argument("--console-usd", type=float, default=None, help="the console's cost for this run's workspace")
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
