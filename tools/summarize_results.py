"""Write results/summary.json from the local runs/ directory (gitignored).

Aggregates only: label counts, detector rates with Wilson intervals, the fix
loop's outcomes and gate table, and booked spend per checkpoint. No traces,
prompts or ledger rows leave runs/.

    .venv/bin/python tools/summarize_results.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfb.analysis import detector_table, rate  # noqa: E402
from sfb.run import ledger_totals, load_rows  # noqa: E402

RUNS = Path("runs")
LEDGERS = {"CP2 pilot + re-pilot": "pilot", "CP3 bench + controls": "cp3", "CP4 Haiku judges": "cp4",
           "CP4 Sonnet judge": "cp4-d1s", "CP4b judge + tool note": "cp4b", "CP5 fix loop": "cp5"}


def _no_nan(value):
    """Undefined rates (n = 0) as null, so the file is strict JSON."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _no_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_no_nan(v) for v in value]
    return value


def labels(run: str) -> dict[str, dict[str, int]]:
    out: dict[str, Counter] = {}
    for r in load_rows(run):
        out.setdefault(f"{r['family']}:{'induced' if r['induced'] else 'natural'}", Counter())[r["label"]] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


def main() -> None:
    bench, controls = load_rows("cp3"), load_rows("cp3-controls")
    positives = {fam: [r for r in bench if r["label"] == fam] for fam in ("loop", "partial")}
    clean = [r for r in bench + controls if not r["induced"] and r["label"] == "clean"]
    detectors = {}
    for det in ("d1", "d1s", "d1c", "d1b", "d2", "d3"):
        entry = {}
        for fam, rows in positives.items():
            judged = [r for r in rows if det in r["verdicts"]]
            if judged:
                entry[f"recall_{fam}"] = rate(sum(bool(r["verdicts"][det]["flag"]) for r in judged), len(judged))
        judged_clean = [r for r in clean if det in r["verdicts"]]
        entry["false_flags_natural_clean"] = rate(sum(bool(r["verdicts"][det]["flag"]) for r in judged_clean),
                                                  len(judged_clean))
        detectors[det] = entry
    fixloop = json.loads((RUNS / "cp3" / "fixloop_summary.json").read_text())
    records = [json.loads(line)["record"] for line in (RUNS / "cp3" / "episodes.jsonl").read_text().splitlines()
               if line.strip() and "record" in json.loads(line)]
    verdicts = [json.loads(line) for line in (RUNS / "cp3" / "detect.jsonl").read_text().splitlines() if line.strip()]
    final = {v["key"]: v for v in verdicts}  # the last row per key wins (retries)
    partial_keys = {r["key"] for r in bench if r["label"] == "partial"}
    cites_accepted = {}
    for det in ("d1", "d1s"):
        judged = [v for v in final.values() if v["episode"] in partial_keys and v["detector"] == det and "flag" in v]
        cites = sum("accepted" in (v.get("reasons") or [""])[0].lower() for v in judged)
        cites_accepted[det] = f"{cites}/{len(judged)}"
    loop_runs = [r for r in records if r["family"] == "loop"]
    loop_pos = [r for r in loop_runs if r["induced"] and r["label"] == "loop"]
    loop_clean = [r for r in loop_runs if not r["induced"] and r["label"] == "clean"]

    def mean(rows, field):
        return round(sum(field(r) for r in rows) / len(rows), 2) if rows else None

    def tokens(r):
        return r["usage"]["input_tokens"] + r["usage"]["output_tokens"]

    retry = {
        "loop_positives": len(loop_pos),
        "reported_pending_or_failed": sum("honest_failure" in r["labels"] for r in loop_pos),
        "model_calls": {"loop_positive": mean(loop_pos, lambda r: r["model_calls"]),
                        "same_task_service_up": mean(loop_clean, lambda r: r["model_calls"])},
        "tokens": {"loop_positive": mean(loop_pos, tokens), "same_task_service_up": mean(loop_clean, tokens)},
    }
    judged = {"d1": ("cp4", ["cp3-d1", "cp3-controls-d1"]), "d1b": ("cp4", ["cp3-d1b", "cp3-controls-d1b"]),
              "d1s": ("cp4-d1s", ["cp3-d1s", "cp3-controls-d1s"]), "d1c": ("cp4b", ["cp3-d1c", "cp3-controls-d1c"])}
    cost_per_1k = {}  # booked spend (retries included) / traces judged
    for det, (ledger, tags) in judged.items():
        for tag in tags:
            run = tag[: -len(det) - 1]
            rows = [json.loads(line) for line in (RUNS / run / "detect.jsonl").read_text().splitlines() if line.strip()]
            traces = len({r["episode"] for r in rows if r["detector"] == det})
            t = ledger_totals([RUNS / ledger / "spend.jsonl"], {tag})
            if traces:
                cost_per_1k[tag] = round(t["usd"] / traces * 1000, 2)
    spend = {name: round(ledger_totals([RUNS / tag / "spend.jsonl"])["usd"], 4) for name, tag in LEDGERS.items()}
    summary = {
        "labels": {"pilot": labels("pilot"), "re-pilot": labels("repilot"), "cp3": labels("cp3"),
                   "cp3-controls": labels("cp3-controls")},
        "detectors_cp3": detectors,
        "detector_tables_cp3_bench": {det: detector_table(bench, det) for det in ("d1", "d1s", "d1b", "d2", "d3")},
        "partial_verdicts_citing_accepted": cites_accepted,
        "retry_task": retry,
        "judge_cost_per_1000_traces_usd_as_booked": cost_per_1k,
        "fix_loop": {k: fixloop[k] for k in ("fixes", "replay_false_pass", "gates", "fixer_failures")},
        "spend_usd": spend | {"total": round(sum(spend.values()), 4)},
    }
    out = Path("results/summary.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(_no_nan(summary), indent=1, default=str, allow_nan=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
