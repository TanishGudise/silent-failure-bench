"""Write results/summary.json from the local runs/ directory (gitignored).

Aggregates only: label counts, detector rates with Wilson intervals, the fix
loop's outcomes and gate table, and booked spend per checkpoint. No traces,
prompts or ledger rows leave runs/.

    .venv/bin/python tools/summarize_results.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfb.analysis import detector_table, rate  # noqa: E402
from sfb.run import ledger_totals, load_rows  # noqa: E402

RUNS = Path("runs")
LEDGERS = {"CP2 pilot + re-pilot": "pilot", "CP3 bench + controls": "cp3", "CP4 Haiku judges": "cp4",
           "CP4 Sonnet judge": "cp4-d1s", "CP4b judge + tool note": "cp4b", "CP5 fix loop": "cp5"}


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
    spend = {name: round(ledger_totals([RUNS / tag / "spend.jsonl"])["usd"], 4) for name, tag in LEDGERS.items()}
    summary = {
        "labels": {"pilot": labels("pilot"), "re-pilot": labels("repilot"), "cp3": labels("cp3"),
                   "cp3-controls": labels("cp3-controls")},
        "detectors_cp3": detectors,
        "detector_tables_cp3_bench": {det: detector_table(bench, det) for det in ("d1", "d1s", "d1b", "d2", "d3")},
        "fix_loop": {k: fixloop[k] for k in ("fixes", "replay_false_pass", "gates", "fixer_failures")},
        "spend_usd": spend | {"total": round(sum(spend.values()), 4)},
    }
    out = Path("results/summary.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
