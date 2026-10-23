# silent-failure-bench

Exactly labeled silent failures in a tool-using agent's traces, three ways to detect them, and a test of whether a fix that passes on the trace that triggered it holds up on new cases.

The failures counted here: the agent changed the wrong record, acted on an instruction the user had corrected, ignored how the user asked for something to be done, retried the same call over and over, reported something done that wasn't, or left part of a request undone without reporting it. What the agent reported is read from the ticket's action list, which the user sees; most of these failures are ones the user wouldn't catch.

## How it works

- **Environment.** A seeded support desk for a fictional airline: customers, reservations, refunds and eight tools. A scripted customer opens each conversation, so every task is reproducible (model replies still vary from run to run).
- **Failures are induced, labels are exact.** Natural silent failures from a capable model are rare, so each failure family has a condition that makes it likely (a near-duplicate reservation, a corrected date, a tool that queues instead of acting, and so on). Every episode is then labeled from the final state of the world, the environment's own call log and the ticket's action list, which is what the customer sees. No model is involved in labeling. An episode where the agent meets the condition and recovers is labeled clean, or honest_failure when the condition makes the request impossible and the agent reports that.
- **Natural slice.** The same tasks and seeds are also run without the condition, and reported separately.
- **Traces.** Episodes are traced with a vendor tracing SDK (MIT, pinned in `pyproject.toml`) and captured locally through its transport hook, so detectors see production-format traces. Nothing is sent anywhere.
- **Detectors.** An LLM judge, a set of LLM behavior checks, deterministic trace rules, and a probe that re-reads the records the agent claims to have changed.
- **Fix loop.** A fixer proposes a prompt or tool-description change from one triggering trace. The triggering task is re-run with the fix, then the fix is checked on held-out seeds of the same failure and on other tasks, paired against the unchanged agent. A null arm re-runs the triggering task, and the gate seeds, with no change, to show how often an empty fix would pass the same gates.

The design, metrics, hypotheses, budget and stop rules are in [preregistration.md](preregistration.md), frozen before any paid run. `manifest.json` hashes it together with the code that defines the measurement.

## Running

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest              # $0: gate, detectors, gateway, end-to-end against a local fake API
.venv/bin/python -m sfb.gate --seeds 40 # $0: scripted agents must get exactly the expected labels
.venv/bin/python -m sfb.run sizes       # $0: trace size against number of model calls
```

Paid runs go through a local gateway that holds the API key (from a gitignored `.env`), meters spend and refuses any request whose worst case could pass the cap. Paid commands also refuse to run unless the code matches `manifest.json` and the 40-seed gate has passed on it. Set `SFB_GATEWAY_TOKEN` to the same random string for both processes:

```bash
scripts/gateway.sh 1.00 runs/pilot/spend.jsonl &
scripts/harness.sh episodes --run pilot --gateway http://127.0.0.1:8765 --seeds 0-4
scripts/harness.sh detect --run pilot --gateway http://127.0.0.1:8765 --detectors d1,d1b,d2,d3
scripts/harness.sh report --run pilot --ledger runs/pilot/spend.jsonl   # --ledger adds cost per 1,000 traces
```

The harness can reach only loopback. The gateway can reach only the API host.
