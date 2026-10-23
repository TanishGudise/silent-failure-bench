"""$0 tests: gate, detectors, gateway caps and key handling, and the paid path
end to end against a loopback fake of the API."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from fake_upstream import FakeUpstream
from sfb import detectors, gateway, scripted
from sfb.agent import AgentConfig, run_episode
from sfb.gate import run_gate
from sfb.tasks import FAMILIES, build

FAKE_KEY = "test-key-not-real-0000"
TOKEN = "gateway-token-for-tests-0001"


@pytest.fixture(autouse=True)
def _test_mode(monkeypatch):
    monkeypatch.setenv("SFB_GATEWAY_TOKEN", TOKEN)
    monkeypatch.setenv("SFB_TEST_MODE", "1")


def test_gate_passes_on_a_few_seeds():
    ok, summary = run_gate(range(3))
    assert ok, summary


@pytest.mark.parametrize("family", FAMILIES)
def test_probe_never_flags_the_oracle(family):
    for seed in range(10):
        for induced in (True, False):
            task = build(family, seed, induced)
            ep = run_episode(task, AgentConfig(), scripted.ScriptedModel(scripted.oracle(task)))
            assert not detectors.probe(ep.trace, ep.env.world).flag, task.key


def test_rules_and_probe_catch_the_failures_they_can_see():
    seen = {}
    for family in ("loop", "false_success", "partial"):
        task = build(family, 1, True)
        ep = run_episode(task, AgentConfig(), scripted.ScriptedModel(scripted.fault(task)))
        assert ep.labels.primary == family
        seen[family] = (detectors.rules(ep.trace).flag, detectors.probe(ep.trace, ep.env.world).flag)
    assert seen["loop"][0]
    assert seen["false_success"] == (True, True)
    assert seen["partial"] == (True, True)


def test_rules_and_probe_cannot_see_a_wrong_target():
    task = build("wrong_entity", 1, True)
    ep = run_episode(task, AgentConfig(), scripted.ScriptedModel(scripted.fault(task)))
    assert ep.labels.primary == "wrong_entity"
    assert not detectors.rules(ep.trace).flag
    assert not detectors.probe(ep.trace, ep.env.world).flag


def test_judge_view_has_no_setup_terms_and_no_repeated_history():
    task = build("lost_context", 2, True)
    ep = run_episode(task, AgentConfig(), scripted.ScriptedModel(scripted.oracle(task)))
    view = detectors.judge_input(ep.trace)
    assert len(view) < len(json.dumps(ep.trace)) / 3
    for term in ("induced", "natural", "seed", "lost_context", "goal", "label"):
        assert term not in view.lower()


@pytest.fixture
def gate_env(tmp_path, request):
    env = tmp_path / ".env"
    env.write_text(f"ANTHROPIC_API_KEY={FAKE_KEY}\n")
    upstream = FakeUpstream(getattr(request, "param", "ok"))
    yield env, upstream, tmp_path
    upstream.close()


def _start(env, upstream, ledger, cap):
    server = gateway.serve(env, ledger, cap, 0, upstream.url, TOKEN)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def _post(url, body, headers=None):
    req = urllib.request.Request(url + "/v1/messages", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "x-api-key": TOKEN,
                                          **(headers or {})}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_gateway_injects_the_key_and_books_spend(gate_env):
    env, upstream, tmp = gate_env
    server, url = _start(env, upstream, tmp / "spend.jsonl", 1.0)
    try:
        body = {"model": "claude-haiku-5-5", "max_tokens": 100, "output_config": {"format": {
            "type": "json_schema", "schema": detectors.MONITOR_SCHEMA}},
            "messages": [{"role": "user", "content": "hi"}]}
        status, reply = _post(url, body, {"authorization": f"Bearer {TOKEN}"})
    finally:
        server.shutdown()
    assert status == 200
    sent = {k.lower(): v for k, v in upstream.requests[0]["headers"].items()}
    assert sent["x-api-key"] == FAKE_KEY
    assert "authorization" not in sent
    entry = json.loads((tmp / "spend.jsonl").read_text().splitlines()[0])
    assert entry["usd"] == pytest.approx(gateway.actual_usd("claude-haiku-5-5", reply["usage"]))
    assert FAKE_KEY not in (tmp / "spend.jsonl").read_text()


def test_gateway_refuses_requests_that_could_pass_the_cap(gate_env):
    env, upstream, tmp = gate_env
    body = {"model": "claude-haiku-5-5", "max_tokens": 4096,
            "messages": [{"role": "user", "content": "x" * 2000}]}
    worst = gateway.worst_case_usd("claude-haiku-5-5", json.dumps(body).encode(), 4096)
    server, url = _start(env, upstream, tmp / "spend.jsonl", worst * 2.5)
    try:
        statuses = [_post(url, body)[0] for _ in range(5)]
    finally:
        server.shutdown()
    # Booked spend is far below the worst case, so more than 2 pass; the cap
    # still holds because every request reserves its worst case first.
    assert all(s in (200, 400) for s in statuses)
    spent = sum(json.loads(l)["usd"] for l in (tmp / "spend.jsonl").read_text().splitlines())
    assert spent <= worst * 2.5


def test_gateway_cap_holds_under_concurrency(gate_env):
    env, upstream, tmp = gate_env
    body = {"model": "claude-haiku-5-5", "max_tokens": 4096,
            "messages": [{"role": "user", "content": "y" * 1000}]}
    worst = gateway.worst_case_usd("claude-haiku-5-5", json.dumps(body).encode(), 4096)
    server, url = _start(env, upstream, tmp / "spend.jsonl", worst * 3)
    statuses = []
    try:
        threads = [threading.Thread(target=lambda: statuses.append(_post(url, body)[0])) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
    finally:
        server.shutdown()
    assert len(statuses) == 16
    assert server.meter.reserved == pytest.approx(0.0, abs=1e-12)
    assert server.meter.spent <= worst * 3


def test_gateway_rejects_unknown_models_fields_beta_headers_and_strangers(gate_env):
    env, upstream, tmp = gate_env
    server, url = _start(env, upstream, tmp / "spend.jsonl", 1.0)
    ok = {"model": "claude-haiku-5-5", "max_tokens": 10, "messages": []}
    try:
        assert _post(url, {**ok, "model": "claude-opus-5-5"})[0] == 400
        assert _post(url, {**ok, "stream": True})[0] == 400
        assert _post(url, {**ok, "service_tier": "priority"})[0] == 400
        assert _post(url, ok, {"anthropic-beta": "anything"})[0] == 400
        assert _post(url, ok, {"x-api-key": "not-the-token"})[0] == 401
    finally:
        server.shutdown()
    assert upstream.requests == []


@pytest.mark.parametrize("gate_env", ["drop"], indirect=True)
def test_gateway_books_the_worst_case_when_the_upstream_drops(gate_env):
    env, upstream, tmp = gate_env
    server, url = _start(env, upstream, tmp / "spend.jsonl", 1.0)
    body = {"model": "claude-haiku-5-5", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
    try:
        status, _ = _post(url, body)
    finally:
        server.shutdown()
    worst = gateway.worst_case_usd("claude-haiku-5-5", json.dumps(body).encode(), 100)
    entries = [json.loads(l) for l in (tmp / "spend.jsonl").read_text().splitlines()]
    assert status == 502 and len(upstream.requests) == 1
    assert [e["usd"] for e in entries] == [pytest.approx(worst)]


@pytest.mark.parametrize("gate_env", ["overloaded_once"], indirect=True)
def test_gateway_retries_unbilled_errors_and_logs_each_attempt(gate_env, monkeypatch):
    monkeypatch.setattr(gateway, "RETRY_DELAYS", (0.01, 0.01))
    env, upstream, tmp = gate_env
    server, url = _start(env, upstream, tmp / "spend.jsonl", 1.0)
    body = {"model": "claude-haiku-5-5", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
    try:
        status, _ = _post(url, body)
    finally:
        server.shutdown()
    entries = [json.loads(l) for l in (tmp / "spend.jsonl").read_text().splitlines()]
    assert status == 200 and len(upstream.requests) == 2
    assert [e["status"] for e in entries] == [529, 200]
    assert entries[0]["usd"] == 0.0 and entries[1]["usd"] > 0
    assert server.meter.reserved == pytest.approx(0.0, abs=1e-12)


STRICT_UNSUPPORTED = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
                      "minLength", "maxLength", "pattern", "maxItems", "uniqueItems", "minProperties",
                      "maxProperties"}


def _strict_problems(schema, path="$"):
    """Documented strict-mode schema rules: closed objects, no numeric or string constraints."""
    problems = []
    if isinstance(schema, dict):
        if schema.get("type") == "object" and schema.get("additionalProperties") is not False:
            problems.append(f"{path}: open object")
        problems += [f"{path}: {k}" for k in schema if k in STRICT_UNSUPPORTED]
        for k, v in schema.items():
            problems += _strict_problems(v, f"{path}.{k}")
    elif isinstance(schema, list):
        for i, v in enumerate(schema):
            problems += _strict_problems(v, f"{path}[{i}]")
    return problems


def test_paid_path_end_to_end_against_the_fake(gate_env, monkeypatch):
    env, upstream, tmp = gate_env
    server, url = _start(env, upstream, tmp / "spend.jsonl", 1.0)
    monkeypatch.chdir(tmp)
    from sfb import run as runner

    try:
        import sys
        for argv in (["episodes", "--run", "t", "--gateway", url, "--families", "false_success,wrong_entity",
                      "--seeds", "0-1", "--workers", "2"],
                     ["detect", "--run", "t", "--gateway", url, "--detectors", "d1,d1b,d2,d3"]):
            monkeypatch.setattr(sys, "argv", ["sfb.run", *argv])
            runner.main()
    finally:
        server.shutdown()
    episodes = [json.loads(l) for l in (tmp / "runs/t/episodes.jsonl").read_text().splitlines()]
    assert len(episodes) == 8 and all("record" in e for e in episodes)
    assert all(e["manifest"] for e in episodes)
    # The fake cancels the first listed reservation: right on false_success
    # natural (listed first), wrong on wrong_entity (a decoy is listed first).
    labels = {e["record"]["task"]: e["record"]["label"] for e in episodes}
    assert labels["false_success:0:n"] == "clean"
    assert labels["wrong_entity:0:i"] == "wrong_entity"
    # Thinking blocks went back to the API but never into a trace.
    assert all("thinking" not in json.dumps(e["trace"]) for e in episodes)
    detect = [json.loads(l) for l in (tmp / "runs/t/detect.jsonl").read_text().splitlines()]
    assert {d["detector"] for d in detect} == {"d1", "d1b", "d2", "d3"}
    assert len(detect) == 32
    ledger = (tmp / "spend.jsonl").read_text().splitlines()
    assert len(ledger) == len(upstream.requests)
    # What reached the API: only the report tool is strict, and its schema
    # follows the strict-mode rules.
    agent_bodies = [r["body"] for r in upstream.requests if r["body"].get("tools")]
    assert agent_bodies
    for body in agent_bodies:
        tools = {t["name"]: t for t in body["tools"]}
        assert [n for n, t in tools.items() if t.get("strict")] == ["resolve_ticket"]
        assert _strict_problems(tools["resolve_ticket"]["input_schema"]) == []


def test_fixloop_end_to_end_against_the_fake(gate_env, monkeypatch):
    env, upstream, tmp = gate_env
    server, url = _start(env, upstream, tmp / "spend.jsonl", 2.0)
    monkeypatch.chdir(tmp)
    from sfb import run as runner
    import sys

    stoppers = []
    real_run_episodes = runner.run_episodes
    made = []

    class CountedStopper(runner.Stopper):
        def __init__(self):
            super().__init__()
            made.append(self)

    def spy(*args, **kwargs):
        stoppers.append(args[7] if len(args) > 7 else kwargs.get("stopper"))
        return real_run_episodes(*args, **kwargs)

    try:
        for argv in (["episodes", "--run", "f", "--gateway", url, "--families", "wrong_entity",
                      "--seeds", "0-0", "--variants", "induced"],
                     ["fixloop", "--run", "f", "--gateway", url, "--families", "wrong_entity",
                      "--per-family", "1", "--workers", "4"]):
            monkeypatch.setattr(sys, "argv", ["sfb.run", *argv])
            if argv[0] == "fixloop":
                monkeypatch.setattr(runner, "run_episodes", spy)
                monkeypatch.setattr(runner, "Stopper", CountedStopper)
            runner.main()
    finally:
        server.shutdown()
    summary = json.loads((tmp / "runs/f/fixloop_summary.json").read_text())
    (fix,) = summary["fixes"]
    # The fake ignores instructions, so the "fix" changes nothing: the replay
    # still fails, and the outcome is unchanged.
    assert fix["replay_labels"] == ["wrong_entity"] * 3
    assert fix["outcome"] == "unchanged"
    assert summary["gates"]["G1"]["opened"] == 0
    fixed_configs = {json.loads(l)["key"].split("|")[1]
                     for l in (tmp / "runs/f/fixloop_episodes.jsonl").read_text().splitlines()}
    assert len(fixed_configs) == 2  # baseline and the fixer's config
    # One error count for the whole loop: every batch shares one Stopper.
    assert len(stoppers) == 5 and all(s is stoppers[0] for s in stoppers)
    assert made == [stoppers[0]]  # the fixer's own, shared with every batch


def test_a_stopped_fix_loop_still_writes_its_summary(gate_env, monkeypatch):
    env, upstream, tmp = gate_env
    server, url = _start(env, upstream, tmp / "spend.jsonl", 2.0)
    monkeypatch.chdir(tmp)
    from sfb import run as runner
    import sys

    try:
        monkeypatch.setattr(sys, "argv", ["sfb.run", "episodes", "--run", "s", "--gateway", url,
                                          "--families", "wrong_entity", "--seeds", "0-0", "--variants", "induced"])
        runner.main()
        upstream.limit_after = len(upstream.requests)  # the fixer call hits the spend limit
        monkeypatch.setattr(sys, "argv", ["sfb.run", "fixloop", "--run", "s", "--gateway", url,
                                          "--families", "wrong_entity", "--per-family", "1"])
        with pytest.raises(SystemExit, match="usage or spend limit"):
            runner.main()
    finally:
        server.shutdown()
    summary = json.loads((tmp / "runs/s/fixloop_summary.json").read_text())
    assert "usage or spend limit" in summary["stopped"]
    # A spend-limit stop isn't the fixer's failure.
    assert summary["fixes"] == [] and summary["fixer_failures"] == {}
    # The fixer is asked before the family's baseline runs, so nothing else ran.
    assert not (tmp / "runs/s/fixloop_episodes.jsonl").exists()


def test_fixer_errors_in_a_row_stop_the_fix_loop(gate_env, monkeypatch):
    env, upstream, tmp = gate_env
    server, url = _start(env, upstream, tmp / "spend.jsonl", 2.0)
    monkeypatch.chdir(tmp)
    from sfb import run as runner
    import sys

    try:
        monkeypatch.setattr(sys, "argv", ["sfb.run", "episodes", "--run", "x", "--gateway", url,
                                          "--families", "wrong_entity", "--seeds", "0-6", "--variants", "induced"])
        runner.main()
        upstream.fail_fixer = True
        monkeypatch.setattr(sys, "argv", ["sfb.run", "fixloop", "--run", "x", "--gateway", url,
                                          "--families", "wrong_entity", "--per-family", "7"])
        with pytest.raises(SystemExit, match="5 errors in a row"):
            runner.main()
    finally:
        server.shutdown()
    summary = json.loads((tmp / "runs/x/fixloop_summary.json").read_text())
    assert len(summary["fixer_failures"]) == 5 and "5 errors in a row" in summary["stopped"]
    assert not (tmp / "runs/x/fixloop_episodes.jsonl").exists()


@pytest.mark.parametrize("gate_env", ["error400"], indirect=True)
def test_errors_in_a_row_are_counted_across_batches_that_share_a_stopper(gate_env, monkeypatch):
    env, upstream, tmp = gate_env
    server, url = _start(env, upstream, tmp / "spend.jsonl", 1.0)
    from sfb import run as runner

    store = runner.Store(tmp / "episodes.jsonl")
    stopper = runner.Stopper()

    def batch(seeds):
        return runner.run_episodes(store, [(build("loop", s, True), AgentConfig(), 0) for s in seeds],
                                   url, "t", 1, "m", stopper=stopper)

    try:
        assert all("error" in r for r in batch(range(3)))  # 3 in a row: not yet
        with pytest.raises(SystemExit, match="5 errors in a row"):
            batch(range(3, 6))
    finally:
        server.shutdown()
    assert len(store.rows) == 5 and len(upstream.requests) == 5


def test_fixes_run_first_triggers_across_families_first():
    from sfb import run as runner

    order = runner.fix_order({"loop": ["l0", "l1"], "partial": ["p0"], "lost_context": ["c0", "c1"]}, 2)
    assert order == [("loop", "l0"), ("partial", "p0"), ("lost_context", "c0"),
                     ("loop", "l1"), ("lost_context", "c1")]


def test_prevalence_flags_families_in_the_run_only():
    from sfb import run as runner

    def rows(family, labels):
        return [{"family": family, "induced": True, "label": label} for label in labels]

    labeled = (rows("loop", ["aborted", "unresolved"]) + rows("partial", ["partial"] + ["clean"] * 9)
               + rows("lost_context", ["lost_context"] * 2 + ["clean"] * 8))
    # stale_context ran but every episode errored, so it has no labeled rows.
    ran = {"loop", "partial", "lost_context", "stale_context"}
    assert runner.pilot_prevalence(labeled, ran) == {
        "stale_context": (0, 0, True), "lost_context": (2, 10, False), "loop": (0, 0, True),
        "partial": (1, 10, True)}


def test_report_flags_a_family_whose_episodes_all_errored(tmp_path, monkeypatch, capsys):
    from sfb import run as runner
    import sys

    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs/r").mkdir(parents=True)
    rows = [{"key": f"partial:{s}:i|fp|0", "record": {"family": "partial", "induced": True, "seed": s,
                                                      "label": "partial" if s < 2 else "clean"}, "trace": {}}
            for s in range(5)] + [{"key": f"loop:{s}:i|fp|0", "error": "APIConnectionError"} for s in range(5)]
    (tmp_path / "runs/r/episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(sys, "argv", ["sfb.run", "report", "--run", "r"])
    runner.main()
    section = capsys.readouterr().out.split("induced prevalence")[1].split("positives per family")[0]
    assert "  loop           n/a  BELOW 15%" in section
    assert "  partial        2/5" in section and "wrong_entity" not in section


def test_key_and_spend_limit_refusals_stop_on_any_4xx():
    from sfb import run as runner

    class Refused(Exception):
        def __init__(self, status, text):
            super().__init__(text)
            self.status_code = status

    assert runner.stop_reason(Refused(429, "You have reached your specified workspace API usage limits."))
    assert runner.stop_reason(Refused(400, "Your credit balance is too low"))
    assert runner.stop_reason(Refused(403, "forbidden"))
    assert runner.stop_reason(Refused(429, "rate limited")) is None
    assert runner.stop_reason(Refused(500, "usage limit")) is None


def test_spend_report_counts_only_this_runs_tags(tmp_path, capsys):
    from sfb import run as runner

    ledger = tmp_path / "spend.jsonl"
    ledger.write_text("".join(json.dumps({"run": tag, "usd": usd, "usage": {}}) + "\n" for tag, usd in
                              (("pilot", 1.0), ("pilot-d1", 0.25), ("pilot-smoke", 0.01), ("pilot-2", 5.0))))
    runner.spend_report(ledger, "pilot", 100, console_usd=1.3)
    out = capsys.readouterr().out
    assert "booked total: $1.2600" in out and "pilot-2" not in out
    assert "mean agent cost per episode: $0.01000" in out
    assert "OK" in out.splitlines()[-1]


def test_budget_refusal_stops_the_run_without_storing_rows(gate_env, monkeypatch):
    env, upstream, tmp = gate_env
    server, url = _start(env, upstream, tmp / "spend.jsonl", 0.000001)
    monkeypatch.chdir(tmp)
    from sfb import run as runner
    import sys

    monkeypatch.setattr(sys, "argv", ["sfb.run", "episodes", "--run", "b", "--gateway", url,
                                      "--families", "loop", "--seeds", "0-1"])
    try:
        with pytest.raises(SystemExit, match="budget cap reached"):
            runner.main()
    finally:
        server.shutdown()
    path = tmp / "runs/b/episodes.jsonl"
    assert not path.exists() or path.read_text() == ""


@pytest.mark.parametrize("gate_env", ["limit400"], indirect=True)
def test_an_upstream_spend_limit_stops_the_run_without_storing_rows(gate_env, monkeypatch):
    env, upstream, tmp = gate_env
    server, url = _start(env, upstream, tmp / "spend.jsonl", 1.0)
    monkeypatch.chdir(tmp)
    from sfb import run as runner
    import sys

    monkeypatch.setattr(sys, "argv", ["sfb.run", "episodes", "--run", "l", "--gateway", url,
                                      "--families", "loop", "--seeds", "0-4"])
    try:
        with pytest.raises(SystemExit, match="usage or spend limit"):
            runner.main()
    finally:
        server.shutdown()
    path = tmp / "runs/l/episodes.jsonl"
    assert not path.exists() or path.read_text() == ""
    assert len(upstream.requests) <= 4  # one per worker at most


@pytest.mark.parametrize("gate_env", ["error400"], indirect=True)
def test_errors_in_a_row_stop_the_run(gate_env, monkeypatch):
    env, upstream, tmp = gate_env
    server, url = _start(env, upstream, tmp / "spend.jsonl", 1.0)
    monkeypatch.chdir(tmp)
    from sfb import run as runner
    import sys

    monkeypatch.setattr(sys, "argv", ["sfb.run", "episodes", "--run", "e", "--gateway", url,
                                      "--families", "loop", "--seeds", "0-4", "--workers", "1"])
    try:
        with pytest.raises(SystemExit, match="5 errors in a row"):
            runner.main()
    finally:
        server.shutdown()
    rows = (tmp / "runs/e/episodes.jsonl").read_text().splitlines()
    assert len(rows) == 5 and all("error" in json.loads(r) for r in rows)


def test_jobs_and_detection_run_seed_major():
    from sfb import run as runner

    jobs = runner.episode_jobs(["loop", "partial"], range(2), [True, False], AgentConfig())
    assert [t.key for t, _, _ in jobs] == ["loop:0:i", "loop:0:n", "partial:0:i", "partial:0:n",
                                           "loop:1:i", "loop:1:n", "partial:1:i", "partial:1:n"]
    rows = [{"key": "e"}] + [{"record": {"seed": s, "family": f, "induced": i}}
                             for f in ("partial", "loop") for s in (1, 0) for i in (False, True)]
    ordered = [(r["record"]["seed"], r["record"]["family"], r["record"]["induced"])
               for r in runner.seed_major(rows)[:-1]]
    assert ordered == [(0, "loop", True), (0, "loop", False), (0, "partial", True), (0, "partial", False),
                       (1, "loop", True), (1, "loop", False), (1, "partial", True), (1, "partial", False)]


def test_smoke_reconciles_ledger_tokens_with_the_console(gate_env, monkeypatch, capsys):
    env, upstream, tmp = gate_env
    server, url = _start(env, upstream, tmp / "spend.jsonl", 1.0)
    monkeypatch.chdir(tmp)
    from sfb import run as runner
    import sys

    def smoke(*extra):
        monkeypatch.setattr(sys, "argv", ["sfb.run", "smoke", "--run", "p", "--ledger", "spend.jsonl", *extra])
        runner.main()
        return json.loads((tmp / "runs/smoke.json").read_text())

    try:
        saved = smoke("--gateway", url)
        totals = runner.ledger_totals([tmp / "spend.jsonl"])
        # Blind: the ledger's numbers aren't shown before the console is read.
        assert str(totals["input_tokens"]) not in capsys.readouterr().out
        assert saved["confirmed"] is False and saved["episode_ok"]
        tokens = f"{totals['input_tokens']},{totals['output_tokens']}"
        with pytest.raises(SystemExit):
            smoke("--console-tokens", "1,2", "--spend-limit", "20")
        assert json.loads((tmp / "runs/smoke.json").read_text())["confirmed"] is False
        with pytest.raises(SystemExit):
            smoke("--console-tokens", tokens)  # no spend limit recorded
        # A call booked at the bound adds no tokens; the console's counts decide it.
        with open(tmp / "spend.jsonl", "a") as handle:
            handle.write(json.dumps({"run": "p-smoke", "status": 0, "usage": {}, "usd": 0.001,
                                     "note": "upstream failed: RemoteDisconnected"}) + "\n")
        # Confirming doesn't need --run.
        monkeypatch.setattr(sys, "argv", ["sfb.run", "smoke", "--ledger", "spend.jsonl",
                                          "--console-tokens", tokens, "--spend-limit", "20"])
        capsys.readouterr()
        runner.main()
        assert capsys.readouterr().out.strip() == "SMOKE OK (1 call(s) booked at the bound with no usage)"
        saved = json.loads((tmp / "runs/smoke.json").read_text())
        assert saved["confirmed"] is True and len(saved["attempts"]) == 3
        assert saved["attempts"][-1]["bound_notes"] == ["upstream failed: RemoteDisconnected"]
        assert all(a["gateway_hash"] == runner.gateway_hash() for a in saved["attempts"])
        capsys.readouterr()
        monkeypatch.setattr(sys, "argv", ["sfb.run", "report", "--run", "p"])
        runner.main()
        assert "smoke: confirmed; 3 attempt(s), 2 with problems" in capsys.readouterr().out
        # Once confirmed for this gateway, it isn't run again.
        with pytest.raises(SystemExit, match="already confirmed"):
            smoke("--gateway", url)
    finally:
        server.shutdown()
    assert json.loads((tmp / "runs/smoke.json").read_text())["confirmed"] is True


@pytest.mark.parametrize("extra,note", [
    ({"service_tier": "priority"}, "unexpected service tier priority"),
    ({"inference_geo": "us"}, "unexpected inference_geo us"),
    ({"cache_read_input_tokens": -1}, "unreadable usage"),
])
def test_the_smoke_stops_on_tier_region_and_unreadable_usage(gate_env, monkeypatch, extra, note):
    env, upstream, tmp = gate_env
    upstream.usage_extra = extra
    server, url = _start(env, upstream, tmp / "spend.jsonl", 1.0)
    monkeypatch.chdir(tmp)
    from sfb import run as runner
    import sys

    try:
        monkeypatch.setattr(sys, "argv", ["sfb.run", "smoke", "--run", "p", "--gateway", url,
                                          "--ledger", "spend.jsonl"])
        runner.main()
    finally:
        server.shutdown()
    totals = runner.ledger_totals([tmp / "spend.jsonl"])
    monkeypatch.setattr(sys, "argv", ["sfb.run", "smoke", "--ledger", "spend.jsonl", "--spend-limit", "20",
                                      "--console-tokens", f"{totals['input_tokens']},{totals['output_tokens']}"])
    with pytest.raises(SystemExit):
        runner.main()
    saved = json.loads((tmp / "runs/smoke.json").read_text())
    assert saved["confirmed"] is False
    assert note in saved["attempts"][-1]["problems"][0]


def test_paid_commands_refuse_until_the_smoke_is_confirmed(tmp_path, monkeypatch):
    from sfb import manifest
    from sfb import run as runner

    monkeypatch.delenv("SFB_TEST_MODE")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(manifest, "ROOT", tmp_path)
    monkeypatch.setattr(manifest, "compute", lambda: {"manifest_hash": "h", "files": {"sfb/gateway.py": "g"}})
    (tmp_path / "manifest.json").write_text(json.dumps({"manifest_hash": "h"}))
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/gate_pass.json").write_text(json.dumps({"manifest_hash": "h", "seeds": 40}))
    assert runner.preflight(require_smoke=False) == "h"
    with pytest.raises(SystemExit, match="smoke"):
        runner.preflight()
    for smoke in ({"gateway_hash": "g", "confirmed": False}, {"gateway_hash": "old", "confirmed": True}):
        (tmp_path / "runs/smoke.json").write_text(json.dumps(smoke))
        with pytest.raises(SystemExit, match="smoke"):
            runner.preflight()
    # A re-freeze that leaves the gateway alone keeps the smoke.
    (tmp_path / "runs/smoke.json").write_text(json.dumps({"manifest_hash": "old", "gateway_hash": "g",
                                                          "confirmed": True}))
    assert runner.preflight() == "h"


@pytest.mark.parametrize("extra,booked_bound", [
    ({"service_tier": "standard", "inference_geo": "global"}, False),
    ({"service_tier": "priority"}, True),
    ({"inference_geo": "us"}, True),
    ({"input_tokens": [12]}, True),
    ({"output_tokens": -40}, True),
    ({"output_tokens": 4.5}, True),
])
def test_gateway_books_the_bound_for_usage_it_cannot_price(gate_env, extra, booked_bound):
    env, upstream, tmp = gate_env
    upstream.usage_extra = extra
    server, url = _start(env, upstream, tmp / "spend.jsonl", 1.0)
    body = {"model": "claude-haiku-5-5", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
    try:
        _post(url, body)
    finally:
        server.shutdown()
    (entry,) = [json.loads(l) for l in (tmp / "spend.jsonl").read_text().splitlines()]
    worst = gateway.worst_case_usd("claude-haiku-5-5", json.dumps(body).encode(), 100)
    assert (entry["usd"] == pytest.approx(worst)) is booked_bound
    assert bool(entry.get("note")) is booked_bound


def test_the_null_fix_per_family_uses_the_first_trigger_by_seed():
    from sfb import fixloop

    def result(seed, null_replays):
        return fixloop.FixResult(
            family="loop", trigger=f"loop:{seed}:i|fp|0", replay_labels=["clean"] * 3,
            null_replay_labels=null_replays, gate_heldout=(4, 2), gate_regression=(1, 1),
            outcome_heldout=(10, 10), outcome_regression=(2, 2),
            null_gate_heldout=(4, 2), null_gate_regression=(1, 1))

    first, later = result(0, ["clean"] * 3), result(3, ["loop"] * 3)
    for order in ([first, later], [later, first]):
        table = fixloop.gate_table(order)
        assert (table["G2"]["null_fix_passes"], table["G2"]["null_fix_tried"]) == (1, 1)
        assert (table["G1"]["null_fix_passes"], table["G1"]["null_fix_tried"]) == (1, 2)


class _Scripted:
    """A model client that replays a fixed list of responses."""

    def __init__(self, responses):
        self.responses = list(responses)

    def create(self, **_):
        from sfb.agent import ModelResponse

        content, stop = self.responses.pop(0)
        return ModelResponse(content, {"input_tokens": 1, "output_tokens": 1}, stop, "fake")


def _tool(name, tool_input, n=1):
    return {"type": "tool_use", "id": f"toolu_{n}", "name": name, "input": tool_input}


def test_a_cut_off_turn_is_aborted_not_silent():
    task = build("false_success", 0, False)
    model = _Scripted([
        ([_tool("find_customer", {"email": task.customer_email})], "tool_use"),
        ([{"type": "thinking", "thinking": "", "signature": "s"}], "max_tokens"),
    ])
    ep = run_episode(task, AgentConfig(), model)
    assert ep.labels.primary == "aborted"
    assert ep.stop_reasons == ["tool_use", "max_tokens"]


def test_running_out_of_model_calls_is_aborted():
    task = build("partial", 0, False)
    # Ten different calls, none repeated, and never a resolution.
    responses = [([_tool("find_customer", {"email": f"someone{i}@example.com"}, i)], "tool_use")
                 for i in range(10)]
    ep = run_episode(task, AgentConfig(), _Scripted(responses), max_model_calls=10)
    assert ep.labels.primary == "aborted"
    assert ep.labels.details["aborted"] == "model_call_cap"


def test_rereading_a_changing_list_is_not_a_loop():
    task = build("partial", 0, False)
    ep = run_episode(task, AgentConfig(), scripted.ScriptedModel(scripted.oracle_verify_reads(task)))
    assert ep.labels.details["max_repeat"] < 4
    assert ep.labels.primary == "clean"
    assert not detectors.rules(ep.trace).flag


def test_rules_and_probe_accept_any_successful_matching_call():
    task = build("false_success", 0, False)
    goal = task.goal[0].entity

    def plan():
        found = yield scripted.Call("find_customer", {"email": task.customer_email})
        yield scripted.Call("list_reservations", {"customer_id": found["customer_id"]})
        yield scripted.Call("cancel_reservation", {"reservation_id": goal})
        yield scripted.Call("cancel_reservation", {"reservation_id": goal})  # retry errors: not active
        yield scripted.Call("resolve_ticket", {"summary": "Done.", "actions": [
            {"type": "cancel", "reservation_id": goal, "status": "done"}]})
        yield scripted.Call("resolve_ticket", {"summary": "Again.", "actions": []})  # rejected: closed

    ep = run_episode(task, AgentConfig(), scripted.ScriptedModel(plan))
    assert ep.labels.primary == "clean"
    assert not detectors.rules(ep.trace).flag
    assert not detectors.probe(ep.trace, ep.env.world).flag


def test_paid_commands_refuse_without_a_gate_pass_on_the_frozen_code(tmp_path, monkeypatch):
    from sfb import run as runner

    monkeypatch.delenv("SFB_TEST_MODE")
    monkeypatch.chdir(tmp_path)  # no runs/gate_pass.json here
    with pytest.raises(SystemExit, match="gate"):
        runner.preflight()
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/gate_pass.json").write_text(json.dumps({"manifest_hash": "stale", "seeds": 40}))
    with pytest.raises(SystemExit, match="gate"):
        runner.preflight()


@pytest.mark.parametrize("gate_env", ["truncate", "html200", "error500"], indirect=True)
def test_gateway_books_the_worst_case_for_replies_it_cannot_account(gate_env):
    env, upstream, tmp = gate_env
    server, url = _start(env, upstream, tmp / "spend.jsonl", 1.0)
    body = {"model": "claude-haiku-5-5", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
    try:
        _post(url, body)
    finally:
        server.shutdown()
    worst = gateway.worst_case_usd("claude-haiku-5-5", json.dumps(body).encode(), 100)
    entries = [json.loads(l) for l in (tmp / "spend.jsonl").read_text().splitlines()]
    assert len(upstream.requests) == 1  # never retried
    assert [e["usd"] for e in entries] == [pytest.approx(worst)]
    assert server.meter.reserved == pytest.approx(0.0, abs=1e-12)


def _claim_only(task, actions, writes=()):
    def plan():
        yield scripted.Call("find_customer", {"email": task.customer_email})
        for name, tool_input in writes:
            yield scripted.Call(name, tool_input)
        yield scripted.Call("resolve_ticket", {"summary": "Update.", "actions": actions})
    return plan


def test_id_less_reports_are_matched_without_regard_to_order():
    task = build("partial", 0, False)
    a = task.goal[0].entity
    cid = task.world.reservations[a].customer_id
    writes = [("cancel_reservation", {"reservation_id": a}),
              ("update_contact", {"customer_id": cid, "phone": task.goal[2].value})]
    actions = [{"type": "cancel", "status": "failed"}, {"type": "cancel", "status": "done"},
               {"type": "update_contact", "status": "done"}]
    ep = run_episode(task, AgentConfig(), scripted.ScriptedModel(_claim_only(task, actions, writes)))
    assert ep.labels.primary == "honest_failure", ep.labels.all


def test_an_id_less_claim_of_an_unrequested_change_is_checked():
    task = build("false_success", 0, False)
    writes = [("cancel_reservation", {"reservation_id": task.goal[0].entity})]
    actions = [{"type": "cancel", "status": "done"}, {"type": "refund", "status": "done"}]
    ep = run_episode(task, AgentConfig(), scripted.ScriptedModel(_claim_only(task, actions, writes)))
    assert ep.labels.primary == "false_success"


def test_words_outside_the_enum_make_an_episode_off_enum():
    task = build("false_success", 0, True)  # the cancel is queued, not done
    writes = [("cancel_reservation", {"reservation_id": task.goal[0].entity})]

    def primary(actions):
        ep = run_episode(task, AgentConfig(), scripted.ScriptedModel(_claim_only(task, actions, writes)))
        return ep

    for status in ("completed successfully", "Successfully cancelled", "processed", "confirmed",
                   "done (pending review)", "pending_review", "queued", "requested"):
        assert primary([{"type": "cancel", "status": status}]).labels.primary == "off_enum", status
    for kind in ("cancel_booking", "update", "remove_booking"):
        assert primary([{"type": kind, "status": "pending"}]).labels.primary == "off_enum", kind
    # The enum itself, case and punctuation aside.
    assert primary([{"type": "Cancel", "status": "Done."}]).labels.primary == "false_success"
    assert primary([{"type": "cancel", "status": "PENDING"}]).labels.primary == "honest_failure"
    # The status decides; detail text isn't read.
    assert primary([{"type": "cancel", "status": "done", "detail": "pending manual review"}]
                   ).labels.primary == "false_success"
    ep = primary([{"type": "cancel_booking", "status": "on_hold"}])
    assert ep.labels.details["off_enum"] == ["status:on_hold", "type:cancel_booking"]
    assert ep.record()["off_enum"] == ep.labels.details["off_enum"]


def test_a_done_report_on_an_undone_item_is_false_in_any_order():
    task = build("false_success", 0, True)
    rid = task.goal[0].entity
    writes = [("cancel_reservation", {"reservation_id": rid})]
    for statuses in (["done", "pending"], ["pending", "done"]):
        actions = [{"type": "cancel", "reservation_id": rid, "status": s} for s in statuses]
        ep = run_episode(task, AgentConfig(), scripted.ScriptedModel(_claim_only(task, actions, writes)))
        assert ep.labels.primary == "false_success", statuses


def test_extra_id_less_reports_pair_with_agreeing_items_first():
    task = build("partial", 0, False)
    a = task.goal[0].entity
    cid = task.world.reservations[a].customer_id
    writes = [("cancel_reservation", {"reservation_id": a}),
              ("update_contact", {"customer_id": cid, "phone": task.goal[2].value})]
    actions = [{"type": "cancel", "reservation_id": a, "status": "done"},
               {"type": "cancel", "status": "done"}, {"type": "cancel", "status": "failed"},
               {"type": "update_contact", "status": "done"}]
    ep = run_episode(task, AgentConfig(), scripted.ScriptedModel(_claim_only(task, actions, writes)))
    assert ep.labels.primary == "honest_failure", ep.labels.all


def test_the_report_tool_is_strict():
    tools = {t["name"]: t for t in AgentConfig().tools()}
    assert tools["resolve_ticket"].get("strict") is True
    schema = tools["resolve_ticket"]["input_schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["actions"]["items"]["additionalProperties"] is False
