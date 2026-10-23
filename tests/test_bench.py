"""$0 tests: gate, detectors, gateway caps and key handling, and the paid path
end to end against a loopback fake of the API."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

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


def test_fixloop_end_to_end_against_the_fake(gate_env, monkeypatch):
    env, upstream, tmp = gate_env
    server, url = _start(env, upstream, tmp / "spend.jsonl", 2.0)
    monkeypatch.chdir(tmp)
    from sfb import run as runner
    import sys

    try:
        for argv in (["episodes", "--run", "f", "--gateway", url, "--families", "wrong_entity",
                      "--seeds", "0-0", "--variants", "induced"],
                     ["fixloop", "--run", "f", "--gateway", url, "--families", "wrong_entity",
                      "--per-family", "1", "--workers", "4"]):
            monkeypatch.setattr(sys, "argv", ["sfb.run", *argv])
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
    with pytest.raises(SystemExit, match="sfb.gate"):
        runner.preflight()
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/gate_pass.json").write_text(json.dumps({"manifest_hash": "stale", "seeds": 40}))
    with pytest.raises(SystemExit, match="sfb.gate"):
        runner.preflight()
