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
def gate_env(tmp_path):
    env = tmp_path / ".env"
    env.write_text(f"ANTHROPIC_API_KEY={FAKE_KEY}\n")
    upstream = FakeUpstream()
    yield env, upstream, tmp_path
    upstream.close()


def _start(env, upstream, ledger, cap):
    server = gateway.serve(env, ledger, cap, 0, upstream.url)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def _post(url, body, headers=None):
    req = urllib.request.Request(url + "/v1/messages", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", **(headers or {})}, method="POST")
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
        status, reply = _post(url, body, {"x-api-key": "wrong", "authorization": "Bearer wrong"})
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


def test_gateway_rejects_unknown_models_and_streaming(gate_env):
    env, upstream, tmp = gate_env
    server, url = _start(env, upstream, tmp / "spend.jsonl", 1.0)
    try:
        assert _post(url, {"model": "claude-opus-5-5", "max_tokens": 10, "messages": []})[0] == 400
        assert _post(url, {"model": "claude-haiku-5-5", "max_tokens": 10, "stream": True, "messages": []})[0] == 400
    finally:
        server.shutdown()
    assert upstream.requests == []


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
    # The fake cancels the first listed reservation: right on false_success
    # natural (listed first), wrong on wrong_entity (a decoy is listed first).
    labels = {e["record"]["task"]: e["record"]["label"] for e in episodes}
    assert labels["false_success:0:n"] in ("clean", "incomplete_silent")
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
