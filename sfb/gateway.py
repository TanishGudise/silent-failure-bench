"""Loopback gateway: holds the API key, meters spend, enforces hard caps.

The harness talks only to http://127.0.0.1:<port>. The gateway is the one
process allowed to reach the upstream API. Before forwarding it reserves the
request's worst case (every request byte counted as an input token, plus
max_tokens of output) and refuses any request that could push spend past the
cap. After the response it books the real cost from `usage` and releases the
reservation. If the upstream connection fails after the request was sent, it
books the full reservation, since the call may have been billed. Error replies
without usage (429, 5xx) aren't billed; those are retried here, at most twice,
and every attempt is written to the ledger. The key is read from a gitignored
.env and never logged. Callers must present the gateway token, and only the
request fields the harness uses are forwarded.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# USD per million tokens: (input, output, input above the long-prompt threshold, output above it)
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50, 0.50, 2.50),
    "claude-sonnet-5-5": (2.00, 10.00, 2.00, 10.00),
}
LONG_PROMPT_TOKENS = 100_000
CACHE_WRITE, CACHE_READ = 1.25, 0.10
ALLOWED_FIELDS = {"model", "max_tokens", "system", "messages", "tools", "output_config",
                  "tool_choice", "stop_sequences", "metadata"}
RETRY_STATUSES = {429, 500, 502, 503, 529}
RETRY_DELAYS = (1.0, 4.0)
UPSTREAM_TIMEOUT = 300


def read_env_key(path: Path) -> str:
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"ANTHROPIC_API_KEY not found in {path}")


def rates(model: str, prompt_tokens: int) -> tuple[float, float]:
    base_in, base_out, long_in, long_out = PRICES[model]
    if prompt_tokens > LONG_PROMPT_TOKENS:
        return long_in, long_out
    return base_in, base_out


def worst_case_usd(model: str, body: bytes, max_tokens: int) -> float:
    prompt_bound = len(body)  # a token is at least one byte
    rate_in, rate_out = rates(model, prompt_bound)
    return (prompt_bound * rate_in + max_tokens * rate_out) / 1e6


def actual_usd(model: str, usage: dict) -> float:
    uncached = int(usage.get("input_tokens") or 0)
    written = int(usage.get("cache_creation_input_tokens") or 0)
    read = int(usage.get("cache_read_input_tokens") or 0)
    out = int(usage.get("output_tokens") or 0)
    rate_in, rate_out = rates(model, uncached + written + read)
    return (uncached * rate_in + written * rate_in * CACHE_WRITE + read * rate_in * CACHE_READ
            + out * rate_out) / 1e6


@dataclass
class Meter:
    ledger: Path
    cap_usd: float

    def __post_init__(self) -> None:
        self.lock = threading.Lock()
        self.reserved = 0.0
        self.spent = self._load()

    def _load(self) -> float:
        if not self.ledger.exists():
            return 0.0
        return sum(json.loads(line)["usd"] for line in self.ledger.read_text().splitlines() if line.strip())

    def reserve(self, usd: float) -> bool:
        with self.lock:
            if self.spent + self.reserved + usd > self.cap_usd:
                return False
            self.reserved += usd
            return True

    def settle(self, reserved: float, entry: dict) -> None:
        with self.lock:
            self.reserved -= reserved
            self.spent += entry["usd"]
            self._append(entry)

    def book(self, entry: dict) -> None:
        """Record an attempt without releasing the reservation (retries)."""
        with self.lock:
            self.spent += entry["usd"]
            self._append(entry)

    def _append(self, entry: dict) -> None:
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger, "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(entry) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle, fcntl.LOCK_UN)


def make_handler(meter: Meter, key: str, upstream: str, token: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:
            pass

        def _reply(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, kind: str, message: str) -> None:
            self._reply(status, json.dumps({"type": "error", "error": {"type": kind, "message": message}}).encode())

        def do_POST(self) -> None:
            if self.path.split("?")[0] != "/v1/messages":
                return self._error(404, "not_found_error", "only /v1/messages is proxied")
            presented = {self.headers.get("x-api-key", ""),
                         self.headers.get("authorization", "").removeprefix("Bearer ").strip()}
            if token not in presented:
                return self._error(401, "authentication_error", "gateway token required")
            if self.headers.get("anthropic-beta"):
                return self._error(400, "invalid_request_error", "beta headers are not forwarded")
            body = self.rfile.read(int(self.headers.get("content-length", 0)))
            try:
                request = json.loads(body)
                model = request["model"]
                max_tokens = int(request["max_tokens"])
            except (ValueError, KeyError):
                return self._error(400, "invalid_request_error", "unreadable request")
            extra = set(request) - ALLOWED_FIELDS
            if model not in PRICES or extra:
                return self._error(400, "invalid_request_error", f"model {model} or fields {sorted(extra)} not allowed")
            bound = worst_case_usd(model, body, max_tokens)
            if not meter.reserve(bound):
                return self._error(400, "invalid_request_error",
                                   f"budget cap reached (spent ${meter.spent:.4f} of ${meter.cap_usd:.2f})")
            headers = {"content-type": "application/json", "x-api-key": key,
                       "anthropic-version": self.headers.get("anthropic-version", "2023-06-01")}
            run = self.headers.get("x-sfb-run", "")
            for attempt in range(len(RETRY_DELAYS) + 1):
                started = time.time()
                try:
                    req = urllib.request.Request(upstream + "/v1/messages", data=body, headers=headers, method="POST")
                    try:
                        with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT) as resp:
                            status, payload = resp.status, resp.read()
                    except urllib.error.HTTPError as err:
                        status, payload = err.code, err.read()
                except OSError as exc:
                    # The request may have reached the API and been billed.
                    meter.settle(bound, {"ts": round(started, 3), "run": run, "model": model, "status": 0,
                                         "usage": {}, "usd": bound, "note": f"upstream failed: {type(exc).__name__}"})
                    return self._error(502, "api_error", f"upstream failed: {type(exc).__name__}")
                try:
                    usage = json.loads(payload).get("usage") or {}
                except ValueError:
                    usage = {}
                last = attempt == len(RETRY_DELAYS)
                retry = status in RETRY_STATUSES and not usage and not last
                if retry:
                    meter.book({"ts": round(started, 3), "run": run, "model": model, "status": status,
                                "usage": {}, "usd": 0.0, "note": "unbilled error reply; retrying"})
                    time.sleep(RETRY_DELAYS[attempt])
                    continue
                meter.settle(bound, {"ts": round(started, 3), "run": run, "model": model, "status": status,
                                     "usage": usage, "usd": actual_usd(model, usage) if usage else 0.0})
                return self._reply(status, payload)

    return Handler


def serve(env_path: Path, ledger: Path, cap_usd: float, port: int, upstream: str,
          token: str) -> ThreadingHTTPServer:
    if len(token) < 16:
        raise SystemExit("set SFB_GATEWAY_TOKEN to a random string of 16+ characters")
    meter = Meter(ledger, cap_usd)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(meter, read_env_key(env_path), upstream, token))
    server.meter = meter  # type: ignore[attr-defined]
    return server


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--cap", type=float, required=True, help="hard cap in USD for this ledger")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--upstream", default="https://api.anthropic.com")
    args = parser.parse_args()
    server = serve(args.env, args.ledger, args.cap, args.port, args.upstream,
                   os.environ.get("SFB_GATEWAY_TOKEN", ""))
    print(f"gateway on 127.0.0.1:{args.port}, spent ${server.meter.spent:.4f} of ${args.cap:.2f}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
