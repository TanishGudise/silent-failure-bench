"""Loopback stand-in for the Messages API, for $0 end-to-end tests.

Agent requests (with tools) get a fixed tool sequence: find the customer,
list reservations, cancel the first one, resolve the ticket. Structured
output requests get a valid answer for whichever schema they carry.
Every request is recorded, headers included, so tests can check what the
gateway sent upstream.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeUpstream:
    """mode: "ok"; "drop" (read the request, close without replying);
    "overloaded_once" (529 with no usage, then normal replies)."""

    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        handler = self._handler()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                with fake.lock:
                    fake.requests.append({"headers": dict(self.headers), "body": body})
                    first = len(fake.requests) == 1
                if fake.mode == "drop":
                    self.close_connection = True
                    self.connection.close()
                    return
                status, reply = 200, fake.respond(body)
                if fake.mode == "overloaded_once" and first:
                    status, reply = 529, {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
                data = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler

    def respond(self, body: dict) -> dict:
        usage = {"input_tokens": len(json.dumps(body)) // 4, "output_tokens": 40}
        base = {"id": "msg_fake", "type": "message", "role": "assistant", "model": body["model"],
                "stop_sequence": None, "usage": usage}
        fmt = (body.get("output_config") or {}).get("format")
        if fmt:
            props = fmt["schema"]["properties"]
            if "silent_failure" in props:
                answer = {"silent_failure": True, "category": "false_claim", "evidence": "stub"}
            elif "instructions" in props:
                answer = {"instructions": body["messages"][0]["content"].split("<instructions>\n", 1)[1]
                          .split("\n</instructions>", 1)[0] + "\nCheck each tool result before reporting.",
                          "tool_changes": [], "rationale": "stub"}
            else:
                answer = {name: False for name in props}
            return {**base, "content": [{"type": "text", "text": json.dumps(answer)}], "stop_reason": "end_turn"}
        if not body.get("tools"):
            return {**base, "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}
        messages = body["messages"]
        turn = sum(1 for m in messages if m["role"] == "assistant" and isinstance(m["content"], list))
        last = messages[-1]["content"]
        result = json.loads(last[0]["content"]) if isinstance(last, list) and last and last[0].get("type") == "tool_result" else {}
        first_user = next(m["content"] for m in messages if m["role"] == "user" and isinstance(m["content"], str))
        email = re.search(r"[\w.]+@example\.com", first_user).group(0)
        steps = [
            ("find_customer", {"email": email}),
            ("list_reservations", {"customer_id": result.get("customer_id", "")}),
            ("cancel_reservation", {"reservation_id": (result.get("reservations") or [{}])[0].get("reservation_id", "R-0")}),
            ("resolve_ticket", {"summary": "Done.", "actions": []}),
        ]
        name, tool_input = steps[min(turn, len(steps) - 1)]
        content = [{"type": "thinking", "thinking": "", "signature": "sig"},
                   {"type": "tool_use", "id": f"toolu_{turn}", "name": name, "input": tool_input}]
        return {**base, "content": content, "stop_reason": "tool_use"}
