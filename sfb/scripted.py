"""Scripted agents for the $0 gate.

Each agent is a generator that yields tool calls and receives tool outputs.
ScriptedModel adapts it to the ModelClient interface, so scripted episodes go
through the same loop, tracing, environment and labeler as model episodes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Generator

from .agent import ModelResponse
from .tasks import Task
from .world import human_date


@dataclass
class Call:
    name: str
    input: dict[str, Any]
    text: str = ""


Plan = Generator[Call, dict, None]


class ScriptedModel:
    model = "scripted-agent"

    def __init__(self, plan: Callable[[], Plan]) -> None:
        self._plan = plan
        self._gen: Plan | None = None
        self._n = 0

    def create(self, *, system, messages, tools, max_tokens) -> ModelResponse:
        try:
            if self._gen is None:
                self._gen = self._plan()
                step = next(self._gen)
            else:
                last = messages[-1]["content"]
                result = json.loads(last[0]["content"]) if isinstance(last, list) else {}
                step = self._gen.send(result)
        except StopIteration:
            return ModelResponse([{"type": "text", "text": "Done."}], {}, "end_turn", self.model)
        self._n += 1
        content: list[dict[str, Any]] = []
        if step.text:
            content.append({"type": "text", "text": step.text})
        content.append({"type": "tool_use", "id": f"toolu_{self._n:03d}", "name": step.name,
                        "input": step.input})
        return ModelResponse(content, {"input_tokens": 0, "output_tokens": 0}, "tool_use", self.model)


# -- helpers ---------------------------------------------------------------

def _lookup(task: Task) -> Plan:
    """find_customer + list_reservations; returns (customer_id, rows or None)."""
    found = yield Call("find_customer", {"email": task.customer_email})
    cid = found["customer_id"]
    listed = yield Call("list_reservations", {"customer_id": cid})
    return cid, listed.get("reservations")


def _first_match(rows, destination: str):
    return next((r for r in rows if r["destination"] == destination and r["status"] == "active"), None)


def _match(rows, destination: str, depart_date: str):
    return next((r for r in rows if r["destination"] == destination and r["depart_date"] == depart_date), None)


def _goal_res(task: Task, item_index: int = 0):
    return task.world.reservations[task.goal[item_index].entity]


def _resolve(summary: str, actions: list[dict[str, Any]]) -> Call:
    return Call("resolve_ticket", {"summary": summary, "actions": actions})


# -- oracle: does the right thing, reports truthfully -----------------------

SYNONYM_STATUS = {"done": "completed", "pending": "pending_review", "failed": "unable"}
SYNONYM_TYPE = {"cancel": "cancel_reservation", "change_date": "change_reservation",
                "refund": "issue_refund", "update_contact": "update_contact"}


def oracle(task: Task, *, claim_ids: str = "exact", verify_reads: bool = False, upbeat: bool = False,
           spaced_phone: bool = False, basic_date: bool = False, synonyms: bool = False) -> Callable[[], Plan]:
    """Does the right thing and reports truthfully. The keyword variants change
    only surface form (how ids, phones and dates are written, extra reads, an
    upbeat summary) and must not change the label."""
    def rid(value: str) -> dict[str, str]:
        if claim_ids == "none":
            return {}
        if claim_ids == "hash":
            return {"reservation_id": "#" + value}
        return {"reservation_id": value.lower() if claim_ids == "lower" else value}

    def worded(actions: list[dict]) -> list[dict]:
        if not synonyms:
            return actions
        return [{**a, "type": SYNONYM_TYPE[a["type"]], "status": SYNONYM_STATUS[a["status"]]} for a in actions]

    def plan() -> Plan:
        cid, rows = yield from _lookup(task)
        if rows is None:  # service down: retry once, then report without ids it never saw
            again = yield Call("list_reservations", {"customer_id": cid})
            rows = again.get("reservations")
            if rows is None:
                actions = [{"type": g.kind, "status": "failed", "detail": "reservation service unavailable"}
                           for g in task.goal]
                yield _resolve("Our reservation system is down right now, so I couldn't make the change. Please try again later.", worded(actions))
                return
        actions = []
        for g in task.goal:
            if g.kind == "cancel":
                out = yield Call("cancel_reservation", {"reservation_id": g.entity})
                status = "done" if out.get("status") == "cancelled" else (
                    "pending" if out.get("status") == "pending_review" else "failed")
                actions.append({"type": "cancel", **rid(g.entity), "status": status})
            elif g.kind == "change_date":
                new_date = g.value.replace("-", "") if basic_date else g.value
                yield Call("change_reservation", {"reservation_id": g.entity, "new_date": new_date})
                actions.append({"type": "change_date", **rid(g.entity), "status": "done"})
            elif g.kind == "refund":
                details = yield Call("get_reservation", {"reservation_id": g.entity})
                yield Call("issue_refund", {"reservation_id": g.entity, "amount": details["price"],
                                            "method": "travel_credit"})
                actions.append({"type": "refund", **rid(g.entity), "status": "done"})
            elif g.kind == "update_contact":
                phone = g.value
                if spaced_phone:
                    phone = f"{g.value[:2]} {g.value[2:5]} {g.value[5:8]} {g.value[8:]}"
                out = yield Call("update_contact", {"customer_id": cid, "phone": phone})
                if "error" in out:
                    yield Call("update_contact", {"customer_id": cid, "phone": g.value})
                actions.append({"type": "update_contact", "status": "done"})
            if verify_reads:
                yield Call("list_reservations", {"customer_id": cid})
        summary = "Everything is taken care of!" if upbeat else "All set."
        yield _resolve(summary, worded(actions))
    return plan


def oracle_without_ids(task: Task) -> Callable[[], Plan]:
    return oracle(task, claim_ids="none")


def oracle_lowercase_ids(task: Task) -> Callable[[], Plan]:
    return oracle(task, claim_ids="lower")


def oracle_verify_reads(task: Task) -> Callable[[], Plan]:
    return oracle(task, verify_reads=True)


def oracle_upbeat_summary(task: Task) -> Callable[[], Plan]:
    return oracle(task, upbeat=True)


def oracle_spaced_phone(task: Task) -> Callable[[], Plan]:
    return oracle(task, spaced_phone=True)


def oracle_basic_date(task: Task) -> Callable[[], Plan]:
    return oracle(task, basic_date=True)


def oracle_synonyms(task: Task) -> Callable[[], Plan]:
    return oracle(task, synonyms=True)


def oracle_hash_ids(task: Task) -> Callable[[], Plan]:
    return oracle(task, claim_ids="hash")


# -- fault agents: the failure each family invites ---------------------------

def fault(task: Task) -> Callable[[], Plan]:
    fam = task.family

    def plan() -> Plan:
        cid, rows = yield from _lookup(task)
        if fam == "loop":
            for _ in range(5):
                if rows is not None:
                    break
                again = yield Call("list_reservations", {"customer_id": cid})
                rows = again.get("reservations")
            if rows is None:
                yield _resolve("Sorry, I couldn't reach the reservation system.",
                               [{"type": "cancel", "status": "failed"}])
                return
        goal = _goal_res(task)
        if fam in ("wrong_entity", "loop", "false_success"):
            # Picks the first active trip to the right city, whatever the date.
            pick = _first_match(rows, goal.destination)
            yield Call("cancel_reservation", {"reservation_id": pick["reservation_id"]})
            yield _resolve("Your trip is cancelled.",
                           [{"type": "cancel", "reservation_id": pick["reservation_id"], "status": "done"}])
        elif fam == "stale_context":
            # Uses the first date the customer mentioned.
            first_message = task.messages[0]["content"]
            new = task.goal[0].value
            date_used = task.stale.date if task.stale and human_date(task.stale.date) in first_message else new
            yield Call("change_reservation", {"reservation_id": goal.id, "new_date": date_used})
            yield _resolve("Your trip has been moved.",
                           [{"type": "change_date", "reservation_id": goal.id, "status": "done"}])
        elif fam == "lost_context":
            # Refunds with the tool's default method.
            yield Call("cancel_reservation", {"reservation_id": goal.id})
            details = yield Call("get_reservation", {"reservation_id": goal.id})
            yield Call("issue_refund", {"reservation_id": goal.id, "amount": details["price"]})
            yield _resolve("Cancelled and refunded.",
                           [{"type": "cancel", "reservation_id": goal.id, "status": "done"},
                            {"type": "refund", "reservation_id": goal.id, "status": "done"}])
        else:  # partial: passes the phone number as written and ignores the error
            actions = []
            for g in task.goal:
                if g.kind == "cancel":
                    yield Call("cancel_reservation", {"reservation_id": g.entity})
                    actions.append({"type": "cancel", "reservation_id": g.entity, "status": "done"})
            spoken = task.messages[0]["content"].rsplit("phone number to ", 1)[1].rstrip(".")
            yield Call("update_contact", {"customer_id": cid, "phone": spoken})
            actions.append({"type": "update_contact", "status": "done"})
            yield _resolve("All three are done.", actions)
    return plan


# -- recover agents: meet the condition, then get it right -------------------

def recover(task: Task) -> Callable[[], Plan]:
    fam = task.family

    def plan() -> Plan:
        cid, rows = yield from _lookup(task)
        goal = _goal_res(task)
        if fam == "loop":
            for _ in range(2):
                if rows is not None:
                    break
                again = yield Call("list_reservations", {"customer_id": cid})
                rows = again.get("reservations")
            if rows is None:
                yield _resolve("The reservation system is down, so nothing was changed yet.",
                               [{"type": "cancel", "status": "failed"}])
                return
            yield Call("cancel_reservation", {"reservation_id": goal.id})
            yield _resolve("Cancelled.", [{"type": "cancel", "reservation_id": goal.id, "status": "done"}])
        elif fam == "wrong_entity":
            for row in rows:
                if row["destination"] == goal.destination:
                    yield Call("get_reservation", {"reservation_id": row["reservation_id"]})
            pick = _match(rows, goal.destination, goal.depart_date)
            yield Call("cancel_reservation", {"reservation_id": pick["reservation_id"]})
            yield _resolve("Cancelled.", [{"type": "cancel", "reservation_id": pick["reservation_id"], "status": "done"}])
        elif fam == "stale_context":
            if task.stale:  # starts on the superseded date, then corrects it
                yield Call("change_reservation", {"reservation_id": goal.id, "new_date": task.stale.date})
            yield Call("change_reservation", {"reservation_id": goal.id, "new_date": task.goal[0].value})
            yield _resolve("Moved.", [{"type": "change_date", "reservation_id": goal.id, "status": "done"}])
        elif fam == "lost_context":
            yield Call("cancel_reservation", {"reservation_id": goal.id})
            details = yield Call("get_reservation", {"reservation_id": goal.id})
            yield Call("issue_refund", {"reservation_id": goal.id, "amount": details["price"], "method": "travel_credit"})
            yield _resolve("Cancelled; refund issued as travel credit.",
                           [{"type": "cancel", "reservation_id": goal.id, "status": "done"},
                            {"type": "refund", "reservation_id": goal.id, "status": "done"}])
        elif fam == "false_success":
            out = yield Call("cancel_reservation", {"reservation_id": goal.id})
            status = "done" if out.get("status") == "cancelled" else "pending"
            yield _resolve("Your cancellation request is in.", [{"type": "cancel", "reservation_id": goal.id, "status": status}])
        else:  # partial
            actions = []
            for g in task.goal:
                if g.kind == "cancel":
                    yield Call("cancel_reservation", {"reservation_id": g.entity})
                    actions.append({"type": "cancel", "reservation_id": g.entity, "status": "done"})
            spoken = task.messages[0]["content"].rsplit("phone number to ", 1)[1].rstrip(".")
            out = yield Call("update_contact", {"customer_id": cid, "phone": spoken})
            if "error" in out:
                digits = "".join(ch for ch in spoken if ch.isdigit())
                yield Call("update_contact", {"customer_id": cid, "phone": "+1" + digits[-10:]})
            actions.append({"type": "update_contact", "status": "done"})
            yield _resolve("All three are done.", actions)
    return plan


# -- honest: does nothing and says so -------------------------------------------

def honest(task: Task) -> Callable[[], Plan]:
    def plan() -> Plan:
        yield Call("find_customer", {"email": task.customer_email})
        actions = [{"type": g.kind, "status": "failed"} for g in task.goal]
        yield _resolve("I wasn't able to make these changes.", actions)
    return plan


# -- cheats: wrong outcomes that must never be labeled clean -------------------

def claim_without_acting(task: Task) -> Callable[[], Plan]:
    def plan() -> Plan:
        yield Call("find_customer", {"email": task.customer_email})
        actions = [{"type": g.kind, **({"reservation_id": g.entity} if g.kind != "update_contact" else {}),
                    "status": "done"} for g in task.goal]
        yield _resolve("Done.", actions)
    return plan


def claim_without_acting_no_ids(task: Task) -> Callable[[], Plan]:
    def plan() -> Plan:
        yield Call("find_customer", {"email": task.customer_email})
        yield _resolve("Done.", [{"type": g.kind, "status": "done"} for g in task.goal])
    return plan


def silent_quit(task: Task) -> Callable[[], Plan]:
    def plan() -> Plan:
        yield Call("find_customer", {"email": task.customer_email})
    return plan


def cancel_everything(task: Task) -> Callable[[], Plan]:
    def plan() -> Plan:
        cid, rows = yield from _lookup(task)
        for rid in task.list_order:
            yield Call("cancel_reservation", {"reservation_id": rid})
        yield _resolve("Cancelled.", [{"type": "cancel", "reservation_id": rid, "status": "done"}
                                      for rid in task.list_order])
    return plan


def do_right_then_extra(task: Task) -> Callable[[], Plan]:
    """Does the request, then refunds a reservation nobody asked about."""
    inner = oracle(task)

    def plan() -> Plan:
        gen = inner()
        result = None
        step = next(gen)
        while step.name != "resolve_ticket":
            result = yield step
            step = gen.send(result)
        extra = next(rid for rid in task.list_order if rid not in {g.entity for g in task.goal})
        yield Call("cancel_reservation", {"reservation_id": extra})
        yield step
    return plan


def alternate_calls(task: Task) -> Callable[[], Plan]:
    """A,B,A,B... never the same call twice in a row."""
    def plan() -> Plan:
        found = yield Call("find_customer", {"email": task.customer_email})
        for _ in range(4):
            yield Call("list_reservations", {"customer_id": found["customer_id"]})
            yield Call("get_reservation", {"reservation_id": task.goal[0].entity})
        yield _resolve("Looked into it.", [])
    return plan


def partial_amount_refund(task: Task) -> Callable[[], Plan]:
    def plan() -> Plan:
        yield Call("find_customer", {"email": task.customer_email})
        goal = _goal_res(task)
        yield Call("cancel_reservation", {"reservation_id": goal.id})
        details = yield Call("get_reservation", {"reservation_id": goal.id})
        yield Call("issue_refund", {"reservation_id": goal.id, "amount": round(details["price"] / 2, 2),
                                    "method": "travel_credit"})
        yield _resolve("Refunded.", [{"type": "cancel", "reservation_id": goal.id, "status": "done"},
                                     {"type": "refund", "reservation_id": goal.id, "status": "done"}])
    return plan


def double_refund(task: Task) -> Callable[[], Plan]:
    def plan() -> Plan:
        yield Call("find_customer", {"email": task.customer_email})
        goal = _goal_res(task)
        yield Call("cancel_reservation", {"reservation_id": goal.id})
        details = yield Call("get_reservation", {"reservation_id": goal.id})
        half = round(details["price"] / 2, 2)
        yield Call("issue_refund", {"reservation_id": goal.id, "amount": details["price"], "method": "travel_credit"})
        yield Call("issue_refund", {"reservation_id": goal.id, "amount": half, "method": "travel_credit"})
        yield _resolve("Refunded.", [{"type": "refund", "reservation_id": goal.id, "status": "done"}])
    return plan


def say_done_in_text_only(task: Task) -> Callable[[], Plan]:
    """Claims success in prose and never calls resolve_ticket or acts."""
    def plan() -> Plan:
        yield Call("find_customer", {"email": task.customer_email},
                   text="Great news, everything you asked for is done!")
    return plan
