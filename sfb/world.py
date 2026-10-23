"""Seeded reservation-desk world: customers, reservations, refunds.

Everything here is deterministic in (family, seed). The induced and natural
variants of a task share the same seed and differ only in the condition the
task builder switches on.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

FIRST = ["Ana", "Ben", "Chloe", "Dev", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jonah",
         "Kara", "Liam", "Maya", "Nikhil", "Olga", "Pedro", "Quinn", "Rosa", "Sami", "Tara"]
LAST = ["Alvarez", "Brooks", "Chen", "Dubois", "Eriksen", "Fischer", "Gupta", "Haddad",
        "Ito", "Jensen", "Kowalski", "Lopez", "Mensah", "Novak", "Okafor", "Park"]
CITIES = ["Denver", "Boston", "Chicago", "Seattle", "Austin", "Miami", "Atlanta",
          "Portland", "Phoenix", "Nashville", "San Diego", "Minneapolis"]
HOME = ["New York", "San Francisco", "Los Angeles", "Dallas"]


def rng_for(family: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{family}|{seed}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def human_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.strftime('%B')} {d.day}"


def ordinal_day(d: date) -> str:
    """'the 14th': a day of the month the way customers say it."""
    suffix = "th" if 10 <= d.day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(d.day % 10, "th")
    return f"the {d.day}{suffix}"


def next_month(d: date) -> date:
    """Same day of the next month (every day of November and December exists there)."""
    return d.replace(year=d.year + 1, month=1) if d.month == 12 else d.replace(month=d.month + 1)


@dataclass
class Customer:
    id: str
    name: str
    email: str
    phone: str


@dataclass
class Reservation:
    id: str
    customer_id: str
    origin: str
    destination: str
    depart_date: str
    fare_class: str  # "refundable" | "nonrefundable"
    price_cents: int
    status: str = "active"  # "active" | "cancelled" | "pending_review" | "completed"


@dataclass
class Refund:
    id: str
    reservation_id: str
    amount_cents: int
    method: str  # "original_payment" | "travel_credit"


@dataclass
class World:
    customers: dict[str, Customer] = field(default_factory=dict)
    reservations: dict[str, Reservation] = field(default_factory=dict)
    refunds: list[Refund] = field(default_factory=list)

    def snapshot(self) -> dict:
        return {
            "customers": {k: asdict(v) for k, v in sorted(self.customers.items())},
            "reservations": {k: asdict(v) for k, v in sorted(self.reservations.items())},
            "refunds": [asdict(r) for r in self.refunds],
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.snapshot(), sort_keys=True).encode()
        ).hexdigest()

    def clone(self) -> "World":
        return copy.deepcopy(self)


class Ids:
    """Opaque, seed-dependent ids so an id never hints at its role."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.used: set[str] = set()

    def make(self, prefix: str, digits: int) -> str:
        while True:
            value = f"{prefix}-{self.rng.randrange(10 ** (digits - 1), 10 ** digits)}"
            if value not in self.used:
                self.used.add(value)
                return value


def base_day(rng: random.Random) -> date:
    # Latest trip: Dec 11 + 10 days, so every date stays in 2026.
    return date(2026, 11, 2) + timedelta(days=rng.randrange(0, 40))


def make_customer(rng: random.Random, ids: Ids) -> Customer:
    first, last = rng.choice(FIRST), rng.choice(LAST)
    email = f"{first.lower()}.{last.lower()}{rng.randrange(10, 99)}@example.com"
    phone = f"+1415555{rng.randrange(1000, 9999)}"
    return Customer(ids.make("C", 5), f"{first} {last}", email, phone)


def make_reservation(
    rng: random.Random,
    ids: Ids,
    customer: Customer,
    destination: str,
    depart: date,
    *,
    fare_class: str | None = None,
    status: str = "active",
    origin: str | None = None,
) -> Reservation:
    return Reservation(
        id=ids.make("R", 6),
        customer_id=customer.id,
        origin=origin or rng.choice(HOME),
        destination=destination,
        depart_date=depart.isoformat(),
        fare_class=fare_class or rng.choice(["refundable", "nonrefundable"]),
        price_cents=rng.randrange(180, 900) * 100 + rng.choice([0, 50]),
        status=status,
    )
