"""The live payment network behind the demo.

One shared simulation runs on the server and every visitor watches the same
stream, so an attack launched by one person shows up for everyone. Simulated
time runs faster than real time so a quiet network of 20,000 customers still
produces a steady flow of payments, and an attack plays out in seconds.

Each payment goes through the same path a production system would use:
streaming features -> model score -> decision against the threshold,
with the time for that path measured on every transaction.
"""

from __future__ import annotations

import random
import time
from array import array
from collections import OrderedDict, deque
from dataclasses import dataclass

import numpy as np

from . import scenarios
from .features import FEATURES, FeatureState
from .geo import ALL_CITIES
from .model import FraudModel
from .simulator import DAY, HOUR, Simulator, Transaction
from .world import CATEGORIES, World

CATEGORY_CODES = {category.name: index for index, category in enumerate(CATEGORIES)}
ATTACK_CODES = {kind: index + 1 for index, kind in enumerate(scenarios.ATTACK_TYPES)}

# Simulated seconds per real second.
SPEED = 90.0
# How many recent transactions are kept for case lookups.
RECENT_LIMIT = 20_000
# Live attacks run faster than background fraud so they finish in seconds.
LIVE_PACE = 0.15  # the fastest tempo in the training data


@dataclass(slots=True)
class Scored:
    txn: Transaction
    row: array  # float32, as in training; a quarter of the memory of a list
    score: float
    latency_us: float


class LiveEngine:
    def __init__(self, model: FraudModel, *, seed: int = 2026, customers: int = 12_000, warmup_days: float = 28.4):
        self.model = model
        # Scores are risk scores (see FraudModel.risk): decline above 0.5.
        self.threshold = 0.5
        self.rng = random.Random(seed)
        self.world = World(seed=seed, customers=customers)
        self.sim = Simulator(self.world, seed=seed + 1)
        self.state = FeatureState(self.world)
        self.state.seed_known_infrastructure()
        self.recent: OrderedDict[int, Scored] = OrderedDict()
        self.latencies: deque[float] = deque(maxlen=4000)
        self.attacks: dict[int, dict] = {}
        self.events: deque[dict] = deque(maxlen=50)
        self.event_seq = 0
        self.totals = {"transactions": 0, "declined": 0, "fraud": 0, "fraud_caught": 0, "false_declines": 0, "fraud_dollars": 0.0, "fraud_dollars_caught": 0.0}
        self._next_background = 0.0
        self._warm_up(warmup_days)

    # ------------------------------------------------------------- warm-up
    def _warm_up(self, days: float) -> None:
        """Give every card some history before going live, with occasional
        background fraud so the account graph has seen past operations."""
        # Ends mid-morning, so a visitor arrives to a busy network.
        end = days * DAY
        for day in range(int(days) + 1):
            for kind in scenarios.ATTACK_TYPES:
                if self.rng.random() < 0.7:
                    scenarios.launch(self.sim, kind, day * DAY + self.rng.uniform(0, DAY), pace=1.0)
        # Observe most of the history; score the last few hours so the first
        # visitor sees a populated map instead of an empty one.
        scored_from = end - 3 * HOUR
        recent: deque[list[float]] = deque(maxlen=4000)
        for txn in self.sim.run_until(scored_from):
            recent.append(self.state.observe(txn))
        if not self.model.calibrated:
            self.model.calibrate_display(np.asarray(recent, dtype=np.float64))
        for txn in self.sim.run_until(end):
            self._process(txn)
        self._next_background = end + self.rng.uniform(6, 24) * SPEED
        self.started_at = end

    # -------------------------------------------------------------- stream
    def advance(self, real_seconds: float) -> list[Scored]:
        """Run the simulation forward and score everything that happened."""
        target = self.sim.now + real_seconds * SPEED
        if target >= self._next_background:
            # Background fraud keeps the map honest even when nobody attacks.
            self.launch(self.rng.choice(scenarios.ATTACK_TYPES), source="background")
            self._next_background = target + self.rng.uniform(30, 75) * SPEED
        return [self._process(txn) for txn in self.sim.run_until(target)]

    def _process(self, txn: Transaction) -> Scored:
        """The production path for one payment: features, score, decision."""
        started = time.perf_counter()
        row = array("f", self.state.observe(txn))
        score = self.model.risk(self.model.score(row))
        latency = (time.perf_counter() - started) * 1e6
        scored = Scored(txn, row, score, latency)
        self._record(scored)
        return scored

    def _record(self, scored: Scored) -> None:
        txn = scored.txn
        self.recent[txn.id] = scored
        while len(self.recent) > RECENT_LIMIT:
            self.recent.popitem(last=False)
        self.latencies.append(scored.latency_us)
        declined = scored.score > self.threshold
        totals = self.totals
        totals["transactions"] += 1
        totals["declined"] += declined
        if txn.is_fraud:
            totals["fraud"] += 1
            totals["fraud_dollars"] += txn.amount
            if declined:
                totals["fraud_caught"] += 1
                totals["fraud_dollars_caught"] += txn.amount
        elif declined:
            totals["false_declines"] += 1
        if txn.scenario and txn.scenario in self.attacks:
            attack = self.attacks[txn.scenario]
            attack["seen"] += 1
            attack["caught"] += declined
            attack["dollars"] += txn.amount
            attack["dollars_caught"] += txn.amount if declined else 0

    # ------------------------------------------------------------- attacks
    def launch(self, kind: str, source: str = "visitor") -> dict:
        start = self.sim.now + self.rng.uniform(0.1, 0.4) * SPEED
        attack = scenarios.launch(self.sim, kind, start, pace=LIVE_PACE if source == "visitor" else 0.5, live=True)
        record = {
            "scenario": attack.scenario,
            "kind": kind,
            "label": scenarios.ATTACK_LABELS[kind],
            "source": source,
            "start": attack.start,
            "end": attack.end,
            "transactions": attack.transactions,
            "accounts": len(attack.accounts),
            "seen": 0,
            "caught": 0,
            "dollars": 0.0,
            "dollars_caught": 0.0,
        }
        self.attacks[attack.scenario] = record
        # Keep the attack log bounded.
        for old in list(self.attacks)[:-200]:
            del self.attacks[old]
        self.event_seq += 1
        self.events.append({"type": "attack", "seq": self.event_seq, **record})
        return record

    # ------------------------------------------------------------- queries
    def stats(self) -> dict:
        return {**self.totals, **self.latency_stats()}

    def latency_stats(self) -> dict:
        if not self.latencies:
            return {"p50_us": 0, "p99_us": 0}
        values = np.fromiter(self.latencies, dtype=np.float64)
        return {"p50_us": round(float(np.percentile(values, 50)), 1), "p99_us": round(float(np.percentile(values, 99)), 1)}

    def case(self, txn_id: int) -> dict | None:
        """Everything about one payment except the SHAP explanation, which the
        server computes separately because it is the slow part."""
        scored = self.recent.get(txn_id)
        if scored is None:
            return None
        txn = scored.txn
        world = self.world
        customer = world.customers[txn.customer]
        merchant = world.merchants[txn.merchant]
        values = dict(zip(FEATURES, scored.row))
        ring = None
        members: set[int] = set()
        if values["ring_size"] >= 3:
            graph = self.state.graph.subgraph(txn.customer)
            members = set(graph["accounts"])
            ring = {
                **graph,
                "edges": [[f"a{a}", f"{kind}{target}"] for _, a, kind, target in graph["edges"]],
                "new_accounts": [a for a in graph["accounts"] if txn.ts - world.customers[a].created_at < 14 * DAY],
            }
        # One pass over recent history for the related lists.
        flagged: set[int] = set()
        related: list[Scored] = []
        history: list[Scored] = []
        for other in self.recent.values():
            if members and other.score > self.threshold and other.txn.customer in members:
                flagged.add(other.txn.customer)
            if txn.scenario and other.txn.scenario == txn.scenario:
                related.append(other)
            if other.txn.card == txn.card and other.txn.id != txn.id:
                history.append(other)
        if ring is not None:
            ring["flagged_accounts"] = sorted(flagged)
        return {
            "transaction": compact(scored, self.threshold, world),
            "merchant": merchant.name,
            "category": txn.category,
            "online": txn.online,
            "city": ALL_CITIES[txn.city].name,
            "country": ALL_CITIES[txn.city].country,
            "customer": {
                "id": txn.customer,
                "last4": customer.last4,
                "home": ALL_CITIES[customer.home_city].name,
                "account_days": round((txn.ts - customer.created_at) / DAY, 1),
                "synthetic": customer.synthetic,
            },
            "score": scored.score,
            "threshold": self.threshold,
            "latency_us": round(scored.latency_us, 1),
            "ring": ring,
            "attack": self.attacks.get(txn.scenario) if txn.scenario else None,
            "related": [compact(s, self.threshold, world) for s in related[-40:]],
            "history": [compact(s, self.threshold, world) for s in history[-8:]],
        }


def compact(scored: Scored, threshold: float, world: World) -> list:
    """The wire format for one transaction: a short array, not a verbose object."""
    txn = scored.txn
    return [
        txn.id,
        round(txn.ts, 1),
        txn.amount,
        CATEGORY_CODES[txn.category],
        int(txn.online),
        round(txn.lat, 4),
        round(txn.lng, 4),
        txn.city,
        round(scored.score, 5),
        int(txn.is_fraud),
        ATTACK_CODES.get(txn.fraud_type or "", 0),
        txn.scenario or 0,
        world.merchants[txn.merchant].name,
        world.customers[txn.customer].last4,
    ]
