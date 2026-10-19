"""Generates a labelled history: honest traffic plus attacks, replayed through
the streaming features exactly as the live server would see it."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from . import scenarios
from .features import FEATURES, FeatureState
from .simulator import DAY, Simulator
from .world import World

# Attacks per day per 10,000 customers. Card testing is rare as a case but
# produces many transactions; rings are rarest and largest.
ATTACK_RATES = {"card_testing": 1.4, "account_takeover": 2.0, "skimmed_card": 1.6, "fraud_ring": 0.4}
# Mix tempos so the model learns patterns, not one fixed timing.
PACES = (0.15, 0.4, 1.0, 1.0, 2.5)


@dataclass
class Dataset:
    X: np.ndarray  # float32 [n, features]
    y: np.ndarray  # int8
    ts: np.ndarray  # float64 seconds
    amount: np.ndarray  # float32
    fraud_type: np.ndarray  # object (str or "")
    scenario: np.ndarray  # int32 (0 = none)


def generate(seed: int = 7, customers: int = 15_000, days: int = 100, warmup_days: int = 14) -> Dataset:
    world = World(seed=seed, customers=customers)
    sim = Simulator(world, seed=seed + 1)
    rng = random.Random(seed + 2)

    scale = customers / 10_000
    for day in range(days):
        for kind, rate in ATTACK_RATES.items():
            for _ in range(_poisson(rng, rate * scale)):
                scenarios.launch(sim, kind, day * DAY + rng.uniform(0, DAY), pace=rng.choice(PACES))

    state = FeatureState(world)
    state.seed_known_infrastructure()

    capacity = int(customers * 1.6 * days * 1.15)
    X = np.empty((capacity, len(FEATURES)), dtype=np.float32)
    y = np.empty(capacity, dtype=np.int8)
    ts = np.empty(capacity, dtype=np.float64)
    amount = np.empty(capacity, dtype=np.float32)
    fraud_type = np.empty(capacity, dtype=object)
    scenario = np.zeros(capacity, dtype=np.int32)

    count = 0
    for txn in sim.run_until(days * DAY):
        row = state.observe(txn)
        if txn.ts < warmup_days * DAY:
            continue  # features need history to mean anything
        if count == capacity:
            capacity = int(capacity * 1.3)
            X, y, ts, amount, fraud_type, scenario = (np.resize(a, (capacity, *a.shape[1:])) for a in (X, y, ts, amount, fraud_type, scenario))
        X[count] = row
        y[count] = txn.is_fraud
        ts[count] = txn.ts
        amount[count] = txn.amount
        fraud_type[count] = txn.fraud_type or ""
        scenario[count] = txn.scenario or 0
        count += 1

    return Dataset(X[:count], y[:count], ts[:count], amount[:count], fraud_type[:count], scenario[:count])


def _poisson(rng: random.Random, mean: float) -> int:
    limit, k, p = math.exp(-mean), 0, rng.random()
    while p > limit:
        k += 1
        p *= rng.random()
    return k
