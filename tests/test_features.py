import math

from snare import scenarios
from snare.features import FEATURES, FeatureState
from snare.simulator import DAY, MINUTE, Simulator
from snare.world import World

INDEX = {name: i for i, name in enumerate(FEATURES)}


def replay(days: float = 2.0):
    world = World(seed=21, customers=500)
    sim = Simulator(world, seed=22)
    scenarios.launch(sim, "card_testing", 0.5 * DAY)
    state = FeatureState(world)
    rows, txns = [], []
    for txn in sim.run_until(days * DAY):
        rows.append(state.observe(txn))
        txns.append(txn)
    return world, txns, rows


def test_row_matches_feature_list():
    _, _, rows = replay(0.5)
    assert all(len(row) == len(FEATURES) for row in rows)


def test_velocity_only_counts_the_past():
    """Brute-force check: the 10-minute count is exactly the card's earlier payments in that window."""
    _, txns, rows = replay()
    by_card: dict[int, list[float]] = {}
    for txn, row in zip(txns, rows):
        history = by_card.setdefault(txn.card, [])
        expected = sum(1 for ts in history if txn.ts - ts <= 10 * MINUTE)
        assert row[INDEX["card_count_10m"]] == expected
        history.append(txn.ts)


def test_features_do_not_change_when_the_future_changes():
    """Feeding in later payments must not alter the features of earlier ones."""
    world, txns, rows = replay()
    state = FeatureState(world)
    early = [t for t in txns if t.ts < DAY]
    for txn, full_row in zip(early, rows):
        row = state.observe(txn)
        assert all((x == y) or (math.isnan(x) and math.isnan(y)) for x, y in zip(row, full_row))


def test_card_testing_bursts_show_up_in_features():
    """Across several attacks, the burst of tiny test charges is visible in the velocity features."""
    peaks = []
    for seed in range(6):
        world = World(seed=40 + seed, customers=300)
        sim = Simulator(world, seed=seed)
        attack = scenarios.launch(sim, "card_testing", 0.5 * DAY)
        state = FeatureState(world)
        peak = 0.0
        for txn in sim.run_until(DAY):
            row = state.observe(txn)
            if txn.scenario == attack.scenario:
                peak = max(peak, row[INDEX["card_count_1h"]])
        peaks.append(peak)
    assert max(peaks) >= 5
    assert sum(peak >= 2 for peak in peaks) >= 4
