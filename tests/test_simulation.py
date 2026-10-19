from snare import scenarios
from snare.simulator import DAY, Simulator
from snare.world import World


def run(seed: int, days: float = 1.0):
    world = World(seed=seed, customers=400)
    sim = Simulator(world, seed=seed + 1)
    scenarios.launch(sim, "card_testing", 0.3 * DAY)
    return [(t.id, round(t.ts, 3), t.customer, t.amount, t.is_fraud) for t in sim.run_until(days * DAY)]


def test_same_seed_same_history():
    assert run(3) == run(3)


def test_different_seed_different_history():
    assert run(3) != run(4)


def test_transactions_arrive_in_time_order():
    times = [ts for _, ts, *_ in run(5)]
    assert times == sorted(times)


def test_every_attack_type_produces_labelled_fraud():
    world = World(seed=9, customers=600)
    sim = Simulator(world, seed=10)
    attacks = {kind: scenarios.launch(sim, kind, 0.2 * DAY) for kind in scenarios.ATTACK_TYPES}
    txns = list(sim.run_until(2 * DAY))
    for kind, attack in attacks.items():
        fraud = [t for t in txns if t.scenario == attack.scenario]
        assert fraud, kind
        assert all(t.is_fraud and t.fraud_type == kind for t in fraud)
        assert len(fraud) == attack.transactions
    # Honest payments vastly outnumber fraud, as in real life.
    assert sum(t.is_fraud for t in txns) / len(txns) < 0.1


def test_fraud_ring_uses_shared_devices_across_new_accounts():
    world = World(seed=11, customers=300)
    sim = Simulator(world, seed=12)
    attack = scenarios.launch(sim, "fraud_ring", 0.1 * DAY, live=True)
    txns = [t for t in sim.run_until(2 * DAY) if t.scenario == attack.scenario]
    accounts = {t.customer for t in txns}
    devices = {t.device for t in txns}
    assert len(accounts) >= 8
    assert len(devices) < len(accounts)
    assert all(world.customers[a].synthetic for a in accounts)
