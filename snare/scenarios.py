"""Fraud attack patterns, modelled on how card fraud actually happens.

Each attack schedules its transactions into the simulator, all labelled with
the attack type and a scenario id so the app can follow one attack from its
first transaction to its last.

    card_testing     A stolen card number is checked with a burst of tiny
                     online purchases, then cashed out on something resellable.
    account_takeover A criminal logs in from a new device far away, changes
                     the shipping address and buys high-value items quickly.
    skimmed_card     A cloned card is used in stores in another city while the
                     real cardholder is still shopping at home.
    fraud_ring       A batch of fake accounts, opened on a few shared phones
                     and shipping to a few shared addresses, each make a couple
                     of ordinary-looking purchases. One at a time they pass;
                     only the links between them give them away.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .geo import ALL_CITIES, US_CITIES, WORLD_CITIES
from .simulator import DAY, HOUR, MINUTE, Simulator, Transaction, us_city_far_from
from .world import CATEGORY_BY_NAME, World

ATTACK_TYPES = ("card_testing", "account_takeover", "skimmed_card", "fraud_ring")

ATTACK_LABELS = {
    "card_testing": "Card testing",
    "account_takeover": "Account takeover",
    "skimmed_card": "Cloned card",
    "fraud_ring": "Fraud ring",
}


@dataclass(slots=True)
class Attack:
    scenario: int
    kind: str
    start: float
    end: float
    transactions: int
    accounts: list[int]


def launch(sim: Simulator, kind: str, start: float, *, pace: float = 1.0, victim: int | None = None, live: bool = False) -> Attack:
    """Schedule an attack starting at `start`.

    `pace` compresses time between steps (<1 is faster). The live demo uses a
    faster pace so a ring plays out in a minute rather than over days; the
    training data mixes paces so the model does not learn one tempo.
    `live` attacks never schedule anything before `start` (a ring cannot age
    its accounts in the past once the stream is running).
    """
    builders = {
        "card_testing": _card_testing,
        "account_takeover": _account_takeover,
        "skimmed_card": _skimmed_card,
        "fraud_ring": _fraud_ring,
    }
    rng = sim.rng
    world = sim.world
    if victim is None and kind != "fraud_ring":
        victim = _pick_victim(rng, world)
    scenario = sim.new_scenario_id()
    txns = builders[kind](sim, rng, world, start, pace, scenario, victim, live)
    for ts, txn in txns:
        sim.inject(ts, txn)
    return Attack(
        scenario=scenario,
        kind=kind,
        start=start,
        end=max(ts for ts, _ in txns),
        transactions=sum(txn.is_fraud for _, txn in txns),
        accounts=sorted({txn.customer for _, txn in txns}),
    )


def _pick_victim(rng: random.Random, world: World) -> int:
    while True:
        customer = world.customers[rng.randrange(len(world.customers))]
        if not customer.synthetic:
            return customer.id


def _fraud(
    sim: Simulator,
    *,
    customer: int,
    merchant: int,
    amount: float,
    city: int,
    online: bool,
    device: int,
    ip: int,
    address: int,
    kind: str,
    scenario: int,
    is_fraud: bool = True,
) -> Transaction:
    world = sim.world
    merchant_record = world.merchants[merchant]
    if online:
        lat = ALL_CITIES[city].lat + sim.rng.gauss(0, 0.08)
        lng = ALL_CITIES[city].lng + sim.rng.gauss(0, 0.1)
    else:
        lat, lng = merchant_record.lat, merchant_record.lng
    return Transaction(
        id=0,
        ts=0.0,
        customer=customer,
        card=world.customers[customer].card_id,
        merchant=merchant,
        category=merchant_record.category,
        online=online,
        amount=round(amount, 2),
        lat=lat,
        lng=lng,
        city=city,
        device=device,
        ip=ip,
        address=address,
        is_fraud=is_fraud,
        fraud_type=kind if is_fraud else None,
        scenario=scenario if is_fraud else 0,
    )


def _origin_city(rng: random.Random, home: int) -> int:
    """Where the fraudster appears to be. Careful ones route through a
    residential proxy in the victim's own city, which defeats location checks."""
    roll = rng.random()
    if roll < 0.35:
        return home
    if roll < 0.55:
        return len(US_CITIES) + rng.randrange(len(WORLD_CITIES))
    return us_city_far_from(rng, home, 500)


def _fraud_device(rng: random.Random, world: World) -> int:
    """Fraud operations reuse equipment across victims; a shared "farm" phone
    is one of the strongest real-world signals once it has been seen before."""
    farm = world.farm_devices
    if farm and rng.random() < 0.4:
        return rng.choice(farm)
    device = world.new_id("device")
    if rng.random() < 0.3:
        farm.append(device)
        del farm[:-40]
    return device


def _card_testing(sim, rng, world, start, pace, scenario, victim, live=False):
    customer = world.customers[victim]
    city = _origin_city(rng, customer.home_city)
    device = _fraud_device(rng, world)
    ip = world.new_ip(city) if city != customer.home_city else world.carrier_ip(city)
    txns: list[tuple[float, Transaction]] = []
    ts = start
    testers = [world.merchant_for(rng.choice(("digital_goods", "subscription", "online_retail")), None) for _ in range(rng.randint(1, 4))]
    careful = rng.random() < 0.4
    tests = rng.randint(2, 6) if careful else rng.randint(6, 28)
    for _ in range(tests):
        # Careful testers space checks out and use believable amounts.
        ts += rng.expovariate(1 / ((300 if careful else 14) * pace))
        amount = rng.choice((0.99, 1.99, 4.99, 9.99, 14.99)) if careful else rng.uniform(0.5, 4.99)
        txns.append(
            (
                ts,
                _fraud(
                    sim,
                    customer=victim,
                    merchant=rng.choice(testers),
                    amount=amount,
                    city=city,
                    online=True,
                    device=device,
                    ip=ip,
                    address=-1,
                    kind="card_testing",
                    scenario=scenario,
                ),
            )
        )
    ts += rng.uniform(1, 30 if careful else 8) * MINUTE * pace
    for _ in range(rng.randint(1, 3)):
        category = rng.choice(("electronics", "gift_cards", "electronics", "online_retail"))
        amount = rng.lognormvariate(math.log(420), 0.7)
        ships = category in ("electronics", "online_retail")
        address = (customer.address if rng.random() < 0.15 else world.new_id("address")) if ships else -1
        txns.append(
            (
                ts,
                _fraud(
                    sim,
                    customer=victim,
                    merchant=world.merchant_for(category, None),
                    amount=amount,
                    city=city,
                    online=True,
                    device=device,
                    ip=ip,
                    address=address,
                    kind="card_testing",
                    scenario=scenario,
                ),
            )
        )
        ts += rng.uniform(0.5, 10) * MINUTE * pace
    return txns


def _account_takeover(sim, rng, world, start, pace, scenario, victim, live=False):
    customer = world.customers[victim]
    city = _origin_city(rng, customer.home_city)
    device = _fraud_device(rng, world)
    ip = world.new_ip(city) if city != customer.home_city else world.carrier_ip(city)
    # Some change the shipping address; careful ones ship to a reshipper
    # that looks like a normal home address, or buy things that never ship.
    drop_address = world.new_id("address") if rng.random() < 0.7 else customer.address
    txns: list[tuple[float, Transaction]] = []
    ts = start
    for _ in range(rng.randint(1, 5)):
        category = rng.choice(("electronics", "gift_cards", "travel", "electronics", "retail", "home"))
        amount = rng.lognormvariate(math.log(360), 0.7)
        ships = category in ("electronics", "retail", "home")
        txns.append(
            (
                ts,
                _fraud(
                    sim,
                    customer=victim,
                    merchant=world.merchant_for(category, None),
                    amount=amount,
                    city=city,
                    online=True,
                    device=device,
                    ip=ip,
                    address=drop_address if ships else -1,
                    kind="account_takeover",
                    scenario=scenario,
                ),
            )
        )
        ts += rng.uniform(2, 45) * MINUTE * pace
    return txns


def _skimmed_card(sim, rng, world, start, pace, scenario, victim, live=False):
    customer = world.customers[victim]
    # Cloned cards are often used locally; distance only helps sometimes.
    city = customer.home_city if rng.random() < 0.4 else us_city_far_from(rng, customer.home_city, 800)
    txns: list[tuple[float, Transaction]] = []
    ts = start
    for _ in range(rng.randint(2, 7)):
        category = rng.choice(("electronics", "retail", "gift_cards", "gas", "electronics", "grocery", "home"))
        median = {"gas": 70, "grocery": 140}.get(category, 240)
        amount = rng.lognormvariate(math.log(median), 0.6)
        txns.append(
            (
                ts,
                _fraud(
                    sim,
                    customer=victim,
                    merchant=world.merchant_for(category, city),
                    amount=amount,
                    city=city,
                    online=False,
                    device=-1,
                    ip=-1,
                    address=-1,
                    kind="skimmed_card",
                    scenario=scenario,
                ),
            )
        )
        ts += rng.uniform(10, 90) * MINUTE * pace
    return txns


def _fraud_ring(sim, rng, world, start, pace, scenario, victim, live=False):
    city = rng.randrange(len(US_CITIES))
    devices = [world.new_id("device") for _ in range(rng.randint(2, 4))]
    addresses = [world.new_id("address") for _ in range(rng.randint(1, 3))]
    ips = [world.new_ip(city) for _ in range(rng.randint(2, 3))]
    # Patient rings ("bust-out" fraud) open accounts months ahead and use them
    # like ordinary customers, so each one has a real-looking history, a
    # familiar device and a known address. Individually they are clean; only
    # the shared phones and addresses give them away.
    aged = not live and rng.random() < 0.6
    opened = max(0.0, start - rng.uniform(30, 120) * DAY) if aged else start
    accounts = []
    for _ in range(rng.randint(6, 26)):
        # Not every account touches the shared phones or addresses.
        device = rng.choice(devices) if rng.random() < 0.7 else world.new_id("device")
        address = rng.choice(addresses) if rng.random() < 0.75 else world.new_id("address")
        # Careful rings give every account its own residential proxy.
        ip = world.new_ip(city) if aged else rng.choice(ips)
        accounts.append((world.add_synthetic_customer(city, opened + (rng.uniform(0, 5 * DAY) if aged else 0.0), address, device, ip), device, address, ip))
    txns: list[tuple[float, Transaction]] = []
    for account, own_device, own_address, own_ip in accounts:
        if aged:
            # Sleeper activity: small, regular purchases. No money is lost yet,
            # so these are not labelled fraud.
            ts = account.created_at + rng.uniform(0.5, 3) * DAY
            while ts < start - DAY:
                category = CATEGORY_BY_NAME[rng.choice(("grocery", "restaurant", "online_retail", "subscription", "pharmacy", "retail", "gas"))]
                amount = rng.lognormvariate(math.log(category.median), 0.6)
                online = rng.random() < category.online
                merchant = world.merchant_for(category.name, None if online else city)
                txns.append(
                    (
                        ts,
                        _fraud(
                            sim,
                            customer=account.id,
                            merchant=merchant,
                            amount=amount,
                            city=city,
                            online=online,
                            device=own_device if online else -1,
                            ip=own_ip if online else -1,
                            address=own_address if online and category.ships else -1,
                            kind="fraud_ring",
                            scenario=scenario,
                            is_fraud=False,
                        ),
                    )
                )
                ts += rng.uniform(0.4, 3) * DAY
        # Careful rings spread the cash-out over a day.
        ts = start + rng.uniform(0, 24 if aged else 3) * HOUR * pace
        for _ in range(rng.randint(1, 3)):
            category = rng.choice(("electronics", "gift_cards", "retail", "online_retail", "electronics"))
            # Deliberately ordinary amounts: individually these look like shopping.
            amount = rng.lognormvariate(math.log(150 if category != "electronics" else 300), 0.5)
            ships = category in ("electronics", "retail", "online_retail")
            device = own_device if aged or rng.random() < 0.6 else rng.choice(devices)
            txns.append(
                (
                    ts,
                    _fraud(
                        sim,
                        customer=account.id,
                        merchant=world.merchant_for(category, None),
                        amount=amount,
                        city=city,
                        online=True,
                        device=device,
                        ip=own_ip if aged else rng.choice(ips),
                        address=own_address if ships else -1,
                        kind="fraud_ring",
                        scenario=scenario,
                    ),
                )
            )
            ts += rng.uniform(5, 90) * MINUTE * pace
    return txns
