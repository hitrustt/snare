"""Turns the world into a time-ordered stream of card transactions.

Each customer spends at their own pace, mostly in the afternoon and evening,
at their regular stores, occasionally travelling or buying a new phone.
Fraud is injected by the attack patterns in `scenarios.py`, which schedule
transactions into the same queue so they interleave with honest traffic
exactly as they would on a real network.
"""

from __future__ import annotations

import heapq
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass

from .geo import ALL_CITIES, US_CITIES
from .world import CATEGORIES, CATEGORY_BY_NAME, Customer, World

DAY = 86_400.0
HOUR = 3_600.0
MINUTE = 60.0


@dataclass(slots=True)
class Transaction:
    id: int
    ts: float  # seconds since the simulation epoch
    customer: int
    card: int
    merchant: int
    category: str
    online: bool
    amount: float
    lat: float
    lng: float
    city: int
    device: int  # -1 for in-store purchases
    ip: int  # -1 for in-store purchases
    address: int  # shipping address, -1 if nothing ships
    is_fraud: bool = False
    fraud_type: str | None = None
    scenario: int | None = None


@dataclass(slots=True)
class _Travel:
    city: int
    until: float
    hotel_ip: int = -1


class Simulator:
    """A discrete-event simulation over a priority queue of upcoming purchases."""

    def __init__(self, world: World, seed: int, start: float = 0.0):
        self.world = world
        self.rng = random.Random(seed)
        self.now = start
        self._queue: list[tuple[float, int, int, object]] = []
        # Plain counters (not itertools.count) so the simulator can be saved
        # and reloaded; see snare/snapshot.py.
        self._seq = 0
        self._last_id = 0
        self._travel: dict[int, _Travel] = {}
        self._last_scenario = 0
        self._next_signups = start + DAY
        for customer in world.customers:
            if customer.daily_rate > 0:
                self._schedule_next(customer, start)

    # ------------------------------------------------------------ queueing
    _NORMAL = 0
    _INJECTED = 1

    def _push(self, ts: float, kind: int, payload: object) -> None:
        self._seq += 1
        heapq.heappush(self._queue, (ts, self._seq, kind, payload))

    def _schedule_next(self, customer: Customer, after: float) -> None:
        # Thinning: draw at the peak rate, accept with the hour's activity.
        peak_rate = customer.daily_rate / DAY * 2.2
        ts = after
        while True:
            ts += self.rng.expovariate(peak_rate)
            if self.rng.random() < _activity(customer, ts) / 2.2:
                break
        self._push(ts, self._NORMAL, customer.id)

    def inject(self, ts: float, txn: Transaction) -> None:
        """Schedule a pre-built (fraudulent) transaction."""
        self._push(ts, self._INJECTED, txn)

    def new_scenario_id(self) -> int:
        self._last_scenario += 1
        return self._last_scenario

    def next_id(self) -> int:
        self._last_id += 1
        return self._last_id

    # ------------------------------------------------------------- stream
    def run_until(self, end: float) -> Iterator[Transaction]:
        while self._queue and self._queue[0][0] <= end:
            if self._queue[0][0] >= self._next_signups:
                self._signups(self._next_signups)
                self._next_signups += DAY
                continue
            ts, _, kind, payload = heapq.heappop(self._queue)
            self.now = ts
            if kind == self._INJECTED:
                txn = payload  # type: ignore[assignment]
                assert isinstance(txn, Transaction)
                txn.id = self.next_id()
                txn.ts = ts
                yield txn
            else:
                customer = self.world.customers[payload]  # type: ignore[index]
                txn = self._normal(customer, ts)
                self._followups(customer, txn)
                yield txn
                self._schedule_next(customer, ts)
        self.now = max(self.now, end)

    def _signups(self, day_start: float) -> None:
        # About 0.1% of the base signs up each day, spread through the day.
        honest = sum(1 for customer in self.world.customers if not customer.synthetic)
        for _ in range(_poisson(self.rng, honest * 0.001)):
            customer = self.world.add_customer(day_start + self.rng.uniform(0, DAY))
            self._schedule_next(customer, customer.created_at)

    # ------------------------------------------------------- honest buying
    def _normal(self, customer: Customer, ts: float) -> Transaction:
        rng = self.rng
        world = self.world
        travel = self._current_travel(customer, ts)
        city = travel.city if travel else customer.home_city

        # Life happens: a new phone now and then.
        if rng.random() < 0.002:
            customer.phone = world.new_id("device")

        category = CATEGORIES[_weighted_index(rng, customer.category_weights)]
        if travel and category.name in ("gas", "grocery") and rng.random() < 0.5:
            category = CATEGORY_BY_NAME["restaurant"]
        online = rng.random() < category.online

        amount = rng.lognormvariate(math.log(category.median * customer.amount_scale), category.spread)
        # A few genuinely big purchases: a TV, a flight, furniture.
        if rng.random() < 0.004:
            amount *= rng.uniform(4, 12)
        amount = round(max(0.5, min(amount, 8000.0)), 2)

        if online:
            merchant_id = world.merchant_for(category.name, None)
            on_phone = rng.random() < 0.65
            device = customer.phone if on_phone else customer.laptop
            if travel:
                ip = travel.hotel_ip if rng.random() < 0.5 else world.carrier_ip(city)
            else:
                ip = world.carrier_ip(city) if on_phone else customer.home_ip
            lat, lng = _jitter(rng, ALL_CITIES[city].lat, ALL_CITIES[city].lng, 0.08)
            address = customer.address if category.ships else -1
            # Gifts and orders to work: an honest new shipping address.
            if address >= 0 and rng.random() < 0.02:
                address = world.new_id("address")
        else:
            regulars = customer.regulars.get(category.name) if not travel else None
            if regulars and rng.random() < 0.75:
                merchant_id = rng.choice(regulars)
            else:
                merchant_id = world.merchant_for(category.name, city)
            merchant = world.merchants[merchant_id]
            lat, lng = merchant.lat, merchant.lng
            device = ip = address = -1

        return Transaction(
            id=self.next_id(),
            ts=ts,
            customer=customer.id,
            card=customer.card_id,
            merchant=merchant_id,
            category=category.name,
            online=online,
            amount=amount,
            lat=lat,
            lng=lng,
            city=city,
            device=device,
            ip=ip,
            address=address,
        )

    def _followups(self, customer: Customer, txn: Transaction) -> None:
        """Honest bursts that look like fraud: a shopping spree, or a run of
        in-app purchases that is indistinguishable from card testing by
        amount and pace alone."""
        rng = self.rng
        if not txn.online:
            return
        if txn.category == "digital_goods" and rng.random() < 0.3:
            extra, gap, small = rng.randint(1, 6), 90.0, True
        elif rng.random() < 0.08:
            extra, gap, small = rng.randint(1, 3), 300.0, False
        else:
            return
        ts = txn.ts
        for _ in range(extra):
            ts += rng.expovariate(1 / gap)
            if small:
                merchant = txn.merchant if rng.random() < 0.7 else self.world.merchant_for("digital_goods", None)
                amount = round(rng.choice((0.99, 1.99, 2.99, 4.99, 9.99)), 2)
                category = "digital_goods"
            else:
                category = rng.choice(("online_retail", "retail", "home", "electronics", "online_retail"))
                merchant = self.world.merchant_for(category, None)
                spec = CATEGORY_BY_NAME[category]
                amount = round(rng.lognormvariate(math.log(spec.median * customer.amount_scale), spec.spread), 2)
            follow = Transaction(
                id=0,
                ts=0.0,
                customer=customer.id,
                card=customer.card_id,
                merchant=merchant,
                category=category,
                online=True,
                amount=amount,
                lat=txn.lat,
                lng=txn.lng,
                city=txn.city,
                device=txn.device,
                ip=txn.ip,
                address=txn.address
                if CATEGORY_BY_NAME[category].ships and txn.address >= 0
                else (customer.address if CATEGORY_BY_NAME[category].ships else -1),
            )
            self.inject(ts, follow)

    def _current_travel(self, customer: Customer, ts: float) -> _Travel | None:
        travel = self._travel.get(customer.id)
        if travel and travel.until < ts:
            del self._travel[customer.id]
            travel = None
        if travel is None and customer.travel_per_day and self.rng.random() < customer.travel_per_day / max(customer.daily_rate, 0.25) * 3:
            destination = self.rng.randrange(len(ALL_CITIES))
            if destination != customer.home_city:
                travel = _Travel(destination, ts + self.rng.uniform(1.5, 7) * DAY, self.world.new_ip(destination))
                self._travel[customer.id] = travel
        return travel


def _activity(customer: Customer, ts: float) -> float:
    """Relative spending activity at this time of day (peaks around the customer's usual hour)."""
    hour = (ts % DAY) / HOUR
    distance = min(abs(hour - customer.peak_hour), 24 - abs(hour - customer.peak_hour))
    night = 0.08 if 1 <= hour < 6 else 0.35
    return night + 1.6 * math.exp(-(distance**2) / (2 * 3.2**2))


def _poisson(rng: random.Random, mean: float) -> int:
    # Knuth's method is fine for the small means used here.
    limit, count, product = math.exp(-mean), 0, rng.random()
    while product > limit:
        count += 1
        product *= rng.random()
    return count


def _weighted_index(rng: random.Random, weights: list[float]) -> int:
    target = rng.random()
    running = 0.0
    for index, weight in enumerate(weights):
        running += weight
        if target <= running:
            return index
    return len(weights) - 1


def _jitter(rng: random.Random, lat: float, lng: float, scale: float) -> tuple[float, float]:
    return lat + rng.gauss(0, scale), lng + rng.gauss(0, scale * 1.2)


def us_city_far_from(rng: random.Random, city: int, min_km: float = 800) -> int:
    from .geo import haversine_km

    origin = ALL_CITIES[city]
    options = [index for index, other in enumerate(US_CITIES) if haversine_km(origin.lat, origin.lng, other.lat, other.lng) >= min_km]
    return rng.choice(options) if options else rng.randrange(len(US_CITIES))
