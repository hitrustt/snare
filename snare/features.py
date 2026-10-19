"""Streaming features: what the model knows about a transaction at the moment
it happens.

`FeatureState.observe(txn)` returns the features for `txn` computed from
everything seen *before* it, then folds `txn` into the state. Training replays
history through this exact code, and the live server runs it on each payment,
so the model is trained on precisely what it sees in production — no
look-ahead leakage and no training/serving skew. (Computing "number of
purchases on this card today" with a pandas groupby over the whole day is the
classic mistake: it counts purchases that had not happened yet.)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .geo import ALL_CITIES, haversine_km
from .graph import AccountGraph
from .simulator import DAY, HOUR, MINUTE, Transaction
from .world import CATEGORIES, CATEGORY_BY_NAME, World

CATEGORY_CODES = {category.name: index for index, category in enumerate(CATEGORIES)}

FEATURES: tuple[str, ...] = (
    # amount
    "amount_log",
    "amount_vs_card",
    "amount_vs_category",
    # velocity on the card
    "card_count_2m",
    "card_count_10m",
    "card_count_1h",
    "card_count_24h",
    "card_sum_1h_log",
    "card_small_10m",
    "card_merchants_1h",
    "secs_since_last_log",
    # location
    "km_from_home_log",
    "km_from_last_log",
    "speed_from_last_kmh_log",
    "store_speed_kmh_log",
    "foreign",
    # novelty for this customer
    "new_merchant",
    "new_category",
    "category_share",
    "device_known_min_log",
    "ip_known_min_log",
    "address_known_min_log",
    "hour_unusual",
    "account_age_days_log",
    "card_history_log",
    # shared infrastructure
    "device_accounts",
    "address_accounts",
    "ip_cards_1h",
    "ring_size",
    "ring_new_accounts",
    # context
    "online",
    "category",
    "hour",
)

CATEGORICAL = ("category",)

SMALL_AMOUNT = 5.0
NEW_ACCOUNT_DAYS = 14


@dataclass(slots=True)
class _Card:
    recent: deque = field(default_factory=deque)  # (ts, amount, merchant) for the last 24h
    count: int = 0
    log_mean: float = 0.0
    log_m2: float = 0.0
    hour_cos: float = 0.0
    hour_sin: float = 0.0
    merchants: set = field(default_factory=set)
    categories: dict = field(default_factory=dict)
    last_ts: float | None = None
    last_lat: float = 0.0
    last_lng: float = 0.0
    store_ts: float | None = None
    store_lat: float = 0.0
    store_lng: float = 0.0


@dataclass(slots=True)
class _Account:
    # First time this account used each device / IP / shipping address.
    devices: dict = field(default_factory=dict)
    ips: dict = field(default_factory=dict)
    addresses: dict = field(default_factory=dict)


class FeatureState:
    def __init__(self, world: World):
        self.world = world
        self.cards: dict[int, _Card] = {}
        self.accounts: dict[int, _Account] = {}
        self.ip_recent: dict[int, deque] = {}  # ip -> deque[(ts, card)] for the last hour
        self.graph = AccountGraph()

    # ------------------------------------------------------------------ api
    def observe(self, txn: Transaction) -> list[float]:
        features = self.compute(txn)
        self.update(txn)
        return features

    def compute(self, txn: Transaction) -> list[float]:
        world = self.world
        customer = world.customers[txn.customer]
        card = self.cards.get(txn.card) or _Card()
        account = self.accounts.get(txn.customer) or _Account()
        ts = txn.ts

        # -- velocity over the card's last 24 hours
        count_2m = count_10m = count_1h = count_24h = small_10m = 0
        sum_1h = 0.0
        merchants_1h: set[int] = set()
        for past_ts, past_amount, past_merchant in reversed(card.recent):
            age = ts - past_ts
            if age > DAY:
                break
            count_24h += 1
            if age <= HOUR:
                count_1h += 1
                sum_1h += past_amount
                merchants_1h.add(past_merchant)
                if age <= 10 * MINUTE:
                    count_10m += 1
                    if past_amount < SMALL_AMOUNT:
                        small_10m += 1
                    if age <= 2 * MINUTE:
                        count_2m += 1

        # -- amount relative to this card's history and the category
        log_amount = math.log1p(txn.amount)
        if card.count >= 3:
            std = math.sqrt(card.log_m2 / (card.count - 1)) if card.count > 1 else 1.0
            amount_vs_card = (log_amount - card.log_mean) / max(std, 0.35)
        else:
            amount_vs_card = 0.0
        amount_vs_category = log_amount - math.log1p(CATEGORY_BY_NAME[txn.category].median)

        # -- location
        km_home = haversine_km(customer.home_lat, customer.home_lng, txn.lat, txn.lng)
        if card.last_ts is not None:
            km_last = haversine_km(card.last_lat, card.last_lng, txn.lat, txn.lng)
            hours = max((ts - card.last_ts) / HOUR, 1 / 60)
            speed = km_last / hours
            secs_since = ts - card.last_ts
        else:
            km_last = speed = 0.0
            secs_since = 30 * DAY
        # In-store to in-store is the reliable impossible-travel check: an IP
        # address only approximates location, a card terminal does not.
        store_speed = 0.0
        if not txn.online and card.store_ts is not None:
            store_km = haversine_km(card.store_lat, card.store_lng, txn.lat, txn.lng)
            store_speed = store_km / max((ts - card.store_ts) / HOUR, 1 / 60)

        # -- novelty
        history = card.count
        new_merchant = float(history > 0 and txn.merchant not in card.merchants)
        new_category = float(history > 0 and txn.category not in card.categories)
        category_share = card.categories.get(txn.category, 0) / history if history else 0.0
        # How long this account has known the device, IP and address. A
        # yes/no "new device" flag only fires on a fraudster's first purchase;
        # "known for 3 minutes" stays suspicious for the whole attack.
        device_known = _known_minutes(account.devices, txn.device, ts) if txn.online else math.nan
        ip_known = _known_minutes(account.ips, txn.ip, ts) if txn.online else math.nan
        address_known = _known_minutes(account.addresses, txn.address, ts) if txn.address >= 0 else math.nan
        hour = (ts % DAY) / HOUR
        if card.count >= 5:
            usual = math.degrees(math.atan2(card.hour_sin, card.hour_cos)) / 15 % 24
            hour_unusual = min(abs(hour - usual), 24 - abs(hour - usual))
        else:
            hour_unusual = 0.0
        account_age = max(0.0, (ts - customer.created_at) / DAY)

        # -- shared infrastructure
        device_accounts = self.graph.device_accounts(txn.device, exclude=txn.customer) if txn.device >= 0 else 0
        address_accounts = self.graph.address_accounts(txn.address, exclude=txn.customer) if txn.address >= 0 else 0
        ip_cards = 0
        if txn.ip >= 0:
            recent_ip = self.ip_recent.get(txn.ip)
            if recent_ip:
                ip_cards = len({past_card for past_ts, past_card in recent_ip if ts - past_ts <= HOUR and past_card != txn.card})
        ring = self.graph.component_if_linked(txn.customer, txn.device, txn.address)
        ring_size = len(ring)
        ring_new = sum(1 for member in ring if ts - world.customers[member].created_at < NEW_ACCOUNT_DAYS * DAY)

        return [
            log_amount,
            amount_vs_card,
            amount_vs_category,
            count_2m,
            count_10m,
            count_1h,
            count_24h,
            math.log1p(sum_1h),
            small_10m,
            len(merchants_1h),
            math.log1p(secs_since),
            math.log1p(km_home),
            math.log1p(km_last),
            math.log1p(speed),
            math.log1p(store_speed),
            float(ALL_CITIES[txn.city].country != "US"),
            new_merchant,
            new_category,
            category_share,
            device_known,
            ip_known,
            address_known,
            hour_unusual,
            math.log1p(account_age),
            math.log1p(history),
            device_accounts,
            address_accounts,
            ip_cards,
            ring_size,
            ring_new,
            float(txn.online),
            CATEGORY_CODES[txn.category],
            hour,
        ]

    def update(self, txn: Transaction) -> None:
        card = self.cards.get(txn.card)
        if card is None:
            card = self.cards[txn.card] = _Card()
        ts = txn.ts

        card.recent.append((ts, txn.amount, txn.merchant))
        while card.recent and ts - card.recent[0][0] > DAY:
            card.recent.popleft()

        # Welford's running mean/variance of log amount.
        card.count += 1
        value = math.log1p(txn.amount)
        delta = value - card.log_mean
        card.log_mean += delta / card.count
        card.log_m2 += delta * (value - card.log_mean)

        angle = math.radians(((ts % DAY) / HOUR) * 15)
        card.hour_cos += math.cos(angle)
        card.hour_sin += math.sin(angle)
        card.merchants.add(txn.merchant)
        card.categories[txn.category] = card.categories.get(txn.category, 0) + 1
        card.last_ts, card.last_lat, card.last_lng = ts, txn.lat, txn.lng
        if not txn.online:
            card.store_ts, card.store_lat, card.store_lng = ts, txn.lat, txn.lng

        account = self.accounts.get(txn.customer)
        if account is None:
            account = self.accounts[txn.customer] = _Account()
        if txn.device >= 0:
            account.devices.setdefault(txn.device, ts)
        if txn.ip >= 0:
            account.ips.setdefault(txn.ip, ts)
            recent_ip = self.ip_recent.setdefault(txn.ip, deque())
            recent_ip.append((ts, txn.card))
            while recent_ip and ts - recent_ip[0][0] > HOUR:
                recent_ip.popleft()
        if txn.address >= 0:
            account.addresses.setdefault(txn.address, ts)

        self.graph.link(txn.customer, txn.device, txn.address)

    def seed_known_infrastructure(self) -> None:
        """Accounts already know their own home devices and address.

        A real issuer has years of history; without this, every customer's
        first online purchase in the simulation would look like a new device.
        """
        for customer in self.world.customers:
            if customer.synthetic:
                continue
            account = self.accounts.setdefault(customer.id, _Account())
            known_since = min(customer.created_at, -180 * DAY)
            for device in (customer.phone, customer.laptop):
                account.devices.setdefault(device, known_since)
            account.ips.setdefault(customer.home_ip, known_since)
            account.addresses.setdefault(customer.address, known_since)
            self.graph.link(customer.id, customer.laptop, customer.address)
            self.graph.link(customer.id, customer.phone, -1)


def _known_minutes(first_seen: dict, key: int, now: float) -> float:
    """log(1 + minutes since the account first used `key`); 0 if never."""
    since = first_seen.get(key)
    return 0.0 if since is None else math.log1p(max(0.0, now - since) / MINUTE)
