"""The simulated payment network: customers, cards, merchants, devices.

Everything is generated from a seed so training and the live demo run on the
same kind of world, and results are reproducible.

Realism that matters for fraud detection — the things that make it hard:
  * households share an address and a laptop, so "shared device" alone is
    not fraud;
  * phones on mobile networks share carrier IP addresses with thousands of
    strangers, so "many cards on one IP" alone is not fraud;
  * people travel, buy a new phone, and occasionally make one big purchase.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .geo import ALL_CITIES, US_CITIES, City


@dataclass(frozen=True, slots=True)
class Category:
    name: str
    label: str
    share: float  # share of everyday purchases
    median: float  # median amount, dollars
    spread: float  # lognormal sigma
    online: float  # probability a purchase is online
    ships: bool  # online purchases ship a physical item to an address


CATEGORIES: tuple[Category, ...] = (
    Category("grocery", "Groceries", 0.22, 46, 0.6, 0.08, True),
    Category("restaurant", "Restaurant", 0.18, 31, 0.6, 0.12, False),
    Category("gas", "Gas station", 0.10, 41, 0.35, 0.0, False),
    Category("retail", "Retail", 0.10, 58, 0.8, 0.35, True),
    Category("online_retail", "Online store", 0.10, 44, 0.9, 1.0, True),
    Category("rideshare", "Rideshare", 0.06, 19, 0.5, 1.0, False),
    Category("pharmacy", "Pharmacy", 0.05, 24, 0.6, 0.1, True),
    Category("subscription", "Subscription", 0.05, 13, 0.4, 1.0, False),
    Category("digital_goods", "Apps and games", 0.04, 9, 0.8, 1.0, False),
    Category("electronics", "Electronics", 0.03, 170, 1.0, 0.5, True),
    Category("travel", "Travel", 0.03, 340, 0.8, 0.8, False),
    Category("home", "Home goods", 0.02, 140, 0.9, 0.45, True),
    Category("gift_cards", "Gift cards", 0.02, 50, 0.6, 0.6, False),
)

CATEGORY_BY_NAME = {category.name: category for category in CATEGORIES}

# Fictional merchant names, so nothing looks like a real company's records.
_NAME_PARTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "grocery": (("Greenleaf", "Corner", "Harvest", "Daily", "Fieldstone", "Orchard"), ("Market", "Grocer", "Foods", "Pantry")),
    "restaurant": (("Blue Plate", "Copper", "Saffron", "Lantern", "Juniper", "Basil"), ("Diner", "Kitchen", "Bistro", "Grill", "Cantina")),
    "gas": (("Northstar", "Route", "Prairie", "Summit", "Bayou"), ("Fuel", "Gas", "Petro", "Stop")),
    "retail": (("Maple", "Union", "Harbor", "Cobalt", "Linden"), ("Outfitters", "Goods", "Supply", "Mercantile")),
    "online_retail": (("Parcelly", "Shipwise", "Cartwheel", "Boxlane", "Ordo"), ("", ".shop", " Online")),
    "rideshare": (("Zip", "Hop", "Glide", "Swift"), ("Ride", "Rides", "Go")),
    "pharmacy": (("Wellspring", "Remedy", "Cornerstone", "Evergreen"), ("Pharmacy", "Drug", "Health")),
    "subscription": (("Streamly", "Tunebox", "Cloudnest", "Readwell", "Fitloop"), (" Premium", " Plus", "")),
    "digital_goods": (("Pixelforge", "Appstack", "Questline", "Gemhub"), (" Store", " Games", "")),
    "electronics": (("Voltage", "Circuit", "Gizmo", "Pixel", "Ampere"), ("Electronics", "Tech", "Depot", "Hub")),
    "travel": (("Skyway", "Wayfare", "Atlas", "Jetstream"), ("Air", "Travel", "Hotels", "Trips")),
    "home": (("Hearth", "Oakline", "Nestwell", "Driftwood"), ("Home", "Furnishings", "Living")),
    "gift_cards": (("GiftVault", "Cardly", "Presento"), ("", " Gift Cards", " eGifts")),
}


@dataclass(slots=True)
class Merchant:
    id: int
    name: str
    category: str
    online: bool
    city: int | None  # index into ALL_CITIES for physical stores
    lat: float
    lng: float


@dataclass(slots=True)
class Customer:
    id: int
    card_id: int
    last4: str
    home_city: int
    home_lat: float
    home_lng: float
    household: int
    address: int
    home_ip: int
    phone: int
    laptop: int
    daily_rate: float
    amount_scale: float
    peak_hour: float
    category_weights: list[float]
    travel_per_day: float
    regulars: dict[str, list[int]] = field(default_factory=dict)
    synthetic: bool = False  # created by a fraud ring, not a real person
    created_at: float = 0.0


class World:
    """All entities, plus counters for minting new ones mid-simulation."""

    def __init__(self, seed: int, customers: int, merchants_per_city: int = 6):
        self.rng = random.Random(seed)
        self.customers: list[Customer] = []
        self.merchants: list[Merchant] = []
        self.merchants_by: dict[tuple[str, int | None], list[int]] = {}
        self._next = {"device": 1, "ip": 1, "address": 1, "card": 1}
        # Carrier NAT pools: each city's mobile users share a handful of IPs.
        self.carrier_ips: dict[int, list[int]] = {}
        self.ip_city: dict[int, int] = {}
        # Phones reused by fraud operations across many victims.
        self.farm_devices: list[int] = []

        self._build_merchants(merchants_per_city)
        self._build_customers(customers)

    # ------------------------------------------------------------------ ids
    def new_id(self, kind: str) -> int:
        value = self._next[kind]
        self._next[kind] = value + 1
        return value

    def new_ip(self, city: int) -> int:
        ip = self.new_id("ip")
        self.ip_city[ip] = city
        return ip

    def carrier_ip(self, city: int) -> int:
        pool = self.carrier_ips.get(city)
        if pool is None:
            pool = [self.new_ip(city) for _ in range(8)]
            self.carrier_ips[city] = pool
        return self.rng.choice(pool)

    # ------------------------------------------------------------ merchants
    def _build_merchants(self, per_city: int) -> None:
        rng = self.rng
        for category in CATEGORIES:
            firsts, seconds = _NAME_PARTS[category.name]
            if category.online > 0:
                for _ in range(8):
                    self._add_merchant(category, None, f"{rng.choice(firsts)}{rng.choice(seconds)}")
            if category.online < 1:
                for city_index, city in enumerate(ALL_CITIES):
                    count = max(1, round(per_city * min(2.0, city.weight / 3))) if city.country == "US" else 2
                    for _ in range(count):
                        name = f"{rng.choice(firsts)} {rng.choice(seconds)}".strip()
                        self._add_merchant(category, city_index, name, city)

    def _add_merchant(self, category: Category, city_index: int | None, name: str, city: City | None = None) -> None:
        rng = self.rng
        lat = city.lat + rng.gauss(0, 0.05) if city else 0.0
        lng = city.lng + rng.gauss(0, 0.06) if city else 0.0
        merchant = Merchant(len(self.merchants), name, category.name, city is None, city_index, lat, lng)
        self.merchants.append(merchant)
        self.merchants_by.setdefault((category.name, city_index), []).append(merchant.id)

    def merchant_for(self, category: str, city: int | None) -> int:
        options = self.merchants_by.get((category, city)) or self.merchants_by[(category, None)]
        return self.rng.choice(options)

    # ------------------------------------------------------------ customers
    def _build_customers(self, count: int) -> None:
        rng = self.rng
        weights = [city.weight for city in US_CITIES]
        while len(self.customers) < count:
            city_index = rng.choices(range(len(US_CITIES)), weights)[0]
            city = US_CITIES[city_index]
            household = len(self.customers)
            address = self.new_id("address")
            home_ip = self.new_ip(city_index)
            laptop = self.new_id("device")
            lat = city.lat + rng.gauss(0, 0.08)
            lng = city.lng + rng.gauss(0, 0.1)
            # Households of 1-4 share an address, home internet and a laptop.
            size = rng.choices((1, 2, 3, 4), (0.45, 0.35, 0.13, 0.07))[0]
            for _ in range(min(size, count - len(self.customers))):
                self.customers.append(self._make_customer(city_index, lat, lng, household, address, home_ip, laptop))

    def _make_customer(self, city: int, lat: float, lng: float, household: int, address: int, home_ip: int, laptop: int) -> Customer:
        rng = self.rng
        base = [category.share for category in CATEGORIES]
        # Everyone's mix of spending differs.
        weights = [w * math.exp(rng.gauss(0, 0.6)) for w in base]
        total = sum(weights)
        customer = Customer(
            id=len(self.customers),
            card_id=self.new_id("card"),
            last4=f"{rng.randrange(10000):04d}",
            home_city=city,
            home_lat=lat + rng.gauss(0, 0.01),
            home_lng=lng + rng.gauss(0, 0.01),
            household=household,
            address=address,
            home_ip=home_ip,
            phone=self.new_id("device"),
            laptop=laptop,
            daily_rate=max(0.25, rng.lognormvariate(math.log(1.4), 0.55)),
            amount_scale=rng.lognormvariate(0, 0.35),
            peak_hour=min(21.5, max(9.0, rng.gauss(15.5, 2.5))),
            category_weights=[w / total for w in weights],
            travel_per_day=rng.choice((0.0, 0.002, 0.004, 0.01)),
            # Most accounts are months or years old.
            created_at=-rng.uniform(60, 3000) * 86_400.0,
        )
        for category in CATEGORIES:
            if category.online < 1:
                options = self.merchants_by.get((category.name, city), [])
                customer.regulars[category.name] = rng.sample(options, min(3, len(options))) if options else []
        return customer

    def add_customer(self, now: float) -> Customer:
        """An honest new sign-up. New accounts are not fraud by themselves."""
        rng = self.rng
        weights = [city.weight for city in US_CITIES]
        city_index = rng.choices(range(len(US_CITIES)), weights)[0]
        city = US_CITIES[city_index]
        customer = self._make_customer(
            city_index,
            city.lat + rng.gauss(0, 0.08),
            city.lng + rng.gauss(0, 0.1),
            len(self.customers),
            self.new_id("address"),
            self.new_ip(city_index),
            self.new_id("device"),
        )
        customer.created_at = now
        self.customers.append(customer)
        return customer

    def add_synthetic_customer(self, city: int, now: float, address: int, device: int, ip: int) -> Customer:
        """An account opened by a fraud ring: real-looking, but sharing infrastructure."""
        rng = self.rng
        customer = self._make_customer(city, US_CITIES[city].lat, US_CITIES[city].lng, -1, address, ip, device)
        customer.synthetic = True
        customer.phone = device
        customer.daily_rate = 0.0  # it only ever spends as part of the ring
        customer.created_at = now
        customer.home_lat += rng.gauss(0, 0.05)
        customer.home_lng += rng.gauss(0, 0.05)
        self.customers.append(customer)
        return customer
