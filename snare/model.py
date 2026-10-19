"""Loading the trained model, scoring, and explaining decisions in plain English.

Explanations use LightGBM's built-in TreeSHAP (`pred_contrib=True`): each
feature's exact contribution to this transaction's score, in log-odds. The
largest contributions are turned into sentences with the real values, e.g.
"14 earlier purchases on this card in the last 10 minutes".
"""

from __future__ import annotations

import gzip
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .features import FEATURES
from .world import CATEGORIES, CATEGORY_BY_NAME

MODELS = Path(__file__).resolve().parent.parent / "models"
KM_PER_MILE = 1.609344


@dataclass
class Reason:
    feature: str
    text: str
    impact: float  # log-odds contribution; positive = riskier


class FraudModel:
    def __init__(self, path: Path = MODELS / "model.txt.gz", metrics_path: Path = MODELS / "metrics.json"):
        text = gzip.decompress(path.read_bytes()).decode()
        self.booster = lgb.Booster(model_str=text)
        # The server explains on a worker thread while scoring on the main one.
        self._lock = threading.Lock()
        names = self.booster.feature_name()
        if tuple(names) != FEATURES:
            raise RuntimeError("The saved model was trained on different features; run `python -m snare.train`.")
        self.metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        # The model's probability at the operating point (about 1 in 200
        # honest payments above it). Rare-event probabilities are tiny and
        # hard to read, so the app shows a 0-1 *risk score* instead: a
        # monotonic rescaling of the log-odds that puts this threshold at
        # 0.5. Decisions are unchanged; only the display scale differs.
        self.threshold = float(self.metrics.get("operating_point", {}).get("threshold", 0.5))
        self.risk_scale = 1.0
        self.calibrated = False

    def score(self, row: list[float]) -> float:
        # float32, exactly as the training data was stored.
        with self._lock:
            return float(self.booster.predict(np.asarray([row], dtype=np.float32), num_threads=1)[0])

    def calibrate_display(self, rows: np.ndarray, typical_risk: float = 0.03) -> None:
        """Choose the risk-score scale so a typical payment shows as `typical_risk`."""
        median = float(np.median(self.score_many(rows)))
        gap = _logit(self.threshold) - _logit(median)
        if gap > 0:
            self.risk_scale = gap / -_logit(typical_risk)
        self.calibrated = True

    def risk(self, probability: float) -> float:
        """Probability -> risk score in (0, 1), with the decline line at 0.5."""
        z = (_logit(probability) - _logit(self.threshold)) / self.risk_scale
        return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z))))

    def score_many(self, rows: np.ndarray) -> np.ndarray:
        return self.booster.predict(np.asarray(rows, dtype=np.float32), num_threads=1)

    def explain(self, row: list[float], amount: float, category: str, limit: int = 4) -> dict:
        with self._lock:
            contributions = self.booster.predict(np.asarray([row], dtype=np.float32), pred_contrib=True, num_threads=1)[0]
        # Contributions in the risk score's log-odds, so they add up to what is shown.
        contributions = contributions / self.risk_scale
        base = float(contributions[-1]) - _logit(self.threshold) / self.risk_scale
        values = dict(zip(FEATURES, row))
        reasons = []
        for name, impact in zip(FEATURES, contributions[:-1]):
            if abs(impact) < 0.05:
                continue
            text = describe(name, values, amount, category, float(impact))
            if text:
                reasons.append(Reason(name, text, float(impact)))
        reasons.sort(key=lambda reason: -abs(reason.impact))
        risky = [r for r in reasons if r.impact > 0][:limit]
        reassuring = [r for r in reasons if r.impact < 0][:2]
        return {
            "base_log_odds": base,
            "risk": [r.__dict__ for r in risky],
            "reassuring": [r.__dict__ for r in reassuring],
        }


def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def _miles(log_km: float) -> float:
    return math.expm1(log_km) / KM_PER_MILE


def _plural(count: float, word: str, plural: str | None = None) -> str:
    count = round(count)
    return f"{count:,} {word if count == 1 else (plural or word + 's')}"


def _duration(minutes: float) -> str:
    if minutes < 1:
        return "under a minute"
    if minutes < 90:
        return _plural(minutes, "minute")
    if minutes < 48 * 60:
        return _plural(minutes / 60, "hour")
    return _plural(minutes / 1440, "day")


def describe(name: str, v: dict, amount: float, category: str, impact: float) -> str | None:
    """One sentence for one feature, written for the person reviewing the case."""
    label = CATEGORY_BY_NAME[category].label.lower()
    risky = impact > 0
    match name:
        case "card_count_2m" | "card_count_10m" | "card_count_1h" | "card_count_24h":
            window = {"card_count_2m": "2 minutes", "card_count_10m": "10 minutes", "card_count_1h": "hour", "card_count_24h": "24 hours"}[name]
            count = v[name]
            if not risky:
                return f"No other purchases on this card in the last {window}" if count == 0 else None
            return f"{_plural(count, 'earlier purchase')} on this card in the last {window}" if count else None
        case "card_small_10m":
            return f"{_plural(v[name], 'purchase')} under $5 in 10 minutes, a common way to test a stolen card" if v[name] and risky else None
        case "card_merchants_1h":
            return f"Used at {_plural(v[name], 'different merchant')} in the last hour" if v[name] > 1 and risky else None
        case "card_sum_1h_log":
            total = math.expm1(v[name])
            return f"${total:,.0f} already spent on this card in the last hour" if total >= 1 and risky else None
        case "secs_since_last_log":
            seconds = math.expm1(v[name])
            if seconds < 600:
                return f"Only {_duration(seconds / 60)} after the previous purchase" if risky else None
            return f"{_duration(seconds / 60).capitalize()} since the previous purchase" if not risky else None
        case "amount_log":
            if risky:
                return f"${amount:,.2f} is a large purchase" if amount > 150 else None
            return f"Small amount (${amount:,.2f})" if amount < 60 else None
        case "amount_vs_card":
            z = v[name]
            if risky:
                return "Much larger than this customer usually spends" if z > 1.5 else ("Much smaller than this customer usually spends" if z < -1.5 else None)
            return "In line with this customer's usual spending" if abs(z) < 1 else None
        case "amount_vs_category":
            return f"High for {label}" if v[name] > 1 and risky else None
        case "km_from_home_log":
            miles = _miles(v[name])
            if risky and miles > 50:
                return f"{miles:,.0f} miles from the cardholder's home"
            return "Near the cardholder's home" if not risky and miles < 30 else None
        case "km_from_last_log":
            miles = _miles(v[name])
            return f"{miles:,.0f} miles from the previous purchase" if risky and miles > 50 else None
        case "speed_from_last_kmh_log" | "store_speed_kmh_log":
            mph = _miles(v[name])
            if risky and mph > 300:
                where = "in-store purchase" if name == "store_speed_kmh_log" else "purchase"
                return f"Would mean travelling {mph:,.0f} mph since the last {where}"
            return None
        case "foreign":
            return "Made outside the US" if v[name] and risky else None
        case "new_merchant":
            return (
                "First purchase at this merchant" if v[name] and risky else ("A merchant this customer uses regularly" if not v[name] and not risky else None)
            )
        case "new_category":
            return f"First {label} purchase on this card" if v[name] and risky else None
        case "category_share":
            share = v[name]
            if risky and share < 0.05:
                return f"This customer rarely buys {label}"
            return f"This customer often buys {label}" if not risky and share > 0.15 else None
        case "device_known_min_log" | "ip_known_min_log" | "address_known_min_log":
            if math.isnan(v[name]):
                return None
            minutes = math.expm1(v[name])
            thing = {"device_known_min_log": "device", "ip_known_min_log": "network", "address_known_min_log": "shipping address"}[name]
            if minutes == 0:
                return f"First time this account has used this {thing}" if risky else None
            if risky and minutes < 24 * 60:
                return f"This {thing} was first used on the account {_duration(minutes)} ago"
            return f"A {thing} the account has used for {_duration(minutes)}" if not risky and minutes > 7 * 1440 else None
        case "hour_unusual":
            hours = v[name]
            return f"About {hours:.0f} hours from this customer's usual time of day" if risky and hours >= 4 else None
        case "account_age_days_log":
            days = math.expm1(v[name])
            if risky and days < 30:
                return f"Account opened {_duration(days * 1440)} ago"
            return f"Account open for {_duration(days * 1440)}" if not risky and days > 180 else None
        case "card_history_log":
            count = math.expm1(v[name])
            if not risky or count >= 10:
                return None
            return "First purchase on this card" if round(count) == 0 else f"Only {_plural(count, 'earlier purchase')} on this card"
        case "device_accounts":
            return f"This device has been used by {_plural(v[name], 'other account')}" if v[name] >= 2 and risky else None
        case "address_accounts":
            return f"{_plural(v[name], 'other account')} ship to this address" if v[name] >= 2 and risky else None
        case "ip_cards_1h":
            return f"{_plural(v[name], 'other card')} used on this network in the last hour" if v[name] >= 3 and risky else None
        case "ring_size":
            size = v[name]
            if risky and size >= 5:
                return f"Linked to {_plural(size - 1, 'other account')} through shared devices and addresses"
            return None
        case "ring_new_accounts":
            return f"{_plural(v[name], 'linked account')} opened in the last 2 weeks" if v[name] >= 2 and risky else None
        case "online":
            return ("Online purchase, where stolen card details are easiest to use" if v[name] else "In store, where a cloned card works") if risky else None
        case "category":
            name_ = CATEGORIES[int(v[name])].name
            if risky and name_ in ("gift_cards", "electronics", "digital_goods"):
                return f"{CATEGORIES[int(v[name])].label} are easy to resell and often targeted"
            return None
        case "hour":
            hour = v[name]
            return "Made in the middle of the night" if risky and 1 <= hour < 5 else None
    return None
