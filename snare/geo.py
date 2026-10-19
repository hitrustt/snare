"""Places and distances.

Metro areas weighted by population so simulated customers live where people
do. A few international cities exist for travel: both honest travellers and
fraudsters buy things far from home, which is what makes location a useful
signal rather than a giveaway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class City:
    name: str
    lat: float
    lng: float
    weight: float  # relative population
    country: str = "US"


US_CITIES: tuple[City, ...] = (
    City("New York", 40.7128, -74.0060, 19.6),
    City("Los Angeles", 34.0522, -118.2437, 12.9),
    City("Chicago", 41.8781, -87.6298, 9.4),
    City("Dallas", 32.7767, -96.7970, 7.9),
    City("Houston", 29.7604, -95.3698, 7.3),
    City("Washington", 38.9072, -77.0369, 6.4),
    City("Philadelphia", 39.9526, -75.1652, 6.2),
    City("Miami", 25.7617, -80.1918, 6.1),
    City("Atlanta", 33.7490, -84.3880, 6.2),
    City("Boston", 42.3601, -71.0589, 4.9),
    City("Phoenix", 33.4484, -112.0740, 5.0),
    City("San Francisco", 37.7749, -122.4194, 4.7),
    City("Seattle", 47.6062, -122.3321, 4.0),
    City("Minneapolis", 44.9778, -93.2650, 3.7),
    City("San Diego", 32.7157, -117.1611, 3.3),
    City("Denver", 39.7392, -104.9903, 3.0),
    City("Tampa", 27.9506, -82.4572, 3.3),
    City("Baltimore", 39.2904, -76.6122, 2.8),
    City("St. Louis", 38.6270, -90.1994, 2.8),
    City("Orlando", 28.5383, -81.3792, 2.7),
    City("Charlotte", 35.2271, -80.8431, 2.7),
    City("San Antonio", 29.4241, -98.4936, 2.6),
    City("Portland", 45.5152, -122.6784, 2.5),
    City("Sacramento", 38.5816, -121.4944, 2.4),
    City("Pittsburgh", 40.4406, -79.9959, 2.4),
    City("Austin", 30.2672, -97.7431, 2.4),
    City("Las Vegas", 36.1699, -115.1398, 2.3),
    City("Cincinnati", 39.1031, -84.5120, 2.3),
    City("Kansas City", 39.0997, -94.5786, 2.2),
    City("Columbus", 39.9612, -82.9988, 2.1),
    City("Indianapolis", 39.7684, -86.1581, 2.1),
    City("Cleveland", 41.4993, -81.6944, 2.1),
    City("Nashville", 36.1627, -86.7816, 2.0),
    City("San Jose", 37.3382, -121.8863, 2.0),
    City("Detroit", 42.3314, -83.0458, 4.3),
    City("New Orleans", 29.9511, -90.0715, 1.3),
    City("Salt Lake City", 40.7608, -111.8910, 1.3),
    City("Raleigh", 35.7796, -78.6382, 1.4),
    City("Milwaukee", 43.0389, -87.9065, 1.6),
    City("Jacksonville", 30.3322, -81.6557, 1.6),
    City("Oklahoma City", 35.4676, -97.5164, 1.4),
    City("Memphis", 35.1495, -90.0490, 1.3),
    City("Louisville", 38.2527, -85.7585, 1.3),
    City("Richmond", 37.5407, -77.4360, 1.3),
    City("Albuquerque", 35.0844, -106.6504, 0.9),
    City("Boise", 43.6150, -116.2023, 0.8),
    City("Anchorage", 61.2181, -149.9003, 0.4),
    City("Honolulu", 21.3069, -157.8583, 1.0),
)

WORLD_CITIES: tuple[City, ...] = (
    City("London", 51.5074, -0.1278, 1, "GB"),
    City("Paris", 48.8566, 2.3522, 1, "FR"),
    City("Toronto", 43.6532, -79.3832, 1, "CA"),
    City("Mexico City", 19.4326, -99.1332, 1, "MX"),
    City("Cancún", 21.1619, -86.8515, 1, "MX"),
    City("Tokyo", 35.6762, 139.6503, 1, "JP"),
    City("Berlin", 52.5200, 13.4050, 1, "DE"),
    City("Madrid", 40.4168, -3.7038, 1, "ES"),
    City("São Paulo", -23.5505, -46.6333, 1, "BR"),
    City("Seoul", 37.5665, 126.9780, 1, "KR"),
    City("Sydney", -33.8688, 151.2093, 1, "AU"),
    City("Amsterdam", 52.3676, 4.9041, 1, "NL"),
    City("Bangkok", 13.7563, 100.5018, 1, "TH"),
    City("Dublin", 53.3498, -6.2603, 1, "IE"),
)

ALL_CITIES: tuple[City, ...] = US_CITIES + WORLD_CITIES

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lng2 - lng1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))
