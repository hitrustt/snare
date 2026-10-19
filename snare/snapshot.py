"""A saved, warmed-up payment network, so the server starts in a second.

Warming up simulates a month of history for every cardholder, which takes
about 30 CPU-seconds. Free hosting gives a fraction of one CPU, so the Docker
build does it once ahead of time:

    python -m snare.snapshot        # writes models/network.pkl.gz

The server loads this file at startup and again whenever it resets the
network, with fresh random seeds each time so no two runs play out alike.
"""

from __future__ import annotations

import gzip
import pickle
import random
import time
from pathlib import Path

from .engine import LiveEngine
from .model import MODELS, FraudModel

PATH = MODELS / "network.pkl.gz"


def save(engine: LiveEngine, path: Path = PATH) -> None:
    model, engine.model = engine.model, None  # the model is loaded separately
    try:
        state = {"engine": engine, "risk_scale": model.risk_scale}
        path.write_bytes(gzip.compress(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL), compresslevel=6))
    finally:
        engine.model = model


def load(model: FraudModel, seed: int, path: Path = PATH) -> LiveEngine:
    # The file is written by this project's own build step (see the Dockerfile).
    state = pickle.loads(gzip.decompress(path.read_bytes()))
    engine: LiveEngine = state["engine"]
    engine.model = model
    if not model.calibrated:
        model.risk_scale = state["risk_scale"]
        model.calibrated = True
    # New seeds, so the future differs every time the snapshot is loaded.
    engine.rng = random.Random(seed)
    engine.sim.rng = random.Random(seed + 1)
    engine.world.rng = random.Random(seed + 2)
    return engine


def main() -> None:
    started = time.time()
    engine = LiveEngine(FraudModel())
    save(engine)
    print(f"Saved {PATH.name} ({PATH.stat().st_size / 1e6:.1f} MB) in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
