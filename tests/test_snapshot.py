from pathlib import Path

import pytest

MODEL = Path(__file__).resolve().parent.parent / "models" / "model.txt.gz"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="train the model first: python -m snare.train")


def test_saved_network_resumes_where_it_left_off(tmp_path):
    from snare import snapshot
    from snare.engine import LiveEngine
    from snare.model import FraudModel

    model = FraudModel()
    engine = LiveEngine(model, customers=800, warmup_days=3.4)
    last_id, now = engine.sim._last_id, engine.sim.now
    path = tmp_path / "network.pkl.gz"
    snapshot.save(engine, path)
    assert engine.model is model  # saving leaves the live engine intact

    loaded = snapshot.load(model, seed=99, path=path)
    assert loaded.sim.now == now
    batch = loaded.advance(30)
    assert batch, "payments keep flowing after a reload"
    assert batch[0].txn.id == last_id + 1
    assert all(0 <= item.score <= 1 for item in batch)
