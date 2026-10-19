import os
import time
from pathlib import Path

import pytest

MODEL = Path(__file__).resolve().parent.parent / "models" / "model.txt.gz"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="train the model first: python -m snare.train")

os.environ.setdefault("SNARE_CUSTOMERS", "1500")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from snare.server import app

    with TestClient(app) as test_client:
        # The network warms up in the background; wait for it.
        for _ in range(600):
            if test_client.get("/api/health").json()["ready"]:
                break
            time.sleep(0.1)
        yield test_client


def test_health(client):
    assert client.get("/api/health").json() == {"ok": True, "ready": True}


def test_model_reports_metrics(client):
    body = client.get("/api/model").json()
    assert 0 < body["threshold"] < 1
    assert body["metrics"]["model"]["recall"] > 0.5


def test_stream_attack_and_case(client):
    with client.websocket_connect("/api/stream") as socket:
        hello = socket.receive_json()
        assert hello["type"] == "hello"
        assert hello["txns"], "new viewers get recent history"

        attack = client.post("/api/attacks", json={"kind": "card_testing"})
        assert attack.status_code == 200
        assert client.post("/api/attacks", json={"kind": "card_testing"}).status_code == 429  # cooldown

        tick = socket.receive_json()
        assert tick["type"] == "tick"
        assert any(event["scenario"] == attack.json()["scenario"] for event in tick["events"])

    txn_id = hello["txns"][-1][0]
    case = client.get(f"/api/cases/{txn_id}").json()
    assert case["transaction"][0] == txn_id
    assert {"risk", "reassuring"} <= case["explanation"].keys()


def test_unknown_attack_is_rejected(client):
    assert client.post("/api/attacks", json={"kind": "nope"}).status_code == 400


def test_missing_case_is_404(client):
    assert client.get("/api/cases/999999999").status_code == 404
