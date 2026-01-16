import math
from fastapi.testclient import TestClient
from hedge_engine.main import app
from hedge_engine.sizer import compute_hedge
from hedge_engine.decision_logger import DecisionLogger

client = TestClient(app)

def test_hedge_placeholder(monkeypatch):
    async def dummy_log(record):
        return None

    monkeypatch.setattr(DecisionLogger, "log", dummy_log)
    payload = {"asset": "BTC", "amount_usd": 100000, "override_score": 0.7}
    resp = client.post("/hedge", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    expected_pct, expected_conf = compute_hedge(0.7, depth1pct_usd=5_000_000)
    assert math.isclose(data["hedge_pct"], expected_pct, rel_tol=1e-6)
    assert math.isclose(data["notional_usd"], 100000 * expected_pct, rel_tol=1e-6)
    assert math.isclose(data["confidence"], expected_conf, rel_tol=1e-6)
    assert data["version"] == "0.1.0" 