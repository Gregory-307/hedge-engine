from fastapi.testclient import TestClient
from hedge_engine.main import app

client = TestClient(app)

def test_hedge_placeholder():
    payload = {"asset": "BTC", "amount_usd": 100000}
    resp = client.post("/hedge", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["hedge_pct"] == 0.10
    assert data["notional_usd"] == 100000 * 0.10
    assert data["version"] == "0.1.0" 