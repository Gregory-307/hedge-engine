from fastapi.testclient import TestClient
from hedge_engine.main import app

client = TestClient(app)

def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"} 