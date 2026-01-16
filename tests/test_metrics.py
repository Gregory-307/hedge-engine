from fastapi.testclient import TestClient
from hedge_engine.main import app

client = TestClient(app)


def test_metrics_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    # Should contain our custom gauge name
    assert "hedge_decision_logger_circuit_state" in resp.text
