from fastapi import FastAPI
from .api import router as api_router
from prometheus_client import make_asgi_app


def get_application() -> FastAPI:
    app = FastAPI(
        title="Inventory Risk Engine",
        description="Assess inventory risk and get actionable hedge recommendations",
        version="0.2.0",
    )
    app.include_router(api_router)
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    return app


app = get_application()
