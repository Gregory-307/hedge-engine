from fastapi import FastAPI
from .api import router as api_router


def get_application() -> FastAPI:
    app = FastAPI(title="Hedge Engine", version="0.1.0")
    app.include_router(api_router)
    return app

app = get_application() 