from fastapi import APIRouter, status

router = APIRouter()

@router.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz() -> dict[str, str]:
    return {"status": "ok"} 