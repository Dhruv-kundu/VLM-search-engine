import os
from fastapi import HTTPException, Header

VALID_KEYS = set(os.getenv("VALID_API_KEYS", "demo-key-123").split(","))


def verify_api_key(api_key: str = Header(..., alias="api-key")):
    if api_key not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key
