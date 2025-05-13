from fastapi import HTTPException, Depends
import os

def validate_api_key(api_key: str):
    """Validate API key against environment variable."""
    stored_key = os.getenv("API_KEY") or "client_provided_key"  # Replace with your key
    if api_key != stored_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key