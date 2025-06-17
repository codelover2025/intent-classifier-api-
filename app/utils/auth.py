import os
from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

# API key header field
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# This would typically be stored securely, e.g., in environment variables
# For this example, we'll use a simple approach
API_KEYS = {"test_api_key"}  # Initialize with a default test key

def load_api_keys(file_path: str = None):
    """
    Load API keys from a file.
    
    Args:
        file_path: Path to the file containing API keys
    """
    global API_KEYS
    
    # If file_path is provided, load keys from file
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            API_KEYS = set(line.strip() for line in f if line.strip())
    else:
        # For testing, add a default key
        API_KEYS = {"test_api_key"}

def verify_api_key(api_key: str) -> bool:
    """
    Verify if the provided API key is valid.
    
    Args:
        api_key: The API key to verify
        
    Returns:
        bool: True if the API key is valid, False otherwise
    """
    return api_key in API_KEYS

def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Validate API key from request header.
    
    Args:
        api_key_header: API key from request header
        
    Returns:
        API key if valid
        
    Raises:
        HTTPException: If API key is invalid
    """
    if api_key_header in API_KEYS:
        return api_key_header
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Invalid or missing API Key"
    )
