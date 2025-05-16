import requests
import json
import sys

def test_api(text, guid, api_key="test_api_key", url="http://localhost:8000"):
    """
    Test the text intent classification API.
    
    Args:
        text: Text to classify
        guid: GUID for the request
        api_key: API key for authentication
        url: API URL
    """
    # Prepare request
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    
    data = {
        "text": text,
        "guid": guid
    }
    
    # Send request
    try:
        response = requests.post(f"{url}/classify", headers=headers, json=data)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    # Get command line arguments
    if len(sys.argv) < 3:
        print("Usage: python test_api.py <text> <guid> [api_key]")
        sys.exit(1)
    
    text = sys.argv[1]
    guid = sys.argv[2]
    api_key = sys.argv[3] if len(sys.argv) > 3 else "test_api_key"
    
    test_api(text, guid, api_key)
