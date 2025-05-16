import requests
import json

BASE_URL = "http://localhost:8000"

def test_root():
    print("Testing GET /")
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    print("Response:", json.dumps(data, indent=2))

def test_classify_valid():
    print("Testing POST /classify with valid data")
    payload = {
        "text": "I want to hire an SEO expert for my site",
        "guid": "test-guid-123",
        "key": "test_api_key"
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{BASE_URL}/classify", json=payload, headers=headers)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    print("Response:", json.dumps(data, indent=2))

def test_classify_missing_text():
    print("Testing POST /classify with missing 'text'")
    payload = {
        "guid": "test-guid-123",
        "key": "test_api_key"
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{BASE_URL}/classify", json=payload, headers=headers)
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    print("Response:", response.text)

def test_classify_empty_text():
    print("Testing POST /classify with empty 'text'")
    payload = {
        "text": "",
        "guid": "test-guid-123",
        "key": "test_api_key"
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{BASE_URL}/classify", json=payload, headers=headers)
    # Depending on classifier behavior, this might be 200 or error
    print(f"Status code: {response.status_code}")
    print("Response:", response.text)

def run_all_tests():
    test_root()
    test_classify_valid()
    test_classify_missing_text()
    test_classify_empty_text()
    print("All tests completed.")

if __name__ == "__main__":
    run_all_tests()
