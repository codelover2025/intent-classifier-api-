from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
from app.classifier import HybridClassifier
from app.utils.auth import validate_api_key
import requests

app = FastAPI()

# Load model at startup
classifier = HybridClassifier()
classifier.load_model("data/models/classifier.pkl")

# Request model
class PredictRequest(BaseModel):
    key: str       # API key (included in body, optional if handled via query param)
    text: str      # Input text to classify
    guid: str      # Passthrough identifier

# Response model
class PredictResponse(BaseModel):
    intent: str
    text: str
    guid: str

# Main prediction endpoint (POST)
@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    api_key: str = Depends(validate_api_key)
):
    intent = classifier.predict(request.text)
    return PredictResponse(
        intent=intent,
        text=request.text,
        guid=request.guid
    )

# Optional: Handle accidental GET requests to /predict
@app.get("/predict")
async def get_predict_warning():
    return {
        "error": "This endpoint only accepts POST requests. Please use POST /predict with JSON data."
    }

# Welcome route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Intent Classifier API"}

# Status check route
@app.get("/status")
def status():
    return {"status": "Server is running"}

@app.post("/")
def post_root(request: Request):
    # You can define the handling logic for the POST request here.
    return {"message": "This is a POST request to the root!"}


# Local development entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
