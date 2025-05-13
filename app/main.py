from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
from app.classifier import HybridClassifier
from app.utils.auth import validate_api_key
import logging

# Initialize the FastAPI app
app = FastAPI()

# Load model at startup
classifier = HybridClassifier()
try:
    classifier.load_model("data/models/classifier.pkl")
except Exception as e:
    logging.error(f"Model loading failed: {e}")
    raise RuntimeError("Model could not be loaded!")

# Request model
class PredictRequest(BaseModel):
    key: str       # API key
    text: str      # Input text to classify
    guid: str      # Passthrough identifier

# Response model
class PredictResponse(BaseModel):
    intent: str
    text: str
    guid: str

# Prediction endpoint (POST only)
@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    api_key: str = Depends(validate_api_key)
):
    try:
        intent = classifier.predict(request.text)
        return PredictResponse(
            intent=intent,
            text=request.text,
            guid=request.guid
        )
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction")

# Optional: Warn if user sends GET to /predict
@app.get("/predict")
async def get_predict_warning():
    return {
        "error": "❌ This endpoint only accepts POST requests. Use POST /predict with a JSON payload."
    }

# Welcome route (GET /)
@app.get("/")
def read_root():
    return {"message": "👋 Welcome to the Intent Classifier API"}

# Optional: Handle POST / with a basic response
@app.post("/")
def post_root():
    return {
        "info": "⚠️ POST / is not used for predictions. Please use POST /predict with the correct JSON format."
    }

# Server health check
@app.get("/status")
def status():
    return {"status": "✅ Server is running"}

# For local testing only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
