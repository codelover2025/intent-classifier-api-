from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from app.schemas import PredictionRequest
from app.classifier import IntentClassifier
import os

app = FastAPI()

# Allow CORS for testing purposes (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the classifier model at startup
model_path = "data/models/classifier.pkl"
classifier = IntentClassifier()
if os.path.exists(model_path):
    classifier.load(model_path)
else:
    classifier = None

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico")
async def favicon():
    return RedirectResponse(url="/static/favicon.ico")

@app.get("/")
async def root_get():
    return {"message": "Welcome to the FastAPI app"}

@app.post("/")
async def root_post(request: PredictionRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        prediction = classifier.predict(request.text)
        return {
            "guid": request.guid,
            "intent": prediction,
            "text": request.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify")
async def classify(request: PredictionRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        prediction = classifier.predict(request.text)
        return {
            "guid": request.guid,
            "intent": prediction,
            "text": request.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def train_model():
    """Train and save the intent classification model."""
    # Path to dataset
    dataset_path = "data/raw/SEOLeadDataset.csv"
    model_path = "data/models/classifier.pkl"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return False
    
    # Load dataset
    try:
        df = pd.read_csv(dataset_path)
        print(f"Loaded dataset with {len(df)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    # Ensure required columns exist
    required_columns = ['text', 'intent']
    if not all(col in df.columns for col in required_columns):
        print(f"Dataset missing required columns: {required_columns}")
        return False
    
    # Initialize classifier
    classifier = IntentClassifier()
    
    # Train model
    print("Training model...")
    metrics = classifier.train(df['text'].tolist(), df['intent'].tolist())
    
    # Print metrics
    print("Training complete. Metrics:")
    for label, metrics_dict in metrics.items():
        if label in ['macro avg', 'weighted avg']:
            print(f"{label}: precision={metrics_dict['precision']:.2f}, recall={metrics_dict['recall']:.2f}, f1-score={metrics_dict['f1-score']:.2f}")
    
    # Save model
    print(f"Saving model to {model_path}")
    classifier.save(model_path)
    
    return True

if __name__ == "__main__":
    train_model()