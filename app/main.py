from flask import Flask, request, jsonify, redirect
import os
import logging
from functools import wraps

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS configuration (optional)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', '*')
    response.headers.add('Access-Control-Allow-Methods', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Define PredictionRequest class (equivalent to FastAPI schema)
class PredictionRequest:
    def __init__(self, guid: str, text: str):
        self.guid = guid
        self.text = text

# Mock IntentClassifier class (replace with your actual implementation)
class IntentClassifier:
    def __init__(self):
        self.model = None
    
    def load(self, model_path: str) -> bool:
        try:
            # Replace with actual model loading logic
            logger.info(f"Mock loading model from {model_path}")
            self.model = "mock_model"
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, text: str) -> str:
        # Replace with actual prediction logic
        return "mock_prediction"
    
    def train(self, texts: list, intents: list) -> dict:
        # Replace with actual training logic
        return {"macro avg": {"precision": 0.9, "recall": 0.9, "f1-score": 0.9}}
    
    def save(self, model_path: str) -> bool:
        # Replace with actual save logic
        logger.info(f"Mock saving model to {model_path}")
        return True

# Load the classifier model at startup
model_path = "data/models/classifier.pkl"
classifier = None

try:
    classifier = IntentClassifier()
    if os.path.exists(model_path):
        loaded = classifier.load(model_path)
        if loaded:
            logger.info(f"Model loaded successfully from {model_path}")
        else:
            logger.error(f"Failed to load model from {model_path}")
            classifier = None
    else:
        logger.error(f"Model file does not exist at {model_path}")
        classifier = None
except Exception as e:
    logger.error(f"Exception during model loading: {e}")
    classifier = None

@app.route('/')
def home():
    return jsonify({"message": "Hello from Flask on cPanel!"})

@app.route('/favicon.ico')
def favicon():
    return redirect('/static/favicon.ico')

@app.route('/', methods=['GET'])
def root_get():
    return jsonify({"message": "Welcome to the Flask app"})

@app.route('/', methods=['POST'])
def root_post():
    try:
        data = request.get_json()
        if not data or 'guid' not in data or 'text' not in data:
            raise ValueError("Invalid request format")
        
        request_data = PredictionRequest(**data)
    except Exception as e:
        logger.error(f"Invalid request data: {e}")
        return jsonify({"error": "Invalid request format"}), 400

    if classifier is None:
        logger.error("Prediction request received but model is not loaded")
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        logger.info(f"Prediction request received: guid={request_data.guid}, text={request_data.text}")
        prediction = classifier.predict(request_data.text)
        logger.info(f"Prediction result: {prediction}")
        return jsonify({
            "guid": request_data.guid,
            "intent": prediction,
            "text": request_data.text
        })
    except Exception as e:
        logger.error(f"Exception during prediction: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

def validate_key(key: str) -> bool:
    valid_key = os.getenv('API_KEY', 'default_key')  # Use a default for testing
    return key == valid_key

def api_key_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        key = request.args.get('key')
        if not key or not validate_key(key):
            logger.error("Invalid API Key provided for classify endpoint")
            return jsonify({"error": "Invalid API Key"}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/classify', methods=['POST'])
@api_key_required
def classify():
    try:
        data = request.get_json()
        if not data or 'guid' not in data or 'text' not in data:
            raise ValueError("Invalid request format")
        
        request_data = PredictionRequest(**data)
    except Exception as e:
        logger.error(f"Invalid request data: {e}")
        return jsonify({"error": "Invalid request format"}), 400

    # Rule-based filtering
    if "PHP" in request_data.text:
        logger.info("Classified as PHPLead by rule-based filter.")
        return jsonify({
            "guid": request_data.guid,
            "intent": "PHPLead",
            "text": request_data.text
        })

    if classifier is None:
        logger.error("Classify request received but model is not loaded")
        return jsonify({"error": "Model not loaded"}), 503

    try:
        logger.info(f"Classify request received: guid={request_data.guid}, text={request_data.text}")
        prediction = classifier.predict(request_data.text)
        logger.info(f"Classified as {prediction} by ML model.")
        return jsonify({
            "guid": request_data.guid,
            "intent": prediction,
            "text": request_data.text
        })
    except Exception as e:
        logger.error(f"Exception during classify prediction: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

def train_model():
    """Train and save the intent classification model."""
    import pandas as pd
    
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
    app.run(debug=True)