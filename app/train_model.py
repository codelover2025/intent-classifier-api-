import pandas as pd
import os
from sklearn.model_selection import train_test_split
from app.classifier import IntentClassifier

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
