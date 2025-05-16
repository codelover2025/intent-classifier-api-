import pickle
import os
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from app.utils.preprocess import clean_text

class IntentClassifier:
    """Text intent classifier using a hybrid approach."""
    
    def __init__(self):
        """Initialize the classifier."""
        self.model = None
        self.keywords = {
            "SEOLead": ["seo", "search engine optimization", "serp", "backlink", "keyword", 
                       "ranking", "google", "organic traffic", "link building", "meta tags"]
        }
        
    def train(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """
        Train the classifier model.
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels
            
        Returns:
            Dictionary with training metrics
        """
        # Preprocess texts
        processed_texts = [clean_text(text) for text in texts]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=0.2, random_state=42
        )
        
        # Create and train pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return report
    
    def predict(self, text: str) -> str:
        """
        Predict intent for a given text.
        
        Args:
            text: Input text
            
        Returns:
            Predicted intent label
        """
        # Clean text
        cleaned_text = clean_text(text)
        
        # Rule-based classification first
        for intent, keywords in self.keywords.items():
            for keyword in keywords:
                if keyword in cleaned_text:
                    return intent
        
        # If no rules match and model exists, use ML model
        if self.model:
            return self.model.predict([cleaned_text])[0]
        
        # Default fallback
        return "Other"
    
    def save(self, path: str):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path: str):
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            return True
        return False
