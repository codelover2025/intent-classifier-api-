import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from app.utils.preprocess import clean_text

class HybridClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.keyword_rules = {
            "SEOLead": ["hire seo", "looking for seo", "need seo expert"],
            "RemoteSEOJob": ["remote seo", "work from home seo"],
            "NoLead": ["not interested", "unavailable"]
        }

    def load_model(self, model_path: str):
        """Load pre-trained model and vectorizer."""
        self.vectorizer, self.model = joblib.load(model_path)

    def predict(self, text: str) -> str:
        """Hybrid prediction: Rule-based first, fallback to ML."""
        text = clean_text(text.lower())
        
        # Rule-based matching
        for intent, keywords in self.keyword_rules.items():
            if any(re.search(rf"\b{kw}\b", text) for kw in keywords):
                return intent

        # ML model prediction
        if self.model:
            X = self.vectorizer.transform([text])
            return self.model.predict(X)[0]
        
        return "NotLead"  # Default fallback