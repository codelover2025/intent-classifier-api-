import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from app.utils.preprocess import clean_text

def train_model(data_path: str, model_save_path: str):
    """Train and save intent classifier."""
    df = pd.read_csv(data_path)
    df["text"] = df["text"].apply(clean_text)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )
    
    # Feature extraction
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    
    # Save model + vectorizer
    joblib.dump((vectorizer, model), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_model(
        data_path="data/raw/SEOLeadDataset.csv",
        model_save_path="data/models/classifier.pkl"
    )