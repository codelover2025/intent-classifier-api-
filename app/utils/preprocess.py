import re

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"\S+@\S+", "", text)                  # Remove emails
    text = re.sub(r"\d{10,}", "", text)                  # Remove phone numbers
    text = re.sub(r"[^\w\s]", "", text)                  # Remove punctuation
    return text.strip()