import re
import string
from typing import List

def clean_text(text: str) -> str:
    """
    Clean and preprocess text for classification.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Input text string
        
    Returns:
        List of tokens
    """
    return text.split()
