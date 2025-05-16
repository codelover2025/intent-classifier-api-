from pydantic import BaseModel
from typing import Optional

# Define the PredictionRequest schema
class PredictionRequest(BaseModel):
    text: str
    guid: str
    key: Optional[str] = None  # Optional API key

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I want to hire an SEO expert for my site",
                "guid": "user-1234",
                "key": "client_api_key"
            }
        }

# Example retrain request schema (optional)
class RetrainRequest(BaseModel):
    api_key: str

    class Config:
        json_schema_extra = {
            "example": {
                "api_key": "retrain_api_key"
            }
        }
