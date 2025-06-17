# predict_csv.py

import pandas as pd
import joblib
from app.utils.preprocess import clean_text

# Paths
INPUT_CSV = "data/raw/SEOLeadDataset.csv"
MODEL_PATH = "data/models/classifier.pkl"
OUTPUT_CSV = "predicted_output.csv"

# Load model
print("[🔁] Loading model...")
model = joblib.load(MODEL_PATH)

# Load data
print(f"[📄] Reading input CSV from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# Validate 'text' column
if 'text' not in df.columns:
    raise Exception("CSV must have a 'text' column!")

# Clean and predict
print("[🧹] Preprocessing text...")
df['cleaned'] = df['text'].astype(str).apply(clean_text)

print("[🤖] Predicting intents...")
df['Predicted Intent'] = model.predict(df['cleaned'])

# Save output
df[['text', 'Predicted Intent']].to_csv(OUTPUT_CSV, index=False)
print(f"[✅] Predictions saved to: {OUTPUT_CSV}")
