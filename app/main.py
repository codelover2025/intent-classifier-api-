# app/main.py

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

import os
import joblib

from app.utils.preprocess import clean_text
from app.utils.auth import verify_api_key

# Setup
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# CORS (optional if frontend is hosted separately)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = os.path.join("data", "models", "classifier.pkl")
model = joblib.load(MODEL_PATH)

# -------------------------------
# ✅ Root Route
# -------------------------------
@app.get("/")
async def root():
    return RedirectResponse(url="/form")

# -------------------------------
# ✅ API Route: /predict
# -------------------------------
@app.post("/predict")
async def predict_api(text: str, api_key: str):
    if not verify_api_key(api_key):
        return JSONResponse(status_code=401, content={"error": "Invalid API key"})

    cleaned = clean_text(text)
    intent = model.predict([cleaned])[0]
    return {"text": text, "predicted_intent": intent}


# -------------------------------
# ✅ Web Form Routes
# -------------------------------
@app.get("/form", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "result": None})


@app.post("/form", response_class=HTMLResponse)
async def submit_form(request: Request, user_text: str = Form(...)):
    cleaned = clean_text(user_text)
    intent = model.predict([cleaned])[0]
    return templates.TemplateResponse("form.html", {
        "request": request,
        "result": intent,
        "input_text": user_text
    })
