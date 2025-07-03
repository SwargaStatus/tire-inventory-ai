from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.utils.supabase_client import fetch_training_data
from backend.models.ensemble import train_ensemble_models, generate_recommendations
from backend.models.data_processor import process_training_data
import os

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory model storage (for demo; use persistent storage in production)
trained_models = None
last_processed = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/train-models")
async def train_models(request: Request):
    global trained_models, last_processed
    config = await request.json()
    try:
        # Fetch data from Supabase
        df = fetch_training_data(
            days_to_fetch=config.get('daysToFetch', 90),
            lookback_days=config.get('lookbackDays', 30)
        )
        # Process data
        processed = process_training_data(df, config)
        # Train models
        trained_models = train_ensemble_models(processed, config)
        last_processed = processed
        return {"success": True, "metrics": trained_models['metrics']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-recommendations")
async def generate_recommendations_endpoint(request: Request):
    global trained_models, last_processed
    try:
        if trained_models is None or last_processed is None:
            raise Exception("Models not trained yet. Please train first.")
        # Optionally, accept new data for recommendations
        # For now, use last processed data
        recommendations = generate_recommendations(last_processed, trained_models)
        return {"success": True, "recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 