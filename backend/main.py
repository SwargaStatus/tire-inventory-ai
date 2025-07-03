import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.utils.supabase_client import fetch_training_data, get_supabase_client, inspect_table_schema
from backend.models.ensemble import train_ensemble_models, generate_recommendations
from backend.models.data_processor import process_training_data
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Validate environment variables on startup
REQUIRED_ENVS = ["SUPABASE_URL", "SUPABASE_KEY"]
missing_envs = [env for env in REQUIRED_ENVS if not os.getenv(env)]
if missing_envs:
    logger.error(f"Missing required environment variables: {missing_envs}")
else:
    logger.info("All required environment variables are set.")

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

@app.get("/debug/env")
def debug_env():
    env_status = {env: os.getenv(env, None) for env in REQUIRED_ENVS}
    return {"env": env_status, "missing": [k for k, v in env_status.items() if not v]}

@app.get("/debug/db-connection")
def debug_db_connection():
    try:
        client = get_supabase_client()
        # Try a simple query
        response = client.table('trainingdata').select('*').limit(1).execute()
        if response.data:
            return {"success": True, "message": "Supabase connection successful.", "sample": response.data}
        else:
            return {"success": False, "message": "No data returned from Supabase."}
    except Exception as e:
        logger.exception("Supabase connection failed")
        return {"success": False, "error": str(e)}

@app.get("/debug/table-schema")
def debug_table_schema():
    try:
        columns = inspect_table_schema()
        return {"success": True, "columns": columns}
    except Exception as e:
        logger.exception("Failed to inspect table schema")
        return {"success": False, "error": str(e)}

@app.post("/api/train-models")
async def train_models(request: Request):
    global trained_models, last_processed
    config = await request.json()
    logger.info(f"Received train-models request with config: {config}")
    try:
        logger.info("Fetching data from Supabase...")
        df = fetch_training_data(
            days_to_fetch=config.get('daysToFetch', 90),
            lookback_days=config.get('lookbackDays', 30)
        )
        logger.info(f"Fetched data shape: {df.shape}, columns: {list(df.columns)}")
        logger.info("Processing training data...")
        processed = process_training_data(df, config)
        logger.info("Training ensemble models...")
        trained_models = train_ensemble_models(processed, config)
        last_processed = processed
        logger.info(f"Training complete. Metrics: {trained_models['metrics']}")
        return {"success": True, "metrics": trained_models['metrics']}
    except Exception as e:
        logger.exception("Error in /api/train-models")
        raise HTTPException(status_code=500, detail=f"Train models error: {str(e)}")

@app.post("/api/generate-recommendations")
async def generate_recommendations_endpoint(request: Request):
    global trained_models, last_processed
    logger.info("Received generate-recommendations request")
    try:
        if trained_models is None or last_processed is None:
            logger.error("Models not trained yet. Please train first.")
            raise Exception("Models not trained yet. Please train first.")
        recommendations = generate_recommendations(last_processed, trained_models)
        logger.info(f"Generated {len(recommendations)} recommendations.")
        return {"success": True, "recommendations": recommendations}
    except Exception as e:
        logger.exception("Error in /api/generate-recommendations")
        raise HTTPException(status_code=500, detail=f"Generate recommendations error: {str(e)}") 