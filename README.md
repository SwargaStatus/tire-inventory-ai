# Tire Inventory AI

A production-ready AI-powered tire inventory management system. Features include:
- Machine learning model training and recommendations
- FastAPI backend with CORS
- Supabase integration for database
- Ready for Railway deployment

## Project Structure

```
/frontend
  index.html
/backend
  main.py
  requirements.txt
  /models
    __init__.py
    ensemble.py
    data_processor.py
  /utils
    __init__.py
    supabase_client.py
README.md
.gitignore
railway.toml
Procfile
```

## Backend Setup

1. Install dependencies:
    ```bash
    cd backend
    pip install -r requirements.txt
    ```
2. Set environment variables for Supabase:
    - `SUPABASE_URL`
    - `SUPABASE_KEY`
3. Run the server:
    ```bash
    uvicorn main:app --reload
    ```

## Deployment

- Ready for [Railway](https://railway.app/) deployment.
- Procfile and railway.toml included.

## Endpoints
- `GET /health` - Health check
- `POST /api/train-models` - Train ML models
- `POST /api/generate-recommendations` - Get recommendations 