from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any
import shutil
import os
import json
import tempfile
import logging
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_uploadTo_pinecone import run_incremental, train, get_config
from edge_client_full import get_user_centroid_from_pinecone, query_user_fingerprint, PINECONE_API_KEY, INDEX_NAME

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("api")

app = FastAPI(title="Discord Fingerprinting API")

@app.get("/")
def health_check():
    return {"status": "running"}

@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    """
    Upload a new JSON file of messages to update the model and Pinecone index.
    """
    # Save uploaded file to temp
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    try:
        with temp_file as f:
            shutil.copyfileobj(file.file, f)
        temp_path = temp_file.name
        
        logger.info(f"Received file {file.filename}, saved to {temp_path}")

        # Run Update (Initial or Incremental)
        logger.info("Starting model update...")
        try:
            config = get_config()
            if not os.path.exists(config.model_path):
                logger.info(f"Model not found at {config.model_path}. Running initial training...")
                train(temp_path)
            else:
                logger.info(f"Model found. Running incremental update...")
                run_incremental(temp_path)
            
            return {"status": "success", "message": "Model and Pinecone index updated successfully"}
            
        except Exception as e:
            logger.error(f"Error during model update: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error updating model: {str(e)}")

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")

@app.get("/inference")
async def run_inference(banned_users: List[str] = Query(..., description="List of banned user IDs to check")):
    """
    Check for flagged users based on the provided list of banned users.
    """
    logger.info(f"Starting inference for banned users: {banned_users}")
    results = {}
    
    if not PINECONE_API_KEY:
            logger.error("PINECONE_API_KEY not set")
            raise HTTPException(status_code=500, detail="PINECONE_API_KEY not set")

    for user_id in banned_users:
        try:
            logger.info(f"Querying for banned user: {user_id}")
            # Get centroid for the banned user
            centroid = get_user_centroid_from_pinecone(
                user_id, 
                PINECONE_API_KEY, 
                INDEX_NAME
            )
            
            if centroid is None:
                logger.warning(f"No fingerprints found for banned user {user_id}")
                results[user_id] = []
                continue

            # Query for matches
            matches = query_user_fingerprint(
                messages_df=None, 
                extractor=None, 
                top_k=10, 
                current_user_id=user_id, 
                threshold=0.1, 
                user_vector=centroid
            )
            
            # Format results
            flagged = []
            for m in matches:
                flagged.append({
                    "user_id": m['user_id'],
                    "score": float(m['score'])
                })
            results[user_id] = flagged
            
        except Exception as e:
            logger.error(f"Error processing banned user {user_id}: {e}")
            results[user_id] = {"error": str(e)}

    return {"status": "success", "flagged_users": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)