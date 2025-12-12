from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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

@app.post("/process")
async def process_data(
    file: UploadFile = File(...),
    banned_users_json: str = Form(...)
):
    """
    Process a new JSON file of messages:
    1. Update the Pinecone index with new fingerprints (incremental training).
       If no model exists, it performs initial training.
    2. Check for flagged users based on the provided list of banned users.
    
    Args:
        file: JSON file containing new messages (NDJSON format).
        banned_users_json: JSON string representing a list of banned user IDs (e.g. '["user1", "user2"]').
    
    Returns:
        JSON object with "status" and "flagged_users" (mapping of banned_user_id -> list of matches).
    """
    
    # Parse banned users
    try:
        banned_users = json.loads(banned_users_json)
        if not isinstance(banned_users, list):
            raise ValueError("banned_users must be a list of strings")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON for banned_users")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Save uploaded file to temp
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    try:
        with temp_file as f:
            shutil.copyfileobj(file.file, f)
        temp_path = temp_file.name
        
        logger.info(f"Received file {file.filename}, saved to {temp_path}")
        logger.info(f"Banned users to check: {banned_users}")

        # 1. Run Update (Initial or Incremental)
        logger.info("Starting model update...")
        try:
            config = get_config()
            if not os.path.exists(config.model_path):
                logger.info(f"Model not found at {config.model_path}. Running initial training...")
                # train() returns system, metrics. We just need it to run and save.
                train(temp_path)
            else:
                logger.info(f"Model found. Running incremental update...")
                run_incremental(temp_path)
        except Exception as e:
            logger.error(f"Error during model update: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error updating model: {str(e)}")

        # 2. Run Inference (Check for banned users)
        logger.info("Starting inference for banned users...")
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
                    top_k=5, 
                    current_user_id=user_id, 
                    threshold=0.7, 
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

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
