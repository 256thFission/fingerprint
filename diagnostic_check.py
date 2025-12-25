#!/usr/bin/env python
"""
Diagnostic script for Fingerprinting System
"""
import os
import sys
import logging
import argparse
from dotenv import load_dotenv
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fingerprint_core_uploadTo_pinecone import FingerprintingSystem, FingerprintConfig

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def run_diagnostic(args):
    logger.info("Starting diagnostic check...")
    
    # 1. Check Environment
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        logger.error("PINECONE_API_KEY not found in environment variables.")
        if args.upload:
            logger.error("Cannot proceed with upload check.")
            return False
        else:
            logger.warning("Proceeding without API key (upload skipped).")

    # 2. Configure System for Test Data
    # test.json has ~49 lines. We need to lower thresholds.
    config = FingerprintConfig(
        data_path=args.input_file,
        cache_path="diagnostic_cache.parquet",
        model_path="diagnostic_model.pkl",
        min_messages=5,  # Lowered for test.json
        messages_per_fingerprint=5, # Lowered for test.json
        window_step_size=2,
        max_fingerprints_per_user=5,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        device="cpu" # Force CPU for simple diagnostic
    )
    
    logger.info(f"Configuration: min_messages={config.min_messages}, messages_per_fingerprint={config.messages_per_fingerprint}")

    try:
        system = FingerprintingSystem(config)
        
        # 3. Load Data
        logger.info(f"Loading data from {config.data_path}...")
        df = system.load_data(use_cache=False) # Force reload
        
        # Preprocess if needed (extract ids from author)
        if 'ids' not in df.columns and 'author' in df.columns:
            logger.info("Preprocessing: Extracting user IDs from author field...")
            df['ids'] = df['author'].apply(lambda x: x.get('id') if isinstance(x, dict) else None)
            
        logger.info(f"Loaded {len(df)} messages.")
        
        if len(df) == 0:
            logger.error("No data loaded.")
            return False

        # 4. Generate Fingerprints
        logger.info("Generating fingerprints...")
        train_X, train_y, val_X, val_y, test_X, test_y = system.prepare_dataset(df)
        
        total_fingerprints = len(train_X) + len(val_X) + len(test_X)
        logger.info(f"Generated {total_fingerprints} fingerprints.")
        
        if total_fingerprints == 0:
            logger.error("No fingerprints generated.")
            return False
            
        # 5. Train Model (Cosine)
        logger.info("Training cosine similarity model...")
        system.train_cosine_similarity(train_X, train_y, val_X, val_y)
        
        if system.train_fingerprints is None:
            logger.error("Model training failed (train_fingerprints is None).")
            return False
            
        logger.info("Model trained successfully.")

        # 6. Upload to Pinecone (Optional)
        if args.upload and api_key:
            index_name = args.index_name
            logger.info(f"Attempting upload to Pinecone index: {index_name}")
            try:
                system.upload_to_pinecone(api_key, index_name)
                logger.info("Upload function completed without error.")
                
                # Verify upload by querying (simple check)
                from pinecone import Pinecone
                pc = Pinecone(api_key=api_key)
                
                # Check if index exists
                if index_name in pc.list_indexes().names():
                    idx = pc.Index(index_name)
                    stats = idx.describe_index_stats()
                    logger.info(f"Index Stats: {stats}")
                    
                    if stats.total_vector_count > 0:
                        logger.info("Verification successful: Index contains vectors.")
                    else:
                        logger.warning("Verification warning: Index is empty (might take a moment to update).")
                else:
                    logger.error(f"Index {index_name} was not found after upload attempt.")
                    
            except Exception as e:
                logger.error(f"Pinecone upload failed: {e}")
                return False
        else:
            logger.info("Skipping Pinecone upload (use --upload to enable).")

        logger.info("Diagnostic check passed!")
        return True

    except Exception as e:
        logger.error(f"Diagnostic check failed with exception: {e}", exc_info=True)
        return False
    finally:
        # Cleanup
        if os.path.exists("diagnostic_cache.parquet"):
            try:
                os.remove("diagnostic_cache.parquet")
            except:
                pass
        if os.path.exists("diagnostic_model.pkl"):
            try:
                os.remove("diagnostic_model.pkl")
            except:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fingerprint System Diagnostic")
    parser.add_argument("--input_file", default="test.json", help="Path to input JSON file")
    parser.add_argument("--upload", action="store_true", help="Attempt upload to Pinecone")
    parser.add_argument("--index_name", default="diagnostic-test-index", help="Pinecone index name for testing")
    
    args = parser.parse_args()
    
    success = run_diagnostic(args)
    sys.exit(0 if success else 1)
