#!/usr/bin/env python
"""
Production Discord User Fingerprinting Client

Minimal client for querying user fingerprints against Pinecone database.
Designed for production use with error handling and clean API.
"""

import os
import json
import sys
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from pinecone import Pinecone

# Import core fingerprinting logic
# Ensure the current directory is in the path to import the sibling file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fingerprint_core_uploadTo_pinecone import FingerprintConfig, FingerprintExtractor

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "discord-fingerprints-test"


def get_user_centroid_from_pinecone(user_id: str, 
                                   api_key: str, 
                                   index_name: str) -> Optional[np.ndarray]:
    """Fetch all fingerprints for a user from Pinecone and calculate centroid."""
    print(f"Fetching fingerprints for user {user_id} from Pinecone...")
    pc = Pinecone(api_key=api_key)
    idx = pc.Index(index_name)
    
    # 1. List all IDs for this user
    # IDs are stored as f"{user_id}_{global_index}"
    # We use prefix search. Note: user_id should be exact, so we append "_" to ensure we don't match user_id_other
    prefix = f"{user_id}_"
    
    user_vector_ids = []
    try:
        # list returns a generator of IDs (Serverless indexes only)
        for ids in idx.list(prefix=prefix):
            user_vector_ids.extend(ids)
    except Exception as e:
        print(f"Error listing IDs for user {user_id}: {e}")
        return None
        
    if not user_vector_ids:
        print(f"No fingerprints found in Pinecone for user {user_id}")
        return None
    
    print(f"Found {len(user_vector_ids)} fingerprints. Downloading...")
        
    # 2. Fetch vectors in batches
    vectors = []
    batch_size = 100
    for i in range(0, len(user_vector_ids), batch_size):
        batch_ids = user_vector_ids[i:i+batch_size]
        fetch_response = idx.fetch(ids=batch_ids)
        
        for v_id in batch_ids:
            if v_id in fetch_response.vectors:
                vectors.append(fetch_response.vectors[v_id].values)
                
    if not vectors:
        return None
        
    # 3. Calculate centroid
    centroid = np.mean(vectors, axis=0)
    return centroid


def query_user_fingerprint(messages_df: Optional[pd.DataFrame] = None, 
                          extractor: Optional[FingerprintExtractor] = None,
                          top_k: int = 5,
                          api_key: Optional[str] = None,
                          index_name: Optional[str] = None,
                          current_user_id: Optional[str] = None,
                          threshold: float = 0.0,
                          user_vector: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Query user fingerprint against database.
    
    Args:
        messages_df: DataFrame of user's messages (optional if user_vector provided)
        extractor: Initialized FingerprintExtractor instance (optional if user_vector provided)
        top_k: Number of top unique user matches to return
        api_key: Pinecone API key (overrides env var)
        index_name: Pinecone index name (overrides default)
        current_user_id: ID of the user being queried (to exclude from results)
        threshold: Minimum similarity score to include in results
        user_vector: Pre-calculated vector to use for query (overrides messages_df)
        
    Returns:
        List of matches with user_id and score
    """
    # Use provided config or defaults
    key = api_key or PINECONE_API_KEY
    index = index_name or INDEX_NAME
    
    if not key:
        raise RuntimeError("PINECONE_API_KEY not set")
    
    # Determine query vector
    if user_vector is not None:
        user_vec = user_vector
    elif messages_df is not None and extractor is not None:
        user_vec = extractor.create_fingerprint(messages_df)
    else:
        raise ValueError("Must provide either user_vector or (messages_df and extractor)")
    
    if user_vec is None:
        raise ValueError("Insufficient messages to generate fingerprint")
    
    # Query Pinecone
    pc = Pinecone(api_key=key)
    idx = pc.Index(index)
    
    # Query for more results to ensure we find enough unique users
    # We ask for 10x the requested top_k to filter through duplicates
    fetch_k = top_k * 10
    
    response = idx.query(
        vector=user_vec.astype(float).tolist(),
        top_k=fetch_k,
        include_metadata=True,
    )
    
    # Format results and filter for unique users
    raw_matches = response["matches"] if isinstance(response, dict) else response.matches
    
    unique_matches = []
    seen_users = set()
    
    for m in raw_matches:
        user_id = m["metadata"].get("user_id") if isinstance(m, dict) else m.metadata.get("user_id")
        score = m["score"] if isinstance(m, dict) else m.score
        
        # Filter out self-matches
        # Note: When testing centroid, we usually WANT to see if the user matches themselves,
        # so we might want to disable this check or make it optional.
        # For now, we keep it but print a note if we find ourselves.
        if current_user_id and str(user_id) == str(current_user_id):
            # We found ourselves! This is good for accuracy testing.
            # If we want to see ourselves in the list, we should NOT continue here.
            # But the original code filtered it out to find "other" similar users.
            # Let's include ourselves for this specific test case.
            pass 
            
        # Filter by threshold
        if score < threshold:
            continue
        
        if user_id not in seen_users:
            unique_matches.append({
                "user_id": user_id,
                "score": score
            })
            seen_users.add(user_id)
            
        if len(unique_matches) >= top_k:
            break
            
    return unique_matches


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess dataframe to extract user IDs"""
    if 'ids' not in df.columns and 'author' in df.columns:
        print("Preprocessing: Extracting user IDs from author field...")
        df['ids'] = df['author'].apply(lambda x: x.get('id') if isinstance(x, dict) else None)
    return df


def load_user_messages(file_path: str, user_ids: List[str]) -> Dict[str, pd.DataFrame]:
    """Load messages for specific users from a JSON file.
    
    Args:
        file_path: Path to the JSON file (NDJSON/JSON Lines)
        user_ids: List of user IDs to extract messages for
        
    Returns:
        Dictionary mapping user_id to DataFrame of their messages
    """
    print(f"Loading data from {file_path}...")
    
    # Read JSON lines into a list of dicts first to handle potential malformed lines gracefully
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}

    if not data:
        print("No data found in file.")
        return {}

    df = pd.DataFrame(data)
    df = preprocess_data(df)
    
    # Filter for target users
    target_df = df[df['ids'].isin(user_ids)]
    
    # Group messages by user
    user_messages = {}
    for user_id in user_ids:
        # Get messages for this user
        user_df = target_df[target_df['ids'] == user_id][['content', 'timestamp']].copy()
        
        # Filter out empty messages
        user_df = user_df[user_df['content'].notna() & (user_df['content'] != "")]
        
        if not user_df.empty:
            user_messages[user_id] = user_df
            
    return user_messages


def main() -> None:
    """Example usage of the fingerprinting client."""
    if not PINECONE_API_KEY:
        print("Error: Set PINECONE_API_KEY environment variable")
        return
    
    # Configuration for batch processing
    # INPUT_FILE = "test.json" # Not used in centroid mode
    
    # Example user IDs from the provided test.json
    TARGET_USERS = [
        "2f85e196681d",  # Anthony Reilly
        "ecfeadbb695e"   # Jayson Bond
    ]
    
    try:
        # Process each user using Pinecone Centroid
        for user_id in TARGET_USERS:
            print(f"\nProcessing User {user_id} (Centroid Mode)...")
            
            try:
                # 1. Get Centroid from Pinecone
                centroid = get_user_centroid_from_pinecone(
                    user_id, 
                    PINECONE_API_KEY, 
                    INDEX_NAME
                )
                
                if centroid is None:
                    print(f"  Skipping user {user_id}: No fingerprints found in DB.")
                    continue

                # 2. Query for matches using the centroid
                matches = query_user_fingerprint(
                    messages_df=None, 
                    extractor=None, 
                    top_k=5,
                    current_user_id=None, # Pass None so we can see if the user matches themselves
                    threshold=0.1,
                    user_vector=centroid
                )
                
                print(f"Found {len(matches)} matches for User {user_id}:")
                for i, match in enumerate(matches, 1):
                    is_self = " (SELF)" if str(match['user_id']) == str(user_id) else ""
                    print(f"  {i}. User {match['user_id']}: {match['score']:.3f}{is_self}")
                    
            except ValueError as ve:
                print(f"  Skipping user {user_id}: {ve}")
            except Exception as e:
                print(f"  Error processing user {user_id}: {e}")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
