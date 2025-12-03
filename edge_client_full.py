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


def query_user_fingerprint(messages_df: pd.DataFrame, 
                          extractor: FingerprintExtractor,
                          top_k: int = 5,
                          api_key: Optional[str] = None,
                          index_name: Optional[str] = None,
                          current_user_id: Optional[str] = None,
                          threshold: float = 0.0) -> List[Dict[str, Any]]:
    """Query user fingerprint against database.
    
    Args:
        messages_df: DataFrame of user's messages with content and timestamp
        extractor: Initialized FingerprintExtractor instance
        top_k: Number of top unique user matches to return
        api_key: Pinecone API key (overrides env var)
        index_name: Pinecone index name (overrides default)
        current_user_id: ID of the user being queried (to exclude from results)
        threshold: Minimum similarity score to include in results
        
    Returns:
        List of matches with user_id and score
    """
    # Use provided config or defaults
    key = api_key or PINECONE_API_KEY
    index = index_name or INDEX_NAME
    
    if not key:
        raise RuntimeError("PINECONE_API_KEY not set")
    
    # Create fingerprint using the core extractor
    user_vec = extractor.create_fingerprint(messages_df)
    
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
        if current_user_id and str(user_id) == str(current_user_id):
            continue
            
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
    INPUT_FILE = "test.json"
    # Example user IDs from the provided test.json
    TARGET_USERS = [
        "2f85e196681d",  # Anthony Reilly
        "ecfeadbb695e"   # Jayson Bond
    ]
    
    # Initialize Extractor with CPU config and low message threshold for testing
    # We set messages_per_fingerprint to 2 so that even users with few messages 
    # (like 4, 9, 15) can generate a fingerprint.
    config = FingerprintConfig(
        device="cpu",
        messages_per_fingerprint=2, 
        aggregation_method='percentile'
    )
    extractor = FingerprintExtractor(config)
    
    try:
        # Load messages for the target users
        user_messages_map = load_user_messages(INPUT_FILE, TARGET_USERS)
        
        if not user_messages_map:
            print("No messages found for the specified users.")
            return

        # Process each user
        for user_id, messages_df in user_messages_map.items():
            print(f"\nProcessing User {user_id} ({len(messages_df)} messages)...")
            
            try:
                # Query for matches
                matches = query_user_fingerprint(
                    messages_df, 
                    extractor, 
                    top_k=5,
                    current_user_id=user_id,
                    threshold=0.1
                )
                
                print(f"Found {len(matches)} matches for User {user_id}:")
                for i, match in enumerate(matches, 1):
                    print(f"  {i}. User {match['user_id']}: {match['score']:.3f}")
            except ValueError as ve:
                print(f"  Skipping user {user_id}: {ve}")
            except Exception as e:
                print(f"  Error processing user {user_id}: {e}")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
