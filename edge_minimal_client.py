#!/usr/bin/env python
"""
Production Discord User Fingerprinting Client

Minimal client for querying user fingerprints against Pinecone database.
Designed for production use with error handling and clean API.
"""

import os
from typing import List, Dict, Any, Optional

import numpy as np
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "discord-fingerprints"
MODEL_NAME = "AnnaWegmann/Style-Embedding"


def embed_messages_cpu(messages: List[str]) -> np.ndarray:
    """Create a single embedding for a user from messages.
    
    Args:
        messages: List of message strings from a user
        
    Returns:
        768-dimensional embedding vector
        
    Raises:
        ValueError: If no valid messages provided
    """
    # Clean messages
    clean = [m.strip() for m in messages if m and m.strip()]
    if not clean:
        raise ValueError("No non-empty messages provided")

    # CPU-only model loading
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    
    # Embed messages in batch
    embs = model.encode(clean, show_progress_bar=False, batch_size=32)
    if isinstance(embs, list):
        embs = np.array(embs)

    # Mean pooling across all messages
    return embs.mean(axis=0)


def query_user_fingerprint(messages: List[str], 
                          top_k: int = 5,
                          api_key: Optional[str] = None,
                          index_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Query user fingerprint against database.
    
    Args:
        messages: List of user's messages
        top_k: Number of top matches to return
        api_key: Pinecone API key (overrides env var)
        index_name: Pinecone index name (overrides default)
        
    Returns:
        List of matches with user_id and score
    """
    # Use provided config or defaults
    key = api_key or PINECONE_API_KEY
    index = index_name or INDEX_NAME
    
    if not key:
        raise RuntimeError("PINECONE_API_KEY not set")
    
    # Create embedding
    user_vec = embed_messages_cpu(messages)
    
    # Query Pinecone
    pc = Pinecone(api_key=key)
    idx = pc.Index(index)
    
    response = idx.query(
        vector=user_vec.astype(float).tolist(),
        top_k=top_k,
        include_metadata=True,
    )
    
    # Format results
    matches = response["matches"] if isinstance(response, dict) else response.matches
    return [
        {
            "user_id": m["metadata"].get("user_id") if isinstance(m, dict) else m.metadata.get("user_id"),
            "score": m["score"] if isinstance(m, dict) else m.score
        }
        for m in matches
    ]


def main() -> None:
    """Example usage of the fingerprinting client."""
    if not PINECONE_API_KEY:
        print("Error: Set PINECONE_API_KEY environment variable")
        return
    
    # Example messages (replace with actual user messages)
    messages = [
        "hey, so I was thinking about that match yesterday...",
        "ngl that play was kinda insane lol", 
        "anyway I'll be on later tonight if you wanna queue",
        "did you see what happened in the game though?",
        "that was actually pretty funny tbh"
    ]
    
    try:
        # Query for matches
        matches = query_user_fingerprint(messages, top_k=5)
        
        print(f"Found {len(matches)} matches:")
        for i, match in enumerate(matches, 1):
            print(f"  {i}. User {match['user_id']}: {match['score']:.3f}")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
