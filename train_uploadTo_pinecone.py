#!/usr/bin/env python
"""
Train fingerprinting model
"""

from fingerprint_core_uploadTo_pinecone import FingerprintingSystem, FingerprintConfig
import numpy as np
import pandas as pd
import sys
import os

def get_config():
    """Default training configuration"""
    return FingerprintConfig(
        data_path="test.json",
        cache_path="cached_test.parquet", 
        model_path="models/model.pkl",
        max_users=None,
        # Adjusted for small test.json dataset
        min_messages=2, 
        messages_per_fingerprint=2,
        window_step_size=1,
        max_fingerprints_per_user=30,
        aggregation_method='percentile',
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        use_siamese=False,
    )


def preprocess_data(df):
    """Preprocess dataframe to extract user IDs"""
    if 'ids' not in df.columns and 'author' in df.columns:
        print("Preprocessing: Extracting user IDs from author field...")
        df['ids'] = df['author'].apply(lambda x: x.get('id') if isinstance(x, dict) else None)
    return df


def train():
    """Train model"""
    config = get_config()
    system = FingerprintingSystem(config)
    
    print("Loading data...")
    # Use the safe loader from system
    df = system.load_data_safe()
    df = preprocess_data(df)
    
    print(f"Loaded {len(df)} messages from {df['ids'].nunique()} users")
    
    print("Creating fingerprints...")
    try:
        train_X, train_y, val_X, val_y, test_X, test_y = system.prepare_dataset(df)
    except ValueError as e:
        print(f"Error preparing dataset: {e}")
        print("Tip: Check if min_messages or messages_per_fingerprint are too high for your dataset.")
        return

    print(f"Train: {len(train_X)} fingerprints from {len(np.unique(train_y))} users")
    print(f"Test: {len(test_X)} fingerprints from {len(np.unique(test_y))} users")
    
    print("Training...")
    system.train_cosine_similarity(train_X, train_y, val_X, val_y)
    
    metrics = system.evaluate(test_X, test_y, method='cosine')
    print(f"Top-5 Accuracy: {metrics.get('top_5_accuracy', 0):.1%}")
    
    system.save()
    print(f"Model saved to {config.model_path}")
        # Upload to Pinecone if API key is available
    api_key = os.getenv("PINECONE_API_KEY")
    if api_key:
        print("Uploading fingerprints to Pinecone...")
        system.upload_to_pinecone(api_key)
        print("Upload complete.")
    else:
        print("Warning: PINECONE_API_KEY not set. Skipping Pinecone upload.")
    
    return system, metrics



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'siamese':
        config = get_config()
        config.use_siamese = True
        config.model_path = "models/siamese_model.pkl"
        system = FingerprintingSystem(config)
        
        print("Loading data...")
        df = system.load_data_safe()
        df = preprocess_data(df)
        
        try:
            train_X, train_y, val_X, val_y, test_X, test_y = system.prepare_dataset(df)
            
            print("Training Siamese...")
            train_scaled = system.scaler.transform(train_X)
            val_scaled = system.scaler.transform(val_X)
            system.train_siamese_network(train_scaled, train_y, val_scaled, val_y)
            
            metrics = system.evaluate(test_X, test_y, method='siamese')
            print(f"Siamese Top-5 Accuracy: {metrics.get('top_5_accuracy', 0):.1%}")
            
            system.save()
            print(f"Siamese model saved to {config.model_path}")
        except ValueError as e:
            print(f"Error preparing dataset for Siamese: {e}")
            print("Tip: Ensure your dataset is large enough and properly formatted.")

                # Upload to Pinecone if API key is available
        api_key = os.getenv("PINECONE_API_KEY")
        if api_key:
            print("Uploading fingerprints to Pinecone...")
            system.upload_to_pinecone(api_key)
            print("Upload complete.")
        else:
            print("Warning: PINECONE_API_KEY not set. Skipping Pinecone upload.")
    else:
        train()
