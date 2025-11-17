#!/usr/bin/env python
"""
Train fingerprinting model
"""

from fingerprinting import FingerprintingSystem, FingerprintConfig
import numpy as np
import sys


def get_config():
    """Default training configuration"""
    return FingerprintConfig(
        data_path="cached_messages.parquet",
        cache_path="cached_messages.parquet", 
        model_path="models/model.pkl",
        max_users=None,
        min_messages=100,
        messages_per_fingerprint=100,
        window_step_size=50,
        max_fingerprints_per_user=30,
        aggregation_method='percentile',
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        use_siamese=False,
    )


def train():
    """Train model"""
    config = get_config()
    system = FingerprintingSystem(config)
    
    print("Loading data...")
    df = system.load_data(use_cache=True)
    print(f"Loaded {len(df)} messages from {df['ids'].nunique()} users")
    
    print("Creating fingerprints...")
    train_X, train_y, val_X, val_y, test_X, test_y = system.prepare_dataset(df)
    
    print(f"Train: {len(train_X)} fingerprints from {len(np.unique(train_y))} users")
    print(f"Test: {len(test_X)} fingerprints from {len(np.unique(test_y))} users")
    
    print("Training...")
    system.train_cosine_similarity(train_X, train_y, val_X, val_y)
    
    metrics = system.evaluate(test_X, test_y, method='cosine')
    print(f"Top-5 Accuracy: {metrics.get('top_5_accuracy', 0):.1%}")
    
    system.save()
    print(f"Model saved to {config.model_path}")
    
    return system, metrics


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'siamese':
        config = get_config()
        config.use_siamese = True
        config.model_path = "models/siamese_model.pkl"
        system = FingerprintingSystem(config)
        
        print("Loading data...")
        df = system.load_data(use_cache=True)
        train_X, train_y, val_X, val_y, test_X, test_y = system.prepare_dataset(df)
        
        print("Training Siamese...")
        train_scaled = system.scaler.transform(train_X)
        val_scaled = system.scaler.transform(val_X)
        system.train_siamese_network(train_scaled, train_y, val_scaled, val_y)
        
        metrics = system.evaluate(test_X, test_y, method='siamese')
        print(f"Siamese Top-5 Accuracy: {metrics.get('top_5_accuracy', 0):.1%}")
        
        system.save()
        print(f"Siamese model saved to {config.model_path}")
    else:
        train()
