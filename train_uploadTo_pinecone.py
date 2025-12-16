#!/usr/bin/env python
"""
Train fingerprinting model
"""

from fingerprint_core_uploadTo_pinecone import FingerprintingSystem, FingerprintConfig
import numpy as np
import pandas as pd
import sys
import os
import time
import logging
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"training_{int(time.time())}.log")

# Clear existing handlers (from imports) to ensure our config applies
root_logger = logging.getLogger()
if root_logger.handlers:
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_config():
    """Default training configuration"""
    return FingerprintConfig(
        data_path="test.json",
        cache_path="cached_test_server.parquet", 
        model_path="Parameters/modelTest.pkl",
        max_users=None,
        
        # Adjusted for Full Server:
        # 1. Filter noise
        min_messages=2, 
        
        # 2. Stability
        messages_per_fingerprint=5,
        
        # 3. Overlap
        window_step_size=2,
        
        # 4. Storage/Diversity
        max_fingerprints_per_user=40,
        
        aggregation_method='percentile',
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        use_siamese=False,
    )


def preprocess_data(df):
    """Preprocess dataframe to extract user IDs"""
    if 'ids' not in df.columns and 'author' in df.columns:
        logger.info("Preprocessing: Extracting user IDs from author field...")
        df['ids'] = df['author'].apply(lambda x: x.get('id') if isinstance(x, dict) else None)
    return df


def plot_results(df, train_y, val_y, test_y, system, val_X, output_dir="plots"):
    """Generate and save visualization plots"""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating plots in {output_dir}...")
    
    # 1. Message Distribution
    try:
        plt.figure(figsize=(10, 6))
        user_msg_counts = df['ids'].value_counts()
        plt.hist(user_msg_counts, bins=50, log=True, color='skyblue', edgecolor='black')
        plt.title("Distribution of Messages per User")
        plt.xlabel("Number of Messages")
        plt.ylabel("Count of Users (Log Scale)")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/message_distribution.png")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot message distribution: {e}")
    
    # 2. Fingerprint Distribution
    try:
        plt.figure(figsize=(10, 6))
        all_labels = np.concatenate([train_y, val_y, test_y])
        unique, counts = np.unique(all_labels, return_counts=True)
        plt.hist(counts, bins=30, color='lightgreen', edgecolor='black')
        plt.title("Distribution of Fingerprints per User")
        plt.xlabel("Number of Fingerprints")
        plt.ylabel("Count of Users")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/fingerprint_distribution.png")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot fingerprint distribution: {e}")
    
    # 3. Threshold Tuning Curve (Re-evaluating on validation set)
    try:
        if hasattr(system, 'train_fingerprints') and system.train_fingerprints is not None:
            logger.info("Generating threshold tuning curve...")
            val_scaled = system.scaler.transform(val_X)
            similarities = cosine_similarity(val_scaled, system.train_fingerprints)
            
            thresholds = np.linspace(0.1, 0.9, 50)
            accuracies = []
            
            for thresh in thresholds:
                correct = 0
                total = 0
                for i, sim_row in enumerate(similarities):
                    idx = np.argmax(sim_row)
                    # Only count as a prediction if score > threshold
                    if sim_row[idx] > thresh:
                        pred = system.train_labels[idx]
                        if pred == val_y[i]:
                            correct += 1
                    total += 1 
                
                accuracies.append(correct / total if total > 0 else 0)
                
            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, accuracies, marker='o', linestyle='-', color='purple')
            plt.title("Accuracy vs. Matching Threshold (Validation Set)")
            plt.xlabel("Cosine Similarity Threshold")
            plt.ylabel("Accuracy")
            plt.grid(True)
            plt.savefig(f"{output_dir}/threshold_tuning.png")
            plt.close()
    except Exception as e:
        logger.error(f"Failed to plot threshold curve: {e}")


def train():
    """Train model"""
    start_time = time.time()
    config = get_config()
    system = FingerprintingSystem(config)
    
    logger.info("Loading data...")
    # Set max_records to None to load the entire file in chunks
    df = system.load_data_safe(max_records=None)
    df = preprocess_data(df)
    
    logger.info(f"Loaded {len(df)} messages from {df['ids'].nunique()} users")
    
    logger.info("Creating fingerprints...")
    try:
        train_X, train_y, val_X, val_y, test_X, test_y, weight_X, weight_y = system.prepare_dataset(df)
    except ValueError as e:
        logger.error(f"Error preparing dataset: {e}")
        logger.info("Tip: Check if min_messages or messages_per_fingerprint are too high for your dataset.")
        return

    logger.info(f"Train: {len(train_X)} fingerprints from {len(np.unique(train_y))} users")
    logger.info(f"Test: {len(test_X)} fingerprints from {len(np.unique(test_y))} users")
    
    logger.info("Training...")
    system.train_cosine_similarity(train_X, train_y, val_X, val_y)
    
    metrics = system.evaluate(test_X, test_y, method='cosine')
    logger.info(f"Top-5 Accuracy: {metrics.get('top_5_accuracy', 0):.1%}")
    
    # Generate plots
    plot_results(df, train_y, val_y, test_y, system, val_X)
    
    system.save()
    logger.info(f"Model saved to {config.model_path}")
    
    # Upload to Pinecone if API key is available
    api_key = os.getenv("PINECONE_API_KEY")
    if api_key:
        logger.info("Uploading fingerprints to Pinecone...")
        system.upload_to_pinecone(api_key)
        logger.info("Upload complete.")
    else:
        logger.warning("PINECONE_API_KEY not set. Skipping Pinecone upload.")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
    
    return system, metrics


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'siamese':
        start_time = time.time()
        config = get_config()
        config.use_siamese = True
        config.model_path = "Parameters/siamese_model.pkl"
        system = FingerprintingSystem(config)
        
        logger.info("Loading data...")
        df = system.load_data_safe(max_records=None)
        df = preprocess_data(df)
        
        try:
            train_X, train_y, val_X, val_y, test_X, test_y, weight_X, weight_y = system.prepare_dataset(df)
            
            logger.info("Training Siamese...")
            train_scaled = system.scaler.transform(train_X)
            val_scaled = system.scaler.transform(val_X)
            system.train_siamese_network(train_scaled, train_y, val_scaled, val_y)
            
            metrics = system.evaluate(test_X, test_y, method='siamese')
            logger.info(f"Siamese Top-5 Accuracy: {metrics.get('top_5_accuracy', 0):.1%}")
            
            system.save()
            logger.info(f"Siamese model saved to {config.model_path}")
            
            # Generate plots (partial support for Siamese)
            plot_results(df, train_y, val_y, test_y, system, val_X)
            
        except ValueError as e:
            logger.error(f"Error preparing dataset for Siamese: {e}")
            logger.info("Tip: Ensure your dataset is large enough and properly formatted.")

        # Upload to Pinecone if API key is available
        api_key = os.getenv("PINECONE_API_KEY")
        if api_key:
            logger.info("Uploading fingerprints to Pinecone...")
            system.upload_to_pinecone(api_key)
            logger.info("Upload complete.")
        else:
            logger.warning("PINECONE_API_KEY not set. Skipping Pinecone upload.")
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
    else:
        train()
