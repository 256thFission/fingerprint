#!/usr/bin/env python
"""
Core fingerprinting system
"""

import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from pathlib import Path
import pickle
import logging
import re
import os
from collections import Counter
import torch.nn.functional as F
from pinecone import Pinecone, ServerlessSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FingerprintConfig:
    data_path: str = "data/messages.json"
    cache_path: str = "data/cached_messages.parquet"
    model_path: str = "models/model.pkl"
    min_messages: int = 50
    messages_per_fingerprint: int = 50
    window_step_size: int = 50
    session_timeout_seconds: int = 1800
    max_fingerprints_per_user: int = 30
    max_users: Optional[int] = None
    aggregation_method: str = 'percentile'
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    use_siamese: bool = False
    siamese_embedding_dim: int = 128
    siamese_epochs: int = 20
    siamese_batch_size: int = 64
    siamese_lr: float = 1e-3
    match_threshold: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def validate(self):
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6
        assert self.aggregation_method in ['percentile', 'mean']


def extract_linguistic_features(texts: List[str]) -> np.ndarray:
    """Extract basic linguistic features"""
    if len(texts) == 0:
        return np.zeros(21)
    
    lengths = [len(text) for text in texts]
    word_counts = [len(text.split()) for text in texts]
    all_words = ' '.join(texts).split()
    all_text = ' '.join(texts)
    total_chars = len(all_text) if all_text else 1
    
    features = [
        np.mean(lengths), np.std(lengths), np.median(lengths),
        np.mean(word_counts), np.std(word_counts),
        np.mean([len(word) for word in all_words]) if all_words else 0,
        len(set(all_words)) / len(all_words) if all_words else 0,
        all_text.count('!') / total_chars,
        all_text.count('?') / total_chars,
        all_text.count('.') / total_chars,
        all_text.count(',') / total_chars,
        all_text.count('...') / len(texts),
        sum(1 for c in all_text if c.isupper()) / total_chars,
        sum(1 for text in texts if text and text[0].isupper()) / len(texts),
        sum(1 for text in texts if text.isupper()) / len(texts),
        sum(text.count(':)') + text.count(':D') + text.count(':(') for text in texts) / len(texts),
        sum(text.count('\n') for text in texts) / len(texts),
        sum(1 for text in texts if any(c.isdigit() for c in text)) / len(texts),
        sum(1 for text in texts if text.strip().lower().startswith(('lol', 'lmao', 'haha'))) / len(texts),
        sum(1 for text in texts if text.strip().lower().startswith(('yeah', 'yea', 'yes'))) / len(texts),
    ]
    
    return np.array(features)


def aggregate_embeddings(embeddings: np.ndarray, method: str = 'percentile') -> np.ndarray:
    """Aggregate message embeddings"""
    if len(embeddings) == 0:
        return np.zeros(768 if method == 'mean' else 768 * 4)
    
    if method == 'percentile':
        features = np.concatenate([
            np.mean(embeddings, axis=0),
            np.std(embeddings, axis=0),
            np.percentile(embeddings, 25, axis=0),
            np.percentile(embeddings, 75, axis=0),
        ])
    else:
        features = np.mean(embeddings, axis=0)
    
    return features


def circular_kde(samples: np.ndarray, grid_points: np.ndarray, bandwidth: float = 0.05) -> np.ndarray:
    """Circular KDE for periodic data"""
    if len(samples) == 0:
        return np.zeros_like(grid_points)
    
    augmented_samples = np.concatenate([samples, samples - 1, samples + 1])
    kde = gaussian_kde(augmented_samples, bw_method=bandwidth)
    density = kde(grid_points)
    density[density < 0] = 0
    
    return density / (density.sum() + 1e-9)


def extract_temporal_features(timestamps: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract temporal activity patterns"""
    if len(timestamps) == 0:
        return np.zeros(24), np.zeros(7), np.zeros(4)
    
    times = pd.to_datetime(timestamps)
    hour_frac = (times.dt.hour + times.dt.minute / 60 + times.dt.second / 3600) / 24.0
    dow_frac = (times.dt.dayofweek + 0.5) / 7.0
    wom_frac = np.clip(((times.dt.day - 1) // 7) / 4.0, 0, 0.99)
    
    hour_grid = np.linspace(0, 1, 24, endpoint=False)
    dow_grid = np.linspace(0, 1, 7, endpoint=False)
    wom_grid = np.linspace(0, 1, 4, endpoint=False)
    
    hour_density = circular_kde(hour_frac.values, hour_grid, bandwidth=0.05)
    dow_density = circular_kde(dow_frac.values, dow_grid, bandwidth=0.1)
    wom_density = circular_kde(wom_frac.values, wom_grid, bandwidth=0.1)
    
    return hour_density, dow_density, wom_density


def evaluate_matching(predictions: List[Optional[str]], true_labels: List[str]) -> Dict[str, float]:
    """Evaluate matching performance"""
    correct = 0
    matched = 0
    
    for pred, true in zip(predictions, true_labels):
        if pred is not None:
            matched += 1
            if pred == true:
                correct += 1
    
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0
    precision = correct / matched if matched > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'matched_count': matched,
        'total': total,
        'correct': correct
    }


class FingerprintExtractor:
    """Extract fingerprints from user messages"""
    
    def __init__(self, config: FingerprintConfig):
        self.config = config
        self.model = SentenceTransformer("AnnaWegmann/Style-Embedding", device=config.device)
        self._embedding_cache = {}
        
    def create_fingerprint(self, messages: pd.DataFrame) -> Optional[np.ndarray]:
        """Create single fingerprint from messages"""
        texts = messages["content"].fillna("").astype(str).values
        texts = [t for t in texts if t.strip()]
        
        if len(texts) < self.config.messages_per_fingerprint // 2:
            return None
        
        embeddings = self.model.encode(texts, show_progress_bar=False, batch_size=32)
        text_features = aggregate_embeddings(embeddings, method=self.config.aggregation_method)
        linguistic_features = extract_linguistic_features(texts)
        hour_density, dow_density, wom_density = extract_temporal_features(messages["timestamp"])
        
        fingerprint = np.concatenate([
            text_features,
            linguistic_features,
            hour_density,
            dow_density,
            wom_density,
        ])
        
        return fingerprint
    
    def create_fingerprints_for_user(self, user_messages: pd.DataFrame) -> List[np.ndarray]:
        """Create multiple fingerprints for user using timestamp clustering"""
        fingerprints = []
        
        if user_messages.empty:
            return fingerprints

        # Ensure sorted by timestamp
        user_messages = user_messages.sort_values('timestamp')
        
        # Convert timestamps to seconds for clustering
        timestamps = pd.to_datetime(user_messages['timestamp'], errors='coerce')
        
        # Filter out invalid timestamps
        valid_mask = timestamps.notna()
        if not valid_mask.any():
            return []
            
        user_messages = user_messages.loc[valid_mask]
        timestamps = timestamps.loc[valid_mask]
        
        # Convert to unix timestamp (seconds). 
        # Note: astype(np.int64) on datetime64[ns] gives nanoseconds
        X = timestamps.astype(np.int64) // 10**9
        X = X.values.reshape(-1, 1)
        
        # Use DBSCAN to cluster messages based on time proximity
        # eps: max time gap in seconds (session_timeout_seconds)
        # min_samples: 1 ensures we keep all points initially, then filter by size
        clustering = DBSCAN(eps=self.config.session_timeout_seconds, min_samples=1).fit(X)
        
        labels = clustering.labels_
        unique_labels = np.unique(labels)
        
        # Collect clusters and sort by time
        clusters = []
        for label in unique_labels:
            if label == -1:
                continue
                
            # Get messages for this cluster
            mask = labels == label
            cluster_batch = user_messages.iloc[mask]
            
            # Store with start time for sorting
            if not cluster_batch.empty:
                clusters.append((cluster_batch.iloc[0]['timestamp'], cluster_batch))
        
        # Sort clusters by start time
        clusters.sort(key=lambda x: x[0])
        
        # Accumulate small clusters until we reach target size
        current_batch_dfs = []
        current_batch_size = 0
        
        for _, cluster_batch in clusters:
            current_batch_dfs.append(cluster_batch)
            current_batch_size += len(cluster_batch)
            
            # If we've accumulated enough messages, create a fingerprint
            if current_batch_size >= self.config.messages_per_fingerprint:
                combined_batch = pd.concat(current_batch_dfs)
                
                fingerprint = self.create_fingerprint(combined_batch)
                if fingerprint is not None:
                    fingerprints.append(fingerprint)
                
                # Reset batch
                current_batch_dfs = []
                current_batch_size = 0
            
            if (self.config.max_fingerprints_per_user and 
                len(fingerprints) >= self.config.max_fingerprints_per_user):
                break
        
        return fingerprints


class SimpleSiameseNetwork(nn.Module):
    """Basic Siamese network"""
    
    def __init__(self, input_dim: int, embedding_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
        )
    
    def forward(self, x):
        return F.normalize(self.encoder(x), dim=1)


class FingerprintingSystem:
    """Main fingerprinting system"""
    
    def __init__(self, config: FingerprintConfig):
        self.config = config
        self.config.validate()
        self.extractor = FingerprintExtractor(config)
        self.scaler = StandardScaler()
        self.siamese_model = None
        self.train_fingerprints = None
        self.train_labels = None
        self.threshold = config.match_threshold
        
    def load_data(self, use_cache: bool = True) -> pd.DataFrame:
        """Load message data"""
        cache_path = Path(self.config.cache_path)
        
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            df = pd.read_parquet(cache_path)
        else:
            logger.info(f"Loading data from {self.config.data_path}")
            df = pd.read_json(self.config.data_path, lines=True)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
        
        return df
    
    def load_data_safe(self, max_records: int = 500000) -> pd.DataFrame:
        """Load data safely, using chunks for large JSON files to avoid OOM"""
        if self.config.data_path.endswith('.json'):
            if not os.path.exists(self.config.data_path):
                 raise FileNotFoundError(f"File not found: {self.config.data_path}")
                 
            logger.info(f"Loading data from {self.config.data_path}...")
            
            chunks = []
            total_loaded = 0
            try:
                # Read in chunks to avoid loading entire file into RAM at once
                chunk_iterator = pd.read_json(self.config.data_path, lines=True, chunksize=10000)
                for chunk in chunk_iterator:
                    chunks.append(chunk)
                    total_loaded += len(chunk)
                    if max_records and total_loaded >= max_records:
                        logger.info(f"Reached safety limit of {max_records} records. Stopping load.")
                        break
                
                if not chunks:
                    return pd.DataFrame()
                    
                return pd.concat(chunks, ignore_index=True)
            except ValueError:
                # Fallback for small files that might not like chunksize
                return pd.read_json(self.config.data_path, lines=True)
        else:
            return self.load_data(use_cache=True)
    
    def prepare_dataset(self, df: pd.DataFrame) -> Tuple:
        """Prepare train/val/test datasets"""
        all_fingerprints = []
        all_labels = []
        
        user_counts = df['ids'].value_counts()
        valid_users = user_counts[user_counts >= self.config.min_messages].index
        
        if self.config.max_users:
            valid_users = valid_users[:self.config.max_users]
        
        logger.info(f"Processing {len(valid_users)} users")
        
        for user_id in valid_users:
            user_messages = df[df['ids'] == user_id].sort_values('timestamp')
            fingerprints = self.extractor.create_fingerprints_for_user(user_messages)
            
            for fp in fingerprints:
                all_fingerprints.append(fp)
                all_labels.append(user_id)
        
        if not all_fingerprints:
            raise ValueError("No fingerprints created")
        
        X = np.array(all_fingerprints)
        y = np.array(all_labels)
        
        # Time-based split
        unique_users = np.unique(y)
        train_X, train_y, val_X, val_y, test_X, test_y = [], [], [], [], [], []
        
        for user in unique_users:
            user_indices = np.where(y == user)[0]
            n = len(user_indices)
            
            train_end = int(n * self.config.train_ratio)
            # Ensure at least one sample for training if data exists
            if train_end == 0 and n > 0:
                train_end = 1
                
            val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
            # Ensure val_end is consistent
            if val_end < train_end:
                val_end = train_end
            
            train_X.extend(X[user_indices[:train_end]])
            train_y.extend(y[user_indices[:train_end]])
            val_X.extend(X[user_indices[train_end:val_end]])
            val_y.extend(y[user_indices[train_end:val_end]])
            test_X.extend(X[user_indices[val_end:]])
            test_y.extend(y[user_indices[val_end:]])

        return (np.array(train_X), np.array(train_y), 
                np.array(val_X), np.array(val_y),
                np.array(test_X), np.array(test_y))
    
    def train_cosine_similarity(self, train_X, train_y, val_X, val_y):
        """Train cosine similarity baseline"""
        if len(train_X) == 0:
            raise ValueError("Training data is empty. Cannot train model.")

        self.train_fingerprints = self.scaler.fit_transform(train_X)
        self.train_labels = train_y
        
        # Simple threshold tuning
        if len(val_X) > 0:
            val_scaled = self.scaler.transform(val_X)
            similarities = cosine_similarity(val_scaled, self.train_fingerprints)
            
            best_acc = 0
            best_thresh = self.threshold
            
            for thresh in np.linspace(0.1, 0.9, 50):
                correct = 0
                total = 0
                for i, sim_row in enumerate(similarities):
                    idx = np.argmax(sim_row)
                    if sim_row[idx] > thresh:
                        pred = self.train_labels[idx]
                        if pred == val_y[i]:
                            correct += 1
                    total += 1
                
                acc = correct / total if total > 0 else 0
                if acc > best_acc:
                    best_acc = acc
                    best_thresh = thresh
            
            self.threshold = best_thresh
            logger.info(f"Best threshold: {self.threshold:.3f}")
        else:
            logger.info("Skipping threshold tuning (no validation data)")
    
    def train_siamese_network(self, train_X, train_y, val_X, val_y):
        """Train Siamese network"""
        if not self.config.use_siamese:
            return
        
        device = torch.device(self.config.device)
        self.siamese_model = SimpleSiameseNetwork(train_X.shape[1]).to(device)
        
        # Simple pairwise dataset
        dataset = []
        for user in np.unique(train_y):
            user_indices = np.where(train_y == user)[0]
            if len(user_indices) >= 2:
                for _ in range(10):
                    i, j = np.random.choice(user_indices, 2)
                    dataset.append((train_X[i], train_X[j], 1))
                    
                    other_user = np.random.choice([u for u in np.unique(train_y) if u != user])
                    other_indices = np.where(train_y == other_user)[0]
                    k = np.random.choice(other_indices)
                    dataset.append((train_X[i], train_X[k], 0))
        
        train_loader = DataLoader(dataset, batch_size=self.config.siamese_batch_size, shuffle=True)
        optimizer = optim.Adam(self.siamese_model.parameters(), lr=self.config.siamese_lr)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(self.config.siamese_epochs):
            self.siamese_model.train()
            total_loss = 0
            
            for x1, x2, labels in train_loader:
                x1, x2, labels = (torch.tensor(x1, dtype=torch.float32).to(device),
                                 torch.tensor(x2, dtype=torch.float32).to(device),
                                 torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device))
                
                e1, e2 = self.siamese_model(x1), self.siamese_model(x2)
                logits = torch.sum(e1 * e2, dim=1, keepdim=True)
                loss = criterion(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.siamese_epochs} - Loss: {total_loss/len(train_loader):.4f}")
    
    def evaluate(self, test_X, test_y, method='cosine') -> Dict:
        """Evaluate model"""
        train_emb = None
        device = torch.device(self.config.device)

        # 1. Calculate similarities for all test fingerprints (Fingerprint Level)
        if method == 'cosine':
            test_scaled = self.scaler.transform(test_X)
            similarities = cosine_similarity(test_scaled, self.train_fingerprints)
        elif method == 'siamese' and self.siamese_model:
            self.siamese_model.eval()
            with torch.no_grad():
                test_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
                train_tensor = torch.tensor(self.train_fingerprints, dtype=torch.float32).to(device)
                
                test_emb = self.siamese_model(test_tensor)
                train_emb = self.siamese_model(train_tensor)
                
                similarities = torch.mm(test_emb, train_emb.T).cpu().numpy()
        else:
            return {}
        
        # --- Metric 1: Per-Fingerprint Accuracy (Legacy/Granular) ---
        n_test = similarities.shape[0]
        sorted_indices = np.argsort(-similarities, axis=1)
        
        fp_top1 = sum(1 for i in range(n_test) if self.train_labels[sorted_indices[i, 0]] == test_y[i])
        fp_top5 = sum(1 for i in range(n_test) if test_y[i] in self.train_labels[sorted_indices[i, :5]])
        
        # --- Metric 2: User-Level Accuracy (Representative Embedding) ---
        # Simulate the client scenario: Combine all test segments for a user into one query vector
        unique_test_users = np.unique(test_y)
        user_top1 = 0
        user_top5 = 0
        
        for user in unique_test_users:
            # Find indices for this user in the test set
            user_indices = np.where(test_y == user)[0]
            
            # Create representative fingerprint (mean of all test windows)
            # This approximates creating one large fingerprint from all test messages
            user_vectors = test_X[user_indices]
            avg_vector = np.mean(user_vectors, axis=0).reshape(1, -1)
            
            # Calculate similarity for this single representative vector
            if method == 'cosine':
                avg_scaled = self.scaler.transform(avg_vector)
                avg_sim = cosine_similarity(avg_scaled, self.train_fingerprints)[0]
            elif method == 'siamese' and self.siamese_model:
                with torch.no_grad():
                    avg_tensor = torch.tensor(avg_vector, dtype=torch.float32).to(device)
                    avg_emb = self.siamese_model(avg_tensor)
                    # Reuse train_emb from above
                    avg_sim = torch.mm(avg_emb, train_emb.T).cpu().numpy()[0]
            
            # Check top-k for this user
            top_k_indices = np.argsort(-avg_sim)[:5]
            top_k_labels = self.train_labels[top_k_indices]
            
            if top_k_labels[0] == user:
                user_top1 += 1
            if user in top_k_labels:
                user_top5 += 1

        return {
            'fingerprint_top_1_accuracy': fp_top1 / n_test,
            'fingerprint_top_5_accuracy': fp_top5 / n_test,
            'user_top_1_accuracy': user_top1 / len(unique_test_users),
            'user_top_5_accuracy': user_top5 / len(unique_test_users),
            # Map standard keys to User-Level metrics as this is the primary goal now
            'top_1_accuracy': user_top1 / len(unique_test_users),
            'top_5_accuracy': user_top5 / len(unique_test_users),
        }
    
    def save(self):
        """Save model"""
        model_data = {
            'config': self.config,
            'scaler': self.scaler,
            'siamese_model': self.siamese_model,
            'train_fingerprints': self.train_fingerprints,
            'train_labels': self.train_labels,
            'threshold': self.threshold,
        }
        
        Path(self.config.model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, model_path: str):
        """Load model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        system = cls(model_data['config'])
        system.scaler = model_data['scaler']
        system.siamese_model = model_data['siamese_model']
        system.train_fingerprints = model_data['train_fingerprints']
        system.train_labels = model_data['train_labels']
        system.threshold = model_data['threshold']
        
        return system
    
    def upload_to_pinecone(self, api_key: str, index_name: str = "discord-fingerprints-full3"):
        """Upload trained fingerprints to Pinecone for querying."""
        if self.train_fingerprints is None or self.train_labels is None:
            raise ValueError("No trained fingerprints to upload. Train the model first.")
        
        pc = Pinecone(api_key=api_key)
        
        # Create index if it doesn't exist (adjust dimensions based on fingerprint size)
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=self.train_fingerprints.shape[1],
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        index = pc.Index(index_name)
        
        # Prepare vectors for upsert: unique ID, vector, metadata
        vectors = []
        for i, (fp, label) in enumerate(zip(self.train_fingerprints, self.train_labels)):
            vectors.append({
                "id": f"{label}_{i}",  # Unique ID: user_id + index
                "values": fp.tolist(),
                "metadata": {"user_id": str(label)}
            })
        
        # Upsert in batches (Pinecone recommends batching for large uploads)
        batch_size = 100
        for j in range(0, len(vectors), batch_size):
            batch = vectors[j:j + batch_size]
            index.upsert(vectors=batch)
        
        logger.info(f"Uploaded {len(vectors)} fingerprints to Pinecone index '{index_name}'")
