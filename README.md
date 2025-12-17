# Discord User Fingerprinting

Minimal system for identifying Discord users by writing style and temporal patterns.

Core pieces:
- `fingerprint_core.py` – feature extraction and matching logic
- `train.py` – training script
- `edge_minimal_client.py` – client for querying a Pinecone index

## Installation

```bash
pip install pandas numpy scipy sentence-transformers torch scikit-learn pinecone-client
```

## Data Format

Input is JSONL with one message per line:

```json
{
  "ids": "user_id",
  "content": "message text",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

Point `FingerprintConfig.data_path` and `cache_path` in `fingerprint_core.py` or `train.py` to your file.

## Training

Default cosine-similarity model:

```bash
python train.py
```

This will:
- load messages
- build fingerprints per user
- train a cosine baseline
- evaluate on a held-out test split
- save the model to `models/model.pkl`

To train a simple Siamese variant:

```bash
python train.py siamese
```

## Using the Model Directly

```python
from fingerprint_core import FingerprintingSystem

system = FingerprintingSystem.load("models/model.pkl")
```

You can then use `system` to embed users and compute similarities in your own code.

## Minimal Pinecone Client

`edge_minimal_client.py` builds a single embedding from a list of messages and queries a Pinecone index.

Environment variable:

```bash
set PINECONE_API_KEY=your_pinecone_key
```

Example:

```bash
python edge_minimal_client.py
```

Edit the hardcoded `messages` list and index name in `edge_minimal_client.py` to match your setup.

## Adaptive Message Grouping

The system now employs a personalized, adaptive grouping logic instead of static time windows. This ensures fingerprints represent coherent conversational contexts.

### How it Works

1.  **Local Analysis (Per Batch)**
    *   When processing a batch of messages, the system calculates time deltas between consecutive messages.
    *   **Jump Detection:** It identifies natural breaks in conversation by looking for significant jumps in time gaps (e.g., typing speed vs. taking a break).
    *   *Example:* If gaps are 2s, 5s, 45s, then 600s, the system detects the jump and suggests a local threshold (e.g., 300s).

2.  **Global Context (Moving Average)**
    *   To avoid reacting too strongly to outliers (e.g., heated arguments), the system maintains a **Historical Threshold** for each user.
    *   **Smoothing:** It uses an Exponential Moving Average (EMA) to blend the new local threshold with the historical one.
    *   *Benefit:* The system adapts slowly to changes in user behavior rather than shattering history based on a single anomalous session.

3.  **Grouping Action**
    *   Using the calculated **Adaptive Threshold**, messages are grouped linearly.
    *   If the gap between Message A and Message B is less than the threshold, they stay in the same group.
    *   If the gap exceeds the threshold, the current group is closed, and a new one begins.

4.  **Resulting Fingerprints**
    *   **Coherent Context:** This prevents mixing distinct contexts (e.g., "Morning Work Chat" vs. "Evening Gaming Session") into a single fingerprint.
    *   **Distinct Modes:** The system accurately captures different "modes" of a user's writing style as separate, clean fingerprints.

## License

MIT
