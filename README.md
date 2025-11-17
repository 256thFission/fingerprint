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

## License

MIT
