# Discord User Fingerprinting Service

This project implements a system for identifying Discord users based on their writing style and temporal patterns. It exposes a FastAPI service that can be called periodically (e.g., every 12 hours) to incrementally train the model with new data and flag potential ban evaders.

## System Architecture

The system consists of the following core components:

*   **`api.py`**: The main entry point. A FastAPI service that handles file uploads, triggers incremental training, and performs inference against a list of banned users.
*   **`fingerprint_core_uploadTo_pinecone.py`**: Contains the core logic for feature extraction (linguistic & temporal), model definition, and Pinecone vector database interactions. It handles the logic for averaging fingerprints to ensure a maximum of 40 fingerprints per user.
*   **`train_uploadTo_pinecone.py`**: Handles the training loop. It supports both initial training and "incremental" mode, where it loads an existing model, processes new data, and uploads it to Pinecone.
*   **`edge_client_full.py`**: Contains the inference logic used by the API to query Pinecone for users similar to a specific banned user.

## Setup

### 1. Dependencies

Ensure you have the required Python packages installed. You can use the provided `environment.yml` or install manually:

```bash
pip install fastapi uvicorn python-dotenv pandas numpy scipy sentence-transformers torch scikit-learn pinecone-client python-multipart
```

### 2. Environment Configuration

Create a [`.env`](.env ) file in the root directory of the project to store your secrets. This file is git-ignored to prevent accidental commits.

**[`.env`](.env ) content:**

```env
PINECONE_API_KEY=your_actual_pinecone_api_key_here
```

### 3. Pinecone Index

The system expects a Pinecone index (default name: `discord-fingerprints-full3`). The code will attempt to create it if it doesn't exist, but ensure your API key has the necessary permissions.

## Running the Service

You can start the API service using the provided helper script:

```bash
./start_api.sh
```

Or manually via `uvicorn`:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The service will be available at `http://localhost:8000`. You can access the interactive API documentation at `http://localhost:8000/docs`.

## API Usage

### Endpoint: `POST /process`

This is the main endpoint used for the periodic update cycle.

**Inputs:**
*   `file`: A JSON file containing the new batch of Discord messages.
*   `banned_users_json`: A JSON string representing a list of user IDs that are currently banned.

**Workflow:**
1.  **Incremental Update**: The system reads the uploaded JSON file.
    *   It generates fingerprints for the users in the file.
    *   It uploads these fingerprints to Pinecone.
    *   **Consolidation**: If a user exceeds 40 fingerprints, the system takes the oldest excess fingerprints, averages them into a single "historical" fingerprint, and deletes the individual old ones.
2.  **Inference**: The system iterates through the provided `banned_users_json` list.
    *   For each banned user, it retrieves their centroid (average fingerprint) from Pinecone.
    *   It queries the index to find other users who have a high similarity to this banned user's centroid.

**Example Request (`curl`):**

```bash
curl -X 'POST' \
  'http://localhost:8000/process' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/new_messages.json;type=application/json' \
  -F 'banned_users_json=["1234567890", "0987654321"]'
```

**Example Response:**

```json
{
  "status": "success",
  "flagged_users": {
    "1234567890": [
      {
        "user_id": "9998887776",
        "score": 0.85
      },
      {
        "user_id": "5554443332",
        "score": 0.72
      }
    ],
    "0987654321": []
  }
}
```

## Automation (12-Hour Cycle)

To automate this process, set up a cron job or a scheduled task on your server that runs every 12 hours. This task should:
1.  Gather the new messages from the last 12 hours into a JSON file.
2.  Fetch the current list of banned users.
3.  Make the `curl` request shown above to the running FastAPI service.
