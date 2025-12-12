#!/bin/bash
# Start the FastAPI service
# Ensure PINECONE_API_KEY is set in the environment before running this
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
