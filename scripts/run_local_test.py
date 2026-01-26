#!/usr/bin/env python3
"""Local test script for Credit Risk & Lending System API.

This script:
1. Starts the FastAPI application using uvicorn (development mode).
2. Sends a sample POST request to the `/api/v1/predict` endpoint using `httpx`.
3. Prints the response JSON.

Make sure the model artifacts are present under `models/Project 1: Credit Default Prediction/` before running.
"""

import subprocess
import time
import sys
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Helper to start the FastAPI server in a subprocess
# ---------------------------------------------------------------------------
def start_server():
    # Assuming the entry point is src/app/main.py
    cmd = ["uvicorn", "src.app.main:app", "--port", "8001"]
    # Start the server in a new process group so we can terminate it later
    proc = subprocess.Popen(
        cmd,
        cwd=Path(__file__).parents[2],  # project root (two levels up from scripts)
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=lambda: None,
    )
    return proc

# ---------------------------------------------------------------------------
# Wait for the server to become ready
# ---------------------------------------------------------------------------
def wait_for_server(url: str, timeout: int = 30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(url)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

# ---------------------------------------------------------------------------
# Main execution flow
# ---------------------------------------------------------------------------
def main():
    print("Starting FastAPI server...")
    server = start_server()
    try:
        # Give the server a moment to start
        if not wait_for_server("http://127.0.0.1:8001/health"):
            print("Server did not become ready in time.")
            sys.exit(1)

        # Prepare a minimal payload matching the PredictionRequest schema
        sample_payload = {
            "features": {
                # TODO: Populate with actual feature names expected by the model.
                # Example placeholder values:
                "feature_1": 0.0,
                "feature_2": 1.0,
                # Add all required features here.
            }
        }

        print("Sending test request to /api/v1/predict ...")
        response = httpx.post("http://127.0.0.1:8001/api/v1/predict", json=sample_payload)
        print("Response status:", response.status_code)
        print("Response body:")
        print(response.json())
    finally:
        print("Shutting down server...")
        server.terminate()
        try:
             outs, errs = server.communicate(timeout=5)
             print("Server Output:")
             print(outs)
        except Exception as e:
            print(f"Error capturing output: {e}")
        server.wait()

if __name__ == "__main__":
    main()
