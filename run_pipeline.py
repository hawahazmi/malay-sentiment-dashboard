"""
This Python script automates the entire system workflow for the dashboard:
1. Data Collection (YouTube Scraper)
2. Data Preprocessing (Cleaning)
4. Model Deployment (Inference)
-------------------------------------------------------------
"""

import subprocess
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def run_command(command, phase_name):
    print(f"\n============================")
    print(f"Running {phase_name} ...")
    print(f"============================")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise SystemExit(f"{phase_name} failed! Check logs above.")
    print(f"{phase_name} completed successfully.\n")


if __name__ == "__main__":
    try:
        # Activate virtual environment (PyCharm handles venv automatically if selected)
        print("============================")
        print("Using virtual environment")
        print("============================")

        # Step 1: Data Collection
        run_command("python -m src.data_collection.youtube_scraper", "Phase 1: Data Collection")

        # Step 2: Data Preprocessing
        run_command("python -m src.preprocessing.yt_preprocess", "Phase 2: Data Preprocessing")

        # Step 4: Model Deployment (auto sentiment labeling)
        run_command("python -m src.model_dev.infer_bilstm", "Phase 3: Model Deployment")

        # Step 5: Prescriptive Analytics
        run_command("python -m src.prescriptive.prescriptive", "Phase 3: Prescriptive Analytics")

        print("All phases completed successfully!")

    except Exception as e:
        print(f"Pipeline failed: {e}")
