import os
import subprocess
import sys
from src.data_processor import TrafficDataProcessor
from src.forecaster import ImpactForecaster

def train_models():
    data_path = 'Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return False
        
    print("--- Phase 1: Initializing Data Pipeline ---")
    processor = TrafficDataProcessor(data_path)
    df = processor.process()
    
    print("\n--- Phase 2: Training ML Forecaster ---")
    forecaster = ImpactForecaster()
    forecaster.train(df)
    
    print("\nTraining Complete! Models are ready.")
    return True

if __name__ == "__main__":
    # Check if models exist
    if not os.path.exists('models/forecaster_model.pkl'):
        success = train_models()
        if not success:
            sys.exit(1)
    else:
        print("Models already exist. Skipping training.")
        
    print("\n--- Phase 3: Launching Command Center ---")
    print("Please view the application in your browser.")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app.py"])
    except KeyboardInterrupt:
        print("Shutting down the Command Center.")
