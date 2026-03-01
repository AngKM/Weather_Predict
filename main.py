import os
import torch
import warnings

# Suppress annoying standard scaler warnings if any
warnings.filterwarnings('ignore')

from src.data_loader import prepare_dataloaders
from src.model import GeomagneticLSTM
from src.train import train_model, test_model

def main():
    print("=== Geomagnetic Data Predictive Model ===")
    print("Task: Predict 6-hour ahead DST index using 24 hours of history")
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
        
    print("\n[1/4] Preparing Data Loaders...")
    batch_size = 256
    window_size = 24
    forecast_horizon = 6
    
    train_loader, val_loader, test_loader, scaler = prepare_dataloaders(
        data_dir, 
        batch_size=batch_size, 
        window_size=window_size, 
        forecast_horizon=forecast_horizon
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[2/4] Initializing Model on {device}...")
    
    # Features: [dst, hp, ap] => input_size=3
    model = GeomagneticLSTM(input_size=3, hidden_size=64, num_layers=2, output_size=1)
    
    print("\n[3/4] Training Model...")
    # Training
    model_path = os.path.join(os.path.dirname(__file__), 'geomap_lstm.pth')
    train_model(
        model, train_loader, val_loader, scaler, 
        num_epochs=15, 
        learning_rate=0.005, 
        device=device,
        model_path=model_path
    )
    
    print("\n[4/4] Testing Model on 2025 Data...")
    # Testing
    test_model(
        model, test_loader, scaler, 
        device=device, 
        model_path=model_path
    )
    
    print("\nDone! Visualizations saved to the current directory.")

if __name__ == '__main__':
    main()