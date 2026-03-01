import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np

def unscale_values(scaled_values, scaler, dst_feature_index=0):
    """
    Reverse the StandardScaler transformation for a 1D array of values.
    Since we only predicted the DST (which is the first feature), 
    we use the mean and scale exactly from that index.
    """
    mean = scaler.mean_[dst_feature_index]
    scale = scaler.scale_[dst_feature_index]
    return (scaled_values * scale) + mean

def train_model(model, train_loader, val_loader, scaler, num_epochs=15, learning_rate=0.001, device='cpu', 
                dst_idx=0, model_path='geomap_lstm.pth'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    train_losses = []
    val_losses = []
    
    print(f"Training on {device}...")
    
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        
        # tqdm for progress
        pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]')
        for sequences, targets in pbar:
            sequences = sequences.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * sequences.size(0)
            pbar.set_postfix({'Loss': loss.item()})
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device).unsqueeze(1)
                
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * sequences.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            print("  --> Saving best model...")
            torch.save(model.state_dict(), model_path)
            
    print("Training complete.")
    
    # Plot losses
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    print("Saved loss curve to loss_curve.png")

def test_model(model, test_loader, scaler, device='cpu', model_path='geomap_lstm.pth'):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    print("Testing model on 2025 data...")
    with torch.no_grad():
        for sequences, targets in tqdm(test_loader, desc="Testing"):
            sequences = sequences.to(device)
            outputs = model(sequences)
            
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy())

    preds = np.array(all_preds)
    actuals = np.array(all_targets)
    
    # Inverse transform
    # DST is feature index 0 exactly
    dst_idx = 0
    preds_unscaled = unscale_values(preds, scaler, dst_idx)
    actuals_unscaled = unscale_values(actuals, scaler, dst_idx)
    
    mse = np.mean((preds_unscaled - actuals_unscaled)**2)
    mae = np.mean(np.abs(preds_unscaled - actuals_unscaled))
    print(f"\nTest Metrics -> MSE: {mse:.4f} | MAE: {mae:.4f}")
    
    # Visualizing first 500 predictions
    plot_len = min(500, len(preds_unscaled))
    plt.figure(figsize=(15, 5))
    plt.plot(actuals_unscaled[:plot_len], label='Actual DST', alpha=0.7)
    plt.plot(preds_unscaled[:plot_len], label='Predicted DST', alpha=0.7)
    plt.title('6-Hour Ahead Prediction: Actual vs Predicted DST (First 500 Test Samples)')
    plt.xlabel('Hours')
    plt.ylabel('DST Index (nT)')
    plt.legend()
    plt.grid()
    plt.savefig('test_predictions.png')
    print("Saved test predictions plot to test_predictions.png")
