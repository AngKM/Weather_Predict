import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

def parse_dst_file(filepath):
    """
    Parses the standard WDC format DST index file and returns a pandas DataFrame.
    Format:
    Chars 1-3: DST
    Chars 4-5: Year (2 digits)
    Chars 6-7: Month
    Chars 8: * or space
    Chars 9-10: Day
    Chars 11-12: blank
    Chars 13-14: X? or space
    Chars 15-16: Century (e.g., 20 for 2000s)
    Chars 17-20: Base value (often '   0')
    Chars 21-24: Value for Hour 1 (00:00 to 01:00 UT)
    ...
    Chars 113-116: Value for Hour 24 (23:00 to 24:00 UT)
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            if len(line.strip()) < 116:
                continue
            
            # The year logic: "00" with century "20" -> 2000
            yy = int(line[3:5])
            century_str = line[14:16].strip()
            century = int(century_str) if century_str else 20
            year = century * 100 + yy
            if year < 1000: # fallback
                 year = 2000 + yy if yy < 50 else 1900 + yy

            month = int(line[5:7])
            day = int(line[8:10])
            
            base_val_str = line[16:20].strip()
            base_val = int(base_val_str) if base_val_str else 0
            
            # Parse 24 hourly values
            for h in range(24):
                val_str = line[20 + h*4 : 24 + h*4].strip()
                if not val_str:
                    continue
                val = int(val_str)
                if val == 9999: # Missing data value
                    val = np.nan
                else:
                    val += base_val
                    
                records.append({
                    'datetime': pd.Timestamp(year=year, month=month, day=day, hour=h),
                    'dst': val
                })
                
    return pd.DataFrame(records)

def parse_hpo_file(filepath):
    """
    Parses the custom HPO data format file.
    Assumes columns: year, month, day, hour_start, hour_end, mjd1, mjd2, hp, ap, extra
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 9:
                continue
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            hour = int(float(parts[3]))
            hp = float(parts[7])
            ap = float(parts[8])
            
            records.append({
                 'datetime': pd.Timestamp(year=year, month=month, day=day, hour=hour),
                 'hp': hp,
                 'ap': ap
            })
            
    return pd.DataFrame(records)

def load_and_merge_data(dst_path, hpo_path):
    print("Parsing DST file...")
    dst_df = parse_dst_file(dst_path)
    print("Parsing HPO file...")
    hpo_df = parse_hpo_file(hpo_path)
    
    print("Merging data...")
    # Merge using datetime as the key
    merged_df = pd.merge(dst_df, hpo_df, on='datetime', how='inner')
    
    # Sort by time and drop missing values
    merged_df = merged_df.sort_values('datetime').reset_index(drop=True)
    merged_df = merged_df.dropna()
    
    return merged_df

class GeomagneticDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def create_sequences(data, feature_cols, target_col, window_size=24, forecast_horizon=6):
    """
    Creates overlapping sequences for time series prediction.
    window_size: 24 hours (features)
    forecast_horizon: 6 hours ahead (target is t + window_size + forecast_horizon - 1)
    """
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        x_seq = data.iloc[i : i + window_size][feature_cols].values
        # Target is strictly "6 hours ahead" from the LAST hour of the input window.
        # e.g., if window is hour 0..23, last hour is 23. 6 hours ahead is hour 29.
        # The index in data for hour 29 is i + 23 + 6 = i + 29
        y_val = data.iloc[i + window_size + forecast_horizon - 1][target_col]
        
        X.append(x_seq)
        y.append(y_val)
        
    return np.array(X), np.array(y)

def prepare_dataloaders(data_dir, batch_size=64, window_size=24, forecast_horizon=6):
    dst_path = os.path.join(data_dir, 'DST_2000-2025.txt')
    hpo_path = os.path.join(data_dir, 'hpodata_2000-2025.txt')
    
    df = load_and_merge_data(dst_path, hpo_path)
    print(f"Total merged records: {len(df)}")
    
    # Feature columns and target column
    feature_cols = ['dst', 'hp', 'ap']
    target_col = 'dst'
    
    # Time splits
    # Train: 2000 - 2020
    # Val: 2021 - 2024
    # Test: 2025
    train_df = df[df['datetime'].dt.year <= 2020].copy()
    val_df = df[(df['datetime'].dt.year >= 2021) & (df['datetime'].dt.year <= 2024)].copy()
    test_df = df[df['datetime'].dt.year >= 2025].copy()
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # Create sequences
    X_train, y_train = create_sequences(train_df, feature_cols, target_col, window_size, forecast_horizon)
    X_val, y_val = create_sequences(val_df, feature_cols, target_col, window_size, forecast_horizon)
    X_test, y_test = create_sequences(test_df, feature_cols, target_col, window_size, forecast_horizon)
    
    # Create PyTorch datasets and loaders
    train_dataset = GeomagneticDataset(X_train, y_train)
    val_dataset = GeomagneticDataset(X_val, y_val)
    test_dataset = GeomagneticDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler

if __name__ == '__main__':
    # Test loading
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    train_loader, val_loader, test_loader, scaler = prepare_dataloaders(data_dir)
    print(f"Train batches: {len(train_loader)}")
    for x, y in train_loader:
        print(f"X shape: {x.shape}, y shape: {y.shape}")
        break
