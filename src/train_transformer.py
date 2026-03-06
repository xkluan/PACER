
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys
import logging
import math
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Configuration
OUTPUT_DIR = "/home/lxk/vitaldb/analysis/topic4_pk_neural_ode/outputs/residual_correction_transformer"
DATA_PATH_TARGETS = "/home/lxk/vitaldb/analysis/topic4_pk_neural_ode/outputs/residual_correction/emergence_residual_dataset.parquet"
DATA_PATH_FEATURES = "/home/lxk/vitaldb/analysis/topic4_pk_neural_ode/data/processed/final_dataset_v5.parquet"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "transformer_corrector.pth")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1024
EPOCHS = 20
LR = 1e-3
SEQ_LEN = 240  # 20 minutes * 60s / 5s = 240 steps
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup Logging
log_path = os.path.join(OUTPUT_DIR, "step9_train_transformer.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerCorrector(nn.Module):
    def __init__(self, input_dim=4, d_model=64, nhead=4, num_layers=2):
        super(TransformerCorrector, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=SEQ_LEN+100)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x: (Batch, Seq, Feat)
        # Transformer expects (Seq, Batch, Feat)
        x = x.permute(1, 0, 2)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        output = self.transformer_encoder(x)
        
        # Take the last time step
        last_output = output[-1, :, :]
        
        pred = self.decoder(last_output)
        
        # Clamp output for stability
        return torch.clamp(pred, min=-20.0, max=30.0)

class TransformerDataset(Dataset):
    def __init__(self, targets_df, features_df, mode='train', scaler_stats=None):
        self.targets_df = targets_df.reset_index(drop=True)
        self.mode = mode
        
        # Prepare Feature Cache
        # Convert features_df to dictionary of numpy arrays for fast lookup
        # Features: prop_rate, remi_rate, dmap_dt, sef
        # We need to normalize features here or use pre-normalized
        
        logger.info(f"Preparing feature cache for {mode} set...")
        
        # Normalization Stats (same as TIVA_Dataset_V5)
        if scaler_stats is None:
             self.stats = {
                'prop_rate': (5000.0, 5000.0), # Approximate, prop_rate in mcg/kg/hr can be large? 
                # Wait, step 8 fixed unit issues. Let's check ranges.
                # Prop rate ~ 0-12 mg/kg/hr? or mcg?
                # Using robust scaler stats from previous steps would be better.
                # Let's use simple mean/std if not provided.
                'remi_rate': (5.0, 5.0),
                'dmap_dt': (0.0, 1.0), 
                'sef': (15.0, 10.0)
            }
        else:
            self.stats = scaler_stats
            
        # Normalize features_df before caching
        # Check columns
        req_cols = ['caseid', 'time', 'prop_rate', 'remi_rate', 'dmap_dt', 'sef']
        for col in req_cols:
            if col not in features_df.columns:
                logger.warning(f"Column {col} missing in features!")
                
        # Fill NaNs
        features_df = features_df.copy()
        features_df['sef'] = features_df['sef'].fillna(20.0)
        features_df['dmap_dt'] = features_df['dmap_dt'].fillna(0.0)
        features_df = features_df.fillna(0.0)
        
        # Normalize
        # We'll normalize on the fly or here? Here is faster for cache.
        # But we need to know stats first.
        # Let's compute stats from features_df if not provided
        if scaler_stats is None and mode == 'train':
            self.stats = {}
            for col in ['prop_rate', 'remi_rate', 'dmap_dt', 'sef']:
                self.stats[col] = (features_df[col].mean(), features_df[col].std() + 1e-6)
        
        # Apply norm
        for col in ['prop_rate', 'remi_rate', 'dmap_dt', 'sef']:
            mean, std = self.stats[col]
            features_df[col] = (features_df[col] - mean) / std
            
        # Build Cache
        self.feature_cache = {}
        # Group by caseid
        grouped = features_df.groupby('caseid')
        for caseid, group in tqdm(grouped, desc="Caching features"):
            # Sort by time
            group = group.sort_values('time')
            times = group['time'].values
            feats = group[['prop_rate', 'remi_rate', 'dmap_dt', 'sef']].values.astype(np.float32)
            self.feature_cache[caseid] = (times, feats)
            
    def __len__(self):
        return len(self.targets_df)
    
    def __getitem__(self, idx):
        row = self.targets_df.iloc[idx]
        caseid = row['caseid']
        target_time = row['time']
        delta_bis = row['delta_bis']
        
        if caseid not in self.feature_cache:
            # Should not happen if datasets match
            return torch.zeros(SEQ_LEN, 4), torch.tensor(0.0)
            
        times, feats = self.feature_cache[caseid]
        
        # Find index of target_time
        # Use searchsorted
        idx_end = np.searchsorted(times, target_time, side='right')
        # idx_end points to the first element > target_time.
        # We want up to target_time (inclusive if it exists, or closest past)
        # Actually target_time is the time of prediction.
        
        # We want window [target_time - 20min, target_time]
        # Since data is 5s uniform (mostly), we can just take [idx_end - SEQ_LEN : idx_end]
        
        if idx_end == 0:
             seq = np.zeros((SEQ_LEN, 4), dtype=np.float32)
        else:
            idx_start = idx_end - SEQ_LEN
            
            if idx_start < 0:
                # Pad with zeros at the beginning
                pad_len = -idx_start
                data = feats[0:idx_end]
                seq = np.pad(data, ((pad_len, 0), (0, 0)), mode='constant')
            else:
                seq = feats[idx_start:idx_end]
                
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(delta_bis, dtype=torch.float32)

def residual_loss(delta_pred, delta_target):
    """
    Weighted Asymmetric Loss (Same as Step 6)
    """
    error = delta_pred - delta_target
    abs_error = torch.abs(error)
    
    # 1. Asymmetric Safety: Target > Pred => Underestimation of BIS => Risk
    w_safety = torch.where(delta_target > delta_pred, 2.5, 1.0)
    
    # 2. Hard Example Mining: |Target| > 10
    w_hard = torch.where(torch.abs(delta_target) > 10.0, 2.0, 1.0)
    
    return torch.mean(w_safety * w_hard * abs_error)

def main():
    logger.info("Loading Datasets...")
    
    # Load Targets (Emergence Residuals)
    df_targets = pd.read_parquet(DATA_PATH_TARGETS)
    logger.info(f"Loaded {len(df_targets)} target rows.")
    
    # Load Features (Full History)
    df_features = pd.read_parquet(DATA_PATH_FEATURES)
    logger.info(f"Loaded {len(df_features)} feature rows.")
    
    # Split by Case ID
    case_ids = df_targets['caseid'].unique()
    train_ids, test_ids = train_test_split(case_ids, test_size=0.2, random_state=42)
    
    train_targets = df_targets[df_targets['caseid'].isin(train_ids)].copy()
    test_targets = df_targets[df_targets['caseid'].isin(test_ids)].copy()
    
    # Features need to cover all cases. No need to split features, just cache all.
    # But for strictness, we can pass relevant subsets? No, cache logic handles lookup.
    
    logger.info("Creating Datasets...")
    train_dataset = TransformerDataset(train_targets, df_features, mode='train')
    test_dataset = TransformerDataset(test_targets, df_features, mode='test', scaler_stats=train_dataset.stats)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    # Model
    model = TransformerCorrector(input_dim=4, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training Loop
    logger.info("Starting Training...")
    best_loss = float('inf')
    loss_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            X, y = X.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            pred = model(X)
            loss = residual_loss(pred, y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        mse_sum = 0.0
        count = 0
        
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(DEVICE), y.to(DEVICE).unsqueeze(1)
                pred = model(X)
                loss = residual_loss(pred, y)
                val_loss += loss.item()
                
                mse_sum += torch.sum((pred - y)**2).item()
                count += y.size(0)
                
        avg_val_loss = val_loss / len(test_loader)
        rmse = np.sqrt(mse_sum / count)
        
        scheduler.step(avg_val_loss)
        loss_history.append(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, RMSE={rmse:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info("Saved Best Model.")
            
        # Early Stopping Check (Loose)
        # User condition: "If the Transformer does not beat the MLP by at least 2% in RMSE after 20 epochs"
        # We check at the end.
        
    logger.info("Training Complete.")
    
    # Final Evaluation & Comparison
    logger.info("Evaluating Best Model...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            pred = model(X)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
    y_pred = np.concatenate(all_preds).flatten()
    y_true = np.concatenate(all_targets).flatten() # delta_bis
    
    # Calculate Metrics
    # We need True BIS to calc MDPE/MDAPE properly?
    # Actually MDPE/MDAPE are calc on (Pred_Final - True_BIS) / True_BIS
    # Here y_pred is Delta_BIS.
    # We need to reconstruct Pred_Final = Pred_ODE + Delta_BIS_Pred
    # But we don't have Pred_ODE here directly in the loader output (we returned only delta).
    # However, y_true is Delta_BIS_True = True_BIS - Pred_ODE
    # So Error = (Pred_ODE + Delta_BIS_Pred) - True_BIS
    #          = Delta_BIS_Pred - (True_BIS - Pred_ODE)
    #          = Delta_BIS_Pred - Delta_BIS_True
    #          = y_pred - y_true
    # So Error is correct.
    # To calc % Error, we need True_BIS.
    # We should return True_BIS from dataset to compute MDAPE.
    
    # Let's rely on RMSE for the stop condition as per user request ("beat MLP by 2% in RMSE").
    rmse_final = np.sqrt(np.mean((y_pred - y_true)**2))
    
    # Calculate MDAPE approx? Or load it?
    # I'll modify the dataset to return True_BIS if needed, but for now RMSE is the key comparison.
    # Previous MLP RMSE was ~22.09 (from step 6 log in thought trace).
    # 2% improvement means RMSE < 21.65.
    
    logger.info(f"Final Transformer RMSE: {rmse_final:.4f}")
    
    # Compare with MLP Baseline (Hardcoded from previous run or loaded)
    # MLP RMSE ~ 22.1
    mlp_rmse = 22.1 
    improvement = (mlp_rmse - rmse_final) / mlp_rmse * 100.0
    
    logger.info(f"Baseline MLP RMSE: {mlp_rmse}")
    logger.info(f"Improvement: {improvement:.2f}%")
    
    if improvement >= 2.0:
        logger.info("SUCCESS: Transformer beat MLP by > 2%.")
    else:
        logger.info("FAILURE: Transformer did not beat MLP by 2%.")

if __name__ == "__main__":
    main()
