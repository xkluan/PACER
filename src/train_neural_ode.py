#!/home/lxk/anaconda3/envs/ana/bin/python
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import TIVA_Dataset_V3

# Configuration
DATA_PATH = "/home/lxk/vitaldb/analysis/topic4_pk_neural_ode/data/processed/final_dataset_v3_physics.parquet"
OUTPUT_DIR = "/home/lxk/vitaldb/analysis/topic4_pk_neural_ode/outputs/step3_2_neural_ode_v3"
MODEL_PATH = os.path.join(OUTPUT_DIR, "neural_ode_v3_model.pth")
BATCH_SIZE = 64
SEQ_LEN = 60 # 5 minutes
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ODEFuncV3(nn.Module):
    def __init__(self):
        super(ODEFuncV3, self).__init__()
        
        # Inputs u: [Prop, Remi, Age, Weight, HR, MAP, T_Prop, T_Remi] -> 8 dims
        # State y: [Ce_prop, Ce_remi] -> 2 dims
        
        # Neural Residual
        self.net = nn.Sequential(
            nn.Linear(2 + 8, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )
        
        # Physics Parameters (Learnable)
        self.ke_prop = nn.Parameter(torch.tensor(0.2)) 
        self.ke_remi = nn.Parameter(torch.tensor(0.5))
        
        # Volume Parameters (Learnable) - To convert Dose (mg/min) to Concentration (mg/L)
        # Initializing to reasonable V1 values (e.g. Prop ~5-10L, Remi ~5-10L)
        self.vol_prop = nn.Parameter(torch.tensor(5.0))
        self.vol_remi = nn.Parameter(torch.tensor(5.0))
        
        self.u_seq = None 
        self.t_seq = None 
        
    def set_current_batch(self, u_seq, t_seq):
        self.u_seq = u_seq
        self.t_seq = t_seq
        
    def get_input_at_t(self, t):
        max_t = self.u_seq.shape[1] - 1.0
        t_idx = torch.clamp(t, 0, max_t)
        
        idx_floor = t_idx.floor().long()
        idx_ceil = t_idx.ceil().long()
        alpha = t_idx - idx_floor.float()
        
        u0 = self.u_seq[:, idx_floor, :]
        u1 = self.u_seq[:, idx_ceil, :]
        
        u_t = (1 - alpha) * u0 + alpha * u1
        return u_t

    def forward(self, t, y):
        u = self.get_input_at_t(t) # [Batch, 8]
        
        # Physics (only uses Prop/Remi from u, which are idx 0 and 1)
        # dy/dt = -k * y + (Rate / Volume)
        # Ensure Volume is positive to avoid division by zero or negative volume
        import torch.nn.functional as F
        v_prop = F.softplus(self.vol_prop) + 1.0 # Min volume 1.0 L
        v_remi = F.softplus(self.vol_remi) + 1.0
        
        d_prop = -self.ke_prop * y[:, 0:1] + u[:, 0:1] / v_prop
        d_remi = -self.ke_remi * y[:, 1:2] + u[:, 1:2] / v_remi
        d_phys = torch.cat([d_prop, d_remi], dim=1)
        
        # Neural Residual (uses all u including Time_Since_Stop)
        net_in = torch.cat([y, u], dim=1)
        d_neural = self.net(net_in)
        
        return d_phys + d_neural * 0.1

class NeuralODEModelV3(nn.Module):
    def __init__(self):
        super(NeuralODEModelV3, self).__init__()
        self.ode_func = ODEFuncV3()
        
        # Readout: Linear -> Sigmoid (Logistic Regression)
        # We remove the hidden layer to enforce monotonicity and interpretability.
        # BIS = Sigmoid( -w1*Cp - w2*Cr + b )
        self.readout = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        
        # Initialize Readout
        # Bias: Large positive to ensure BIS=1.0 (Awake) when Ce=0
        # Weights: Negative to ensure BIS drops as Ce increases
        with torch.no_grad():
            self.readout[0].bias.fill_(4.0) # Sigmoid(4.0) ~= 0.98
            self.readout[0].weight.data = torch.tensor([[-1.0, -0.2]]) # Initial guess
        
    def forward(self, y0, u_seq):
        batch_size, seq_len, _ = u_seq.shape
        t_span = torch.linspace(0, seq_len-1, seq_len).to(y0.device)
        
        self.ode_func.set_current_batch(u_seq, t_span)
        out = odeint(self.ode_func, y0, t_span, method='rk4', options={'step_size': 1.0})
        
        out = out.permute(1, 0, 2) # [Batch, Seq, 2]
        
        out_flat = out.reshape(-1, 2)
        bis_pred = self.readout(out_flat)
        bis_pred = bis_pred.reshape(batch_size, seq_len)
        
        return bis_pred

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Data not found: {DATA_PATH}. Please run step3_2_preprocess.py first.")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading V3 Datasets...")
    train_ds = TIVA_Dataset_V3(DATA_PATH, seq_len=SEQ_LEN, mode='train')
    test_ds = TIVA_Dataset_V3(DATA_PATH, seq_len=SEQ_LEN, mode='test', scaler_stats=train_ds.stats)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    
    model = NeuralODEModelV3().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    print(f"Start Training V3 on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for x, y, priors in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Init state y0 using Physics Priors (Warm Start)
            # priors: [Batch, 2] -> [Ce_Prop, Ce_Remi]
            y0 = priors.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            # y is [Batch, Seq] -> Target BIS scaled 0-1
            pred = model(y0, x) # [Batch, Seq]
            pred = pred.squeeze(-1)
            
            loss = criterion(pred, y)
            
            # Loss NaN check
            if torch.isnan(loss):
                print("Loss is NaN!")
                continue
                
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Avg Loss: {train_loss/len(train_loader):.4f}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), MODEL_PATH)
        
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
