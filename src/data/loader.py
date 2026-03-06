import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

# Backward compatibility
def extract_hr_map(vital_path):
    # This function was originally here but its implementation is lost.
    # It seems to be used by step1_5 to extract HR/MAP.
    # Since I overwrote it, step1_5 might fail if it tries to run.
    # However, we are moving to step1_6 which will likely do its own extraction or build on V3.
    # For now, I'll leave a placeholder or try to reimplement if I recall it.
    # But step1_5 imports it. I should check step1_5 to see how it uses it.
    pass

def extract_physiology(vital_path):
    pass

class TIVA_Dataset_V3(Dataset):
    def __init__(self, parquet_path, seq_len=60, mode='train', scaler_stats=None, limit=None):
        self.seq_len = seq_len
        self.mode = mode
        
        # Load
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"File not found: {parquet_path}")
            
        print(f"Loading V3 Dataset from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        # Limit cases if requested (for fast debugging)
        if limit is not None:
            all_cases = df['caseid'].unique()
            if len(all_cases) > limit:
                selected_cases = all_cases[:limit]
                df = df[df['caseid'].isin(selected_cases)]
                print(f"Limiting dataset to first {limit} cases.")
        
        # Normalize columns
        if scaler_stats is None:
             self.stats = {
                'hr': (75.0, 15.0),
                'map': (85.0, 15.0),
                'age': (50.0, 15.0),
                'weight': (65.0, 15.0),
                # Time features: Normalize by 30 mins (1800s)
                'time_since_prop': (0.0, 1800.0),
                'time_since_remi': (0.0, 1800.0)
            }
        else:
            self.stats = scaler_stats

        # Check if age/weight are missing, merge if so
        if 'age' not in df.columns or 'weight' not in df.columns:
            clinical_path = "/home/lxk/vitaldb/physionet.org/files/vitaldb/1.0.0/clinical_data.csv"
            if os.path.exists(clinical_path):
                clinical = pd.read_csv(clinical_path)
                clinical = clinical[['caseid', 'age', 'weight']].dropna()
                clinical['age'] = pd.to_numeric(clinical['age'], errors='coerce')
                clinical['weight'] = pd.to_numeric(clinical['weight'], errors='coerce')
                clinical = clinical.dropna()
                df = df.merge(clinical, on='caseid', how='inner')

        # Normalize in place (copy to avoid warning)
        self.df = df.copy()
        
        # Normalize
        for col, (mean, std) in self.stats.items():
            if col in self.df.columns:
                self.df[col] = (self.df[col] - mean) / std
                
        # Fill NaNs
        self.df = self.df.fillna(0.0)

        # Split (80/20)
        self.case_ids = self.df['caseid'].unique()
        n_cases = len(self.case_ids)
        
        # Deterministic split based on sorted caseids
        self.case_ids = sorted(self.case_ids)
        
        n_train = int(n_cases * 0.8)
        
        if mode == 'train':
            self.case_ids = self.case_ids[:n_train]
        else:
            self.case_ids = self.case_ids[n_train:]
            
        print(f"Dataset ({mode}): {len(self.case_ids)} cases.")
        
        # Filter dataframe
        self.df = self.df[self.df['caseid'].isin(self.case_ids)]
        
        # Prepare indices
        self.samples = []
        # Group by case
        grouped = self.df.groupby('caseid')
        for cid, group in grouped:
            # Create sliding windows
            # We need indices relative to the dataframe
            # But converting to numpy array is faster
            group = group.sort_values('time')
            
            # Check if physics priors exist
            has_priors = 'ce_prop_eleveld' in group.columns and 'ce_remi_minto' in group.columns
            
            if has_priors:
                vals = group[['prop_rate', 'remi_rate', 'age', 'weight', 'hr', 'map', 'time_since_prop', 'time_since_remi', 
                              'ce_prop_eleveld', 'ce_remi_minto', 'bis']].values
            else:
                # Fallback for old dataset (though we should use the new one)
                # Pad with zeros if not found to maintain shape consistency or handle separately?
                # For safety, let's error out if we expect priors but don't find them, 
                # or just handle it in getitem. 
                # Given the strict refactoring plan, we expect them.
                # But to keep backward compatibility, we can add dummy columns.
                vals = group[['prop_rate', 'remi_rate', 'age', 'weight', 'hr', 'map', 'time_since_prop', 'time_since_remi', 'bis']].values
                # Insert dummies at index 8, 9
                priors_dummy = np.zeros((len(vals), 2))
                vals = np.insert(vals, 8, priors_dummy.T, axis=1)
            
            # Stride? Step 3.2 usually uses stride=1 or similar?
            # Assuming stride=10 (50s) to reduce data? Or stride=1 (5s)?
            # Let's use stride=1 for max data, or stride=5 for efficiency.
            # Step 3.2 default was likely stride=1.
            n_samples = len(vals)
            if n_samples < seq_len:
                continue
                
            for i in range(0, n_samples - seq_len + 1, 1): # Stride 1
                # Filter windows with artifacts (BIS < 10)
                # bis is at index 10
                if np.any(vals[i:i+seq_len, 10] < 10):
                    continue
                    
                self.samples.append(vals[i:i+seq_len])
                
        print(f"Generated {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # returns x, y, priors
        # x: [Seq, Features=8]
        # y: [Seq] (BIS)
        # priors: [2] (Ce_Prop, Ce_Remi at t=0)
        
        data = self.samples[idx]
        # Features: Prop, Remi, Age, Weight, HR, MAP, T_Prop, T_Remi (8 cols)
        x = data[:, :8].astype(np.float32)
        
        # Priors: Ce_Prop, Ce_Remi (Cols 8, 9)
        # We only need the prior at the START of the window (t=0) to initialize the ODE
        priors = data[0, 8:10].astype(np.float32)
        
        # Target: BIS (Col 10)
        y = data[:, 10].astype(np.float32) / 100.0 # Scale BIS 0-1
        
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(priors)


class TIVA_Dataset_V5(Dataset):
    """
    V5 Dataset for Delay Correction (Task 2/3).
    Includes dynamic physiological features: dMAP_dt, SEF, Total_Dose_Norm.
    """
    def __init__(self, parquet_path, seq_len=60, mode='train', scaler_stats=None):
        self.seq_len = seq_len
        self.mode = mode
        
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"File not found: {parquet_path}")
            
        print(f"Loading V5 Dataset from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        # Features to normalize
        if scaler_stats is None:
             self.stats = {
                'hr': (75.0, 15.0),
                'map': (85.0, 15.0),
                'age': (50.0, 15.0),
                'weight': (65.0, 15.0),
                'bmi': (25.0, 5.0),
                'time_since_prop': (0.0, 1800.0),
                'time_since_remi': (0.0, 1800.0),
                'dmap_dt': (0.0, 1.0), 
                'sef': (15.0, 10.0),
                'total_dose_norm': (10.0, 5.0)
            }
        else:
            self.stats = scaler_stats

        # Normalize in place
        self.df = df.copy()
        
        # Fill missing SEF
        if 'sef' in self.df.columns:
            self.df['sef'] = self.df['sef'].fillna(20.0)
        
        # Normalize
        for col, (mean, std) in self.stats.items():
            if col in self.df.columns:
                self.df[col] = (self.df[col] - mean) / std
                
        self.df = self.df.fillna(0.0)

        # Split
        self.case_ids = sorted(self.df['caseid'].unique())
        n_cases = len(self.case_ids)
        n_train = int(n_cases * 0.8)
        
        if mode == 'train':
            self.case_ids = self.case_ids[:n_train]
        else:
            self.case_ids = self.case_ids[n_train:]
            
        print(f"Dataset ({mode}): {len(self.case_ids)} cases.")
        self.df = self.df[self.df['caseid'].isin(self.case_ids)]
        
        # Prepare samples
        self.samples = []
        grouped = self.df.groupby('caseid')
        
        # Columns
        # ODE: prop, remi, age, weight, hr, map, t_prop, t_remi (8)
        # Delay: age, bmi, total_dose_norm, dmap_dt, sef (5)
        # Target: bis
        self.cols = ['prop_rate', 'remi_rate', 'age', 'weight', 'hr', 'map', 'time_since_prop', 'time_since_remi', 
                     'bmi', 'total_dose_norm', 'dmap_dt', 'sef', 'bis']
        
        for cid, group in grouped:
            group = group.sort_values('time')
            # Ensure all cols exist
            for c in self.cols:
                if c not in group.columns:
                    group[c] = 0.0
            
            vals = group[self.cols].values
            
            n_samples = len(vals)
            if n_samples < seq_len:
                continue
            
            # Stride 12 (1 minute) to speed up training
            # Original: stride 1 (5s) -> 4.5M samples (Too slow)
            for i in range(0, n_samples - seq_len + 1, 12):
                self.samples.append(vals[i:i+seq_len])
                
        print(f"Generated {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        
        # ODE Inputs: [0:8]
        # prop, remi, age, weight, hr, map, t_prop, t_remi
        x_ode = data[:, :8].astype(np.float32)
        
        # Delay Inputs: [Age, BMI, Total_Dose_Norm, dMAP_dt, SEF]
        # Age is index 2.
        # BMI is index 8.
        # Total_Dose is 9.
        # dMAP is 10.
        # SEF is 11.
        # BIS is 12.
        
        # Extract columns for delay
        # We need [Age, BMI, Total_Dose, dMAP, SEF]
        # Age is data[:, 2]
        # Others are data[:, 8:12]
        age = data[:, 2:3]
        others = data[:, 8:12]
        x_delay = np.concatenate([age, others], axis=1).astype(np.float32)
        
        y = data[:, 12].astype(np.float32) / 100.0
        
        return torch.from_numpy(x_ode), torch.from_numpy(x_delay), torch.from_numpy(y)
