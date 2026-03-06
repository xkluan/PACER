# PACER: Physics-Informed AI for Clinical Emergence Recovery

PACER is a hybrid framework that combines mechanistic Pharmacokinetic (PK) models with Neural Ordinary Differential Equations (Neural ODE) to predict patient-specific consciousness recovery (Bispectral Index, BIS) during Total Intravenous Anaesthesia (TIVA).

## 📂 Repository Structure

```
PACER/
├── data/                  # Data documentation and sample files
├── models/                # Pre-trained model weights
│   ├── neural_ode_v3_model.pth      # Stage 1: Physics-Informed Neural ODE
│   └── transformer_corrector.pth    # Stage 2: Residual Corrector
├── src/                   # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── utils/             # Metrics and helper functions
│   ├── pk_models.py       # Base PK/PD compartment models
│   ├── train_neural_ode.py   # Training script for Neural ODE
│   └── train_transformer.py  # Training script for Transformer
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- PyTorch 1.12+
- Torchdiffeq
- Pandas, NumPy, Scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
This project uses the **VitalDB** dataset (https://vitaldb.net).
Due to data usage agreements, we cannot publish the raw dataset.

**Instructions:**
1. Download the VitalDB dataset (cases with TIVA and BIS monitoring).
2. Preprocess the data into parquet format (see `src/data/loader.py` for expected schema).
3. Place `clinical_data.csv` (demographics) in the `data/` directory.

### 3. Running the Models

#### Training Neural ODE (Stage 1)
To retrain the Physics-Informed Neural ODE component:
```bash
python src/train_neural_ode.py
```

#### Training Residual Corrector (Stage 2)
To retrain the Transformer-based residual corrector:
```bash
python src/train_transformer.py
```

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation
If you use this code in your research, please cite:
> Gao G, Luan X. "PACER: Physics-Informed AI for Clinical Emergence Recovery." Under Review, 2024.
