# prior_train.py

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
import argparse
from tqdm import tqdm

from signal_gen_1d import generate_spde_true_signals_by_classes
from score_model import ScoreNet, dsm_loss_fn

# ==========================================
# 0. Device & Utils
# ==========================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

# ==========================================
# 1. Dataset
# ==========================================
class MaternSignalDataset(Dataset):
    def __init__(self, config, verbose=True):
        """
        Generates SPDE samples on initialization.
        """
        self.config = config
        data_cfg = config['data']
        
        if verbose:
            print(f"Generating {data_cfg['n_epoch_samples']} SPDE signals...")
            
        # Call the external generator
        out = generate_spde_true_signals_by_classes(
            n_total=data_cfg['n_epoch_samples'],
            n_points=data_cfg['n_points'],
            domain=tuple(data_cfg['domain']),
            classes=data_cfg['classes'],
            save_previews=False
        )
        
        self.data = out["y_true"].astype(np.float32)
        
        if self.data.shape[-1] != data_cfg['n_points']:
            print(f"Warning: Generated data width {self.data.shape[-1]} != config {data_cfg['n_points']}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns (1, L) tensor
        x = self.data[idx]
        return torch.from_numpy(x).unsqueeze(0)

# ==========================================
# 2. Main Train Loop
# ==========================================
def train(config):
    device = get_device()
    print(f"Training on {device}...")
    
    cfg_train = config['training']
    cfg_model = config['model']
    cfg_diffusion = config['diffusion'] 
    
    os.makedirs(cfg_train['save_dir'], exist_ok=True)
    
    with open(os.path.join(cfg_train['save_dir'], 'run_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    print("Initializing Matern Dataset...")
    dataset = MaternSignalDataset(config, verbose=True)
    dataloader = DataLoader(dataset, batch_size=cfg_train['batch_size'], shuffle=True)

    model = ScoreNet(channels=cfg_model['channels'], 
                     num_layers=cfg_model['layers'], 
                     L=cfg_model['L']).to(device)
                     
    optimizer = optim.Adam(model.parameters(), lr=cfg_train['lr'])
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Sigma Schedule: [{cfg_diffusion['sigma_min']}, {cfg_diffusion['sigma_max']}]")
    print("Starting training...")
    
    model.train()
    for epoch in range(cfg_train['epochs']):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Ep {epoch+1}/{cfg_train['epochs']}", leave=False)
        
        for x_batch in pbar:
            x_batch = x_batch.to(device)
            
            optimizer.zero_grad()
            
            # Pass sigma config to loss function
            loss = dsm_loss_fn(
                model, 
                x_batch, 
                sigma_min=cfg_diffusion['sigma_min'],
                sigma_max=cfg_diffusion['sigma_max']
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:04d} | Avg Loss: {avg_loss:.6f}")
            # Checkpoint
            torch.save(model.state_dict(), os.path.join(cfg_train['save_dir'], "score_model_latest.pth"))

    # Final save
    torch.save(model.state_dict(), os.path.join(cfg_train['save_dir'], "score_model_final.pth"))
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="prior_config.json", help="Path to config file")
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        config = load_config(args.config)
        train(config)
    else:
        print(f"Config file {args.config} not found. Please create it first.")