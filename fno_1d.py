import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import operator
from functools import reduce
from timeit import default_timer

# ==========================================
# 1. Utils & Losses
# ==========================================

device = torch.device('cpu')

class L_inf_Loss(object):
    """Relative L-infinity error. Returns (Batch,) if reduction=False"""
    def __init__(self, reduction=True):
        self.reduction = reduction

    def __call__(self, x, y):
        # x, y: [Batch, Length]
        num_examples = x.size(0)
        diff_norms = torch.amax(torch.abs(x.reshape(num_examples, -1) - y.reshape(num_examples, -1)), dim=1)
        y_norms = torch.amax(torch.abs(y.reshape(num_examples, -1)), dim=1)
        res = diff_norms / (y_norms + 1e-6)
        return torch.mean(res) if self.reduction else res

class L2_Rel_Loss(object):
    """Relative L2 error. Returns (Batch,) if reduction=False"""
    def __init__(self, reduction=True):
        self.reduction = reduction

    def __call__(self, x, y):
        num_examples = x.size(0)
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), p=2, dim=1)
        y_norms = torch.norm(y.reshape(num_examples, -1), p=2, dim=1)
        res = diff_norms / (y_norms + 1e-6)
        return torch.mean(res) if self.reduction else res

class MSE_Loss_Func(object):
    """Standard MSE."""
    def __init__(self, reduction=True):
        self.reduction = reduction
        
    def __call__(self, x, y):
        return F.mse_loss(x, y, reduction='mean' if self.reduction else 'none')

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size() + (2,) if p.is_complex() else p.size()))
    return c

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# ==========================================
# 2. Model Architecture (MRA FNO 1D)
# ==========================================

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels, self.out_channels, self.modes1 = in_channels, out_channels, modes1
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
    
    def compl_mul1d(self, i, w): 
        return torch.einsum("bix,iox->box", i, w)
    
    def forward(self, x):
        b = x.shape[0]
        x_ft = torch.fft.rfft(x)
        o_ft = torch.zeros(b, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        o_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        return torch.fft.irfft(o_ft, n=x.size(-1))

class MLP(nn.Module):
    def __init__(self, i, o, m): 
        super(MLP, self).__init__()
        self.l1 = nn.Conv1d(i, m, 1)
        self.l2 = nn.Conv1d(m, o, 1)
    def forward(self, x): 
        return self.l2(F.gelu(self.l1(x)))

class MRA_FNO1d(nn.Module):
    def __init__(self, in_channels, modes, width):
        super(MRA_FNO1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.p = nn.Linear(in_channels, self.width)
        self.c1 = SpectralConv1d(width, width, modes)
        self.c2 = SpectralConv1d(width, width, modes)
        self.c3 = SpectralConv1d(width, width, modes)
        self.m1 = MLP(width, width, width)
        self.m2 = MLP(width, width, width)
        self.m3 = MLP(width, width, width)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)
        self.q = nn.Linear(width, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.p(x)
        x = x.permute(0, 2, 1) 
        
        x1 = self.c1(x); x1 = self.m1(x1); x2 = self.w1(x); x = x1 + x2; x = F.gelu(x)
        x1 = self.c2(x); x1 = self.m2(x1); x2 = self.w2(x); x = x1 + x2; x = F.gelu(x)
        x1 = self.c3(x); x1 = self.m3(x1); x2 = self.w3(x); x = x1 + x2
        
        x = x.permute(0, 2, 1)
        return self.q(x).squeeze(-1)

# ==========================================
# 3. Data Processing Utils
# ==========================================

def inject_grid(x_raw):
    """
    x_raw: (Batch, L)
    Returns: (Batch, 2, L) -> [Mean, Grid]
    """
    n_batch, n_points = x_raw.shape
    # Hardcoded Obs support -2 to 2
    grid = np.linspace(-2, 2, n_points).astype(np.float32)
    grid_expanded = np.tile(grid, (n_batch, 1))
    return np.stack([x_raw, grid_expanded], axis=1)

def prepare_data_for_n(data_dict, n_idx):
    """
    Extracts x_train/test/compare for a specific N index.
    Injects Grid.
    Returns torch tensors.
    """
    # 1. Extract raw channels (Batch, L)
    x_tr_raw = data_dict['x_train'][:, :, n_idx] 
    x_te_raw = data_dict['x_test'][:, :, n_idx]
    x_cp_raw = data_dict['x_compare'][:, :, n_idx] # (4, L)
    
    # 2. Ground Truth
    y_tr = data_dict['y_train']
    y_te = data_dict['y_test']
    y_cp = data_dict['y_compare']
    
    # 3. Inject Grid
    x_tr = inject_grid(x_tr_raw)
    x_te = inject_grid(x_te_raw)
    x_cp = inject_grid(x_cp_raw)
    
    return (
        torch.from_numpy(x_tr).float(), torch.from_numpy(y_tr).float(),
        torch.from_numpy(x_te).float(), torch.from_numpy(y_te).float(),
        torch.from_numpy(x_cp).float(), torch.from_numpy(y_cp).float()
    )

# ==========================================
# 4. Training Engine
# ==========================================

def train_single_experiment(config, x_train, y_train, x_test, y_test, x_compare, y_compare, current_n):
    
    exp_name = f"{config['experiment_params']['name_prefix']}_{current_n}"
    save_dir = os.path.join(config['experiment_params']['results_dir'], exp_name)
    ensure_dir(save_dir)
    
    # Save detailed config
    config['current_n_samples'] = int(current_n)
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
        
    # DataLoaders
    tp = config['training_params']
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                             batch_size=tp['batch_size'], shuffle=True)
    # Test loader (Batch processing for evaluation)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                            batch_size=tp['batch_size'], shuffle=False)
    
    # Model
    model = MRA_FNO1d(in_channels=2, 
                      modes=config['model_params']['modes'], 
                      width=config['model_params']['width']).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=tp['learning_rate'], weight_decay=tp['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tp['epochs'] * len(train_loader))
    
    # Losses
    loss_train_func = L2_Rel_Loss()
    loss_val_l2 = L2_Rel_Loss()
    loss_val_linf = L_inf_Loss()
    loss_val_mse = MSE_Loss_Func()
    
    # History
    history = []
    best_val_l2_rel = float('inf')
    best_val_linf_rel = float('inf')
    best_val_mse = float('inf')
    
    patience_counter = 0
    best_model_path = os.path.join(save_dir, "best_model.pth")
    
    print(f"\n>>> Start Training for N = {current_n} | Dir: {save_dir}")
    train_start_time = default_timer()
    
    # --- Training Loop ---
    for ep in range(tp['epochs']):
        t1 = default_timer()
        model.train()
        train_l2_sum = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_train_func(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_l2_sum += loss.item()
            
        avg_train_loss = train_l2_sum / len(train_loader)
        
        # Validation (on Test Set)
        model.eval()
        val_l2_sum, val_linf_sum, val_mse_sum = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_l2_sum += loss_val_l2(out, y).item()
                val_linf_sum += loss_val_linf(out, y).item()
                val_mse_sum += loss_val_mse(out, y).item()
        
        avg_val_l2 = val_l2_sum / len(test_loader)
        avg_val_linf = val_linf_sum / len(test_loader)
        avg_val_mse = val_mse_sum / len(test_loader)
        
        # History & Checkpointing
        history.append({
            "epoch": ep + 1,
            "train_loss": avg_train_loss,
            "val_l2": avg_val_l2,
            "val_linf": avg_val_linf,
            "val_mse": avg_val_mse,
            "time": default_timer() - t1
        })
        
        print(f"Ep {ep+1:03d} | Tr L2: {avg_train_loss:.4f} | Val L2: {avg_val_l2:.4f} | Val Linf: {avg_val_linf:.4f}")
        
        if avg_val_l2 < best_val_l2_rel:
            best_val_l2_rel = avg_val_l2
            # Update other best metrics corresponding to best L2 model
            best_val_linf_rel = avg_val_linf
            best_val_mse = avg_val_mse
            
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            
        if patience_counter >= tp['patience']:
            print(f"Early stopping at epoch {ep+1}")
            break
            
    train_time_total = default_timer() - train_start_time
    epochs_run = len(history)
    early_stopped = patience_counter >= tp['patience']
    
    # --- Final Evaluation (Inference) ---
    print("\n--- Running Final Evaluation on Best Model ---")
    eval_start_time = default_timer()
    
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # 1. Inference on Test Set
    pred_test_list = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            out = model(x)
            pred_test_list.append(out.cpu().numpy())
    y_pred_test = np.concatenate(pred_test_list, axis=0) # (N_test, L)
    
    # 2. Inference on Fixed Test Functions (Compare Set)
    # x_compare is small (4 samples), run in one go
    x_comp = x_compare.to(device)
    with torch.no_grad():
        y_pred_compare = model(x_comp).cpu().numpy() # (4, L)
    
    # 3. Calculate Compare Metrics (Individual & Average)
    y_true_cp = y_compare.numpy()
    
    def rel_l2_np(p, t): return np.linalg.norm(p-t) / (np.linalg.norm(t) + 1e-6)
    def rel_linf_np(p, t): return np.max(np.abs(p-t)) / (np.max(np.abs(t)) + 1e-6)
    def mse_np(p, t): return np.mean((p-t)**2)
    
    test_func_metrics = {}
    avg_metrics = {"l2": 0.0, "linf": 0.0, "mse": 0.0}
    compare_names = ["f1_ord", "f2_ord", "f3_super", "f4_super"] # Hardcoded matching generation
    
    for i in range(4):
        p, t = y_pred_compare[i], y_true_cp[i]
        m = {
            "l2": float(rel_l2_np(p, t)),
            "linf": float(rel_linf_np(p, t)),
            "mse": float(mse_np(p, t))
        }
        test_func_metrics[compare_names[i]] = m
        avg_metrics["l2"] += m["l2"]
        avg_metrics["linf"] += m["linf"]
        avg_metrics["mse"] += m["mse"]
        
    avg_metrics = {k: v/4.0 for k, v in avg_metrics.items()}
    test_func_metrics["average"] = avg_metrics
    
    eval_time_total = default_timer() - eval_start_time
    
    # --- Saving Results ---
    
    # 1. Save Estimations
    np.savez_compressed(
        os.path.join(save_dir, "estimations.npz"),
        y_pred_test=y_pred_test,
        y_pred_compare=y_pred_compare,
        y_true_test=y_test.numpy(),
        y_true_compare=y_compare.numpy()
    )
    print("Estimations saved.")
    
    # 2. Detailed Summary (Local)
    local_summary = {
        "n_samples": int(current_n),
        "train_time": train_time_total,
        "eval_time": eval_time_total,
        "epochs": epochs_run,
        "early_stop": early_stopped,
        "best_val_l2_rel": best_val_l2_rel,
        "best_val_linf_rel": best_val_linf_rel,
        "best_val_mse": best_val_mse,
        "test_functions": test_func_metrics,
        "history": history
    }
    with open(os.path.join(save_dir, "detailed_summary.json"), 'w') as f:
        json.dump(local_summary, f, indent=4)
        
    # 3. Simplified Summary (Return for Global)
    simple_summary = {
        "n_samples": int(current_n),
        "train_time": train_time_total,
        "eval_time": eval_time_total,
        "best_val_l2_rel": best_val_l2_rel,
        "best_val_linf_rel": best_val_linf_rel,
        "best_val_mse": best_val_mse,
        "epochs": epochs_run,
        "early_stop": early_stopped,
        "test_functions": test_func_metrics
    }
    
    return simple_summary

# ==========================================
# 5. Main Entry
# ==========================================

def main():
    if not os.path.exists('train_config.json'):
        print("Error: train_config.json not found.")
        return

    with open('train_config.json', 'r') as f:
        config = json.load(f)
        
    data_path = config['data_params']['path']
    print(f"Loading Dataset: {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    
    counts_list = data['counts_list']
    print(f"Target N samples: {counts_list}")
    
    all_experiments_summary = {}
    
    for i, n_val in enumerate(counts_list):
        print(f"\n========================================")
        print(f"EXPERIMENT: N_SAMPLES = {n_val}")
        print(f"========================================")
        
        # Prepare Data
        x_tr, y_tr, x_te, y_te, x_cp, y_cp = prepare_data_for_n(data, i)
        
        # Train & Evaluate
        res = train_single_experiment(config, x_tr, y_tr, x_te, y_te, x_cp, y_cp, n_val)
        
        # Collect Summary
        all_experiments_summary[str(n_val)] = res
        
    # Save Global Summary
    global_res_path = os.path.join(config['experiment_params']['results_dir'], "all_experiments_summary.json")
    with open(global_res_path, 'w') as f:
        json.dump(all_experiments_summary, f, indent=4)
        
    print(f"\nAll Done. Global summary saved to {global_res_path}")

if __name__ == "__main__":
    main()