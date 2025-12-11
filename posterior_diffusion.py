import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import argparse
from tqdm import tqdm

from score_model import ScoreNet, get_sigma_schedule

# ==========================================
# 1. Adaptive Scheduler & Utils
# ==========================================
class AdaptiveRefinementScheduler:
    def __init__(self, 
                 method='slope',          
                 freq_range=(1.0, 16.0),  
                 slope_bounds=(4.0, 6.0), 
                 t_bounds=(0.6, 0.1)      
                 ):
        self.method = method
        self.f_min, self.f_max = freq_range
        self.s_min, self.s_max = slope_bounds
        self.t_max, self.t_min = t_bounds

    def compute_psd_metrics(self, y_pred, domain_length=2.0):
        if torch.is_tensor(y_pred):
            y_pred = y_pred.detach().cpu().numpy()
        if y_pred.ndim == 1:
            y_pred = y_pred[None, :]
            
        n_samples, n_points = y_pred.shape
        fft_vals = np.fft.rfft(y_pred, axis=-1)
        psd = np.mean(np.abs(fft_vals)**2, axis=0) 
        
        d = domain_length / n_points
        freqs = np.fft.rfftfreq(n_points, d=d)
        
        mask = (freqs >= self.f_min) & (freqs <= self.f_max)
        if np.sum(mask) < 2:
            return 4.0 
            
        x_fit = np.log(freqs[mask] + 1e-9)
        y_fit = np.log(psd[mask] + 1e-9)
        try:
            slope, _ = np.polyfit(x_fit, y_fit, 1)
            alpha = -slope
        except:
            alpha = 4.0
        return alpha

    def get_t_start(self, y_pred, verbose=True):
        alpha = self.compute_psd_metrics(y_pred)
        metric = alpha
        norm = (metric - self.s_min) / (self.s_max - self.s_min)
        norm = np.clip(norm, 0.0, 1.0)
        t_start = self.t_max + norm * (self.t_min - self.t_max)
        if verbose:
            print(f"    [Adaptive] PSD Slope={-alpha:.2f} -> t_start={t_start:.3f}")
        return t_start

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def load_json(path):
    with open(path, 'r') as f: return json.load(f)

# ==========================================
# 2. Real Data Hybrid Guidance
# ==========================================
class RealDataHybridGuidance:
    def __init__(self, 
                 fno_prediction,   # (1, 64) on [-1, 1]
                 raw_obs_batch,    # (N, 64) on [-2, 2]
                 device='cpu',
                 k_anchor=4, 
                 weight_anchor=100.0, 
                 weight_chain=200.0,   
                 weight_diag=50.0):   
        
        self.device = device 
        self.comp_device = torch.device('cpu') 
        
        self.k_anchor = k_anchor
        self.weight_anchor = weight_anchor
        self.weight_chain = weight_chain
        self.weight_diag = weight_diag
        
        # === 1. Prepare Anchor (From FNO) ===
        # FNO is on [-1, 1], 64 points.
        # To be consistent with the physics of the Obs, we also Pad FNO to [-2, 2]
        with torch.no_grad():
            fno_cpu = fno_prediction.to(self.comp_device)
            self.fno_pad = F.pad(fno_cpu, (32, 32), "constant", 0) 
            
            # FFT of Padded FNO (128 pts -> 65 freq pts)
            self.fno_fft = torch.fft.rfft(self.fno_pad, dim=-1)
            
            # Use the first k_anchor frequencies of the PADDED spectrum
            self.target_anchor = self.fno_fft[..., :self.k_anchor]

        # === 2. Prepare Bispectrum Targets (From Raw Obs) ===
        # Raw Obs is on [-2, 2], 64 points.
        with torch.no_grad():
            obs_cpu = raw_obs_batch.to(self.comp_device)
            # FFT of Obs (64 pts -> 33 freq pts)
            self.target_chain_phase, self.target_diag_phase = \
                self._estimate_bispectrum_from_batch(obs_cpu)

    def _estimate_bispectrum_from_batch(self, obs_fft_input):
        if obs_fft_input.ndim == 2 and obs_fft_input.is_complex():
             Y = obs_fft_input
        else:
             Y = torch.fft.rfft(obs_fft_input, dim=-1)
             
        n_freq = Y.shape[-1]
        k_range = torch.arange(1, n_freq - 1, device=self.comp_device)
        
        # Chain
        Y_1 = Y[:, 1:2]
        Y_k = Y[:, k_range]
        Y_k1 = Y[:, k_range+1]
        B_chain_mean = torch.mean(Y_1 * Y_k * torch.conj(Y_k1), dim=0).unsqueeze(0)
        chain_phase = B_chain_mean / (torch.abs(B_chain_mean) + 1e-8)
        
        # Diag
        valid_k = n_freq // 2
        Y_k_diag = Y[:, :valid_k]
        Y_2k_diag = Y[:, 0:2*valid_k:2]
        min_len = min(Y_k_diag.shape[-1], Y_2k_diag.shape[-1])
        B_diag_mean = torch.mean((Y_k_diag[:, :min_len]**2) * torch.conj(Y_2k_diag[:, :min_len]), dim=0).unsqueeze(0)
        diag_phase = B_diag_mean / (torch.abs(B_diag_mean) + 1e-8)
        
        return chain_phase, diag_phase

    def compute_grad(self, x_t, score_t, sigma_t):
        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True) # (B, 1, 64)
            s_t = sigma_t if not isinstance(sigma_t, torch.Tensor) else sigma_t.view(-1, 1, 1)
            
            # Tweedie Estimate (Clean Signal on [-1, 1])
            hat_x0 = x_in + (s_t ** 2) * score_t.detach()
            hat_x0_cpu = hat_x0.to(self.comp_device)
            
            # === Zero-Padding ===
            # Pad hat_x0 (64) -> hat_x0_pad (128)
            # This aligns the physical domain to [-2, 2]
            hat_x0_pad = F.pad(hat_x0_cpu, (32, 32), "constant", 0)
            
            # FFT (128 pts -> 65 freq bins)
            # Bin resolution: 0.25 Hz
            pred_fft_full = torch.fft.rfft(hat_x0_pad, dim=-1)
            
            # === Frequency Truncation ===
            # Raw Obs has 64 pts -> 33 freq bins (0 to 8 Hz)
            pred_fft_aligned = pred_fft_full[..., :33] 
            
            # Soft Norm
            mag_pred = torch.abs(pred_fft_aligned)
            epsilon_pred = 1e-3 * torch.max(mag_pred).detach() + 1e-6
            pred_fft_phase = pred_fft_aligned / (mag_pred + epsilon_pred)
            
            # --- 1. Anchor Loss ---
            # Use the first k_anchor bins of the PADDED spectrum
            diff_anchor = pred_fft_full[..., :self.k_anchor] - self.target_anchor
            loss_anchor = torch.mean(diff_anchor.real**2 + diff_anchor.imag**2)
            
            # --- 2. Bispectrum Loss (Chain) ---
            # Use the aligned (truncated) spectrum
            n_freq = pred_fft_aligned.shape[-1] # Should be 33
            k_range = torch.arange(1, n_freq - 1, device=self.comp_device)
            
            Bispec_phase_pred = pred_fft_phase[..., 1:2] * pred_fft_phase[..., k_range] * torch.conj(pred_fft_phase[..., k_range+1])
            diff_chain = Bispec_phase_pred - self.target_chain_phase
            loss_chain = torch.mean(diff_chain.real**2 + diff_chain.imag**2)

            # --- 3. Bispectrum Loss (Diag) ---
            valid_k = n_freq // 2
            min_len = self.target_diag_phase.shape[-1]
            
            Bispec_diag_pred = (pred_fft_phase[..., :valid_k][..., :min_len]**2) * torch.conj(pred_fft_phase[..., 0:2*valid_k:2][..., :min_len])
            diff_diag = Bispec_diag_pred - self.target_diag_phase
            loss_diag = torch.mean(diff_diag.real**2 + diff_diag.imag**2)
            
            total_loss = (self.weight_anchor * loss_anchor + 
                          self.weight_chain * loss_chain + 
                          self.weight_diag * loss_diag)
            
            # Gradient will backpropagate through:
            # Slicing -> FFT -> Padding -> hat_x0 -> x_in
            grad = torch.autograd.grad(total_loss, x_in)[0]
            
            if grad.device != x_in.device:
                grad = grad.to(x_in.device)
            
        return grad, total_loss.item()

# ==========================================
# 3. Refinement Loop
# ==========================================
def adaptive_refinement_loop(model, guide, shape, device, diff_config, 
                             init_image, t_start_adaptive, 
                             guidance_scale=5.0):
    
    sigma_min = diff_config['diffusion']['sigma_min']
    sigma_max = diff_config['diffusion']['sigma_max']
    n_steps = 1000 
    
    ts = torch.linspace(1.0, 1e-5, n_steps, device=device)
    start_idx = (torch.abs(ts - t_start_adaptive)).argmin().item()
    
    start_sigma = get_sigma_schedule(ts[start_idx], sigma_min, sigma_max)
    
    if init_image.ndim == 2: init_image = init_image.unsqueeze(1)
    if init_image.shape[0] != shape[0]:
        init_image = init_image.repeat(shape[0], 1, 1)
        
    noise = torch.randn(shape, device=device)
    x = init_image + start_sigma * noise
    
    step_size = ts[0] - ts[1]
    log_sig_ratio = np.log(sigma_max / sigma_min)
    
    # Progress bar handled in batch function
    
    for i in range(start_idx, n_steps):
        t_current = ts[i]
        sigma_current = get_sigma_schedule(t_current, sigma_min, sigma_max)
        
        t_batch = torch.ones(shape[0], device=device) * t_current
        score = model(x, t_batch) 
        
        grad_guide, _ = guide.compute_grad(x, score, sigma_current)
        grad_guide = grad_guide.view(shape)
        
        # Tricks
        mask = torch.ones_like(grad_guide)
        mask[..., :3] = 0.0
        mask[..., -3:] = 0.0
        grad_guide = grad_guide * mask
        grad_guide = torch.clamp(grad_guide, -1.0, 1.0)
        
        grad_norm = torch.norm(grad_guide.reshape(shape[0], -1), dim=-1)
        score_norm = torch.norm(score.reshape(shape[0], -1), dim=-1)
        scale_factor = score_norm / (grad_norm + 1e-8)
        
        modified_score = score - guidance_scale * grad_guide * scale_factor.view(shape[0], 1, 1)
        
        g2_t = 2 * (sigma_current ** 2) * log_sig_ratio
        drift = g2_t * modified_score * step_size
        g_t = torch.sqrt(g2_t)
        z = torch.randn_like(x)
        diffusion = g_t * torch.sqrt(step_size) * z
        
        x = x + drift + diffusion
            
    return x

# ==========================================
# 4. Batch Processing Function
# ==========================================
def run_batch_evaluation(
    model, 
    scheduler, 
    diff_cfg,
    device,
    y_pred_batch,      # (N, L) FNO estimates
    y_true_batch,      # (N, L) Ground Truth
    raw_obs_path,      # Path to .npz containing "signal_i" raw obs
    save_path,         # Where to save results
    guidance_scale=10.0,
    batch_name="Batch"
):
    print(f"\n============================================")
    print(f"Starting {batch_name} Evaluation")
    print(f"Loading Raw Obs: {raw_obs_path}")
    print(f"============================================")
    
    # Load Raw Observations
    raw_obs_data = np.load(raw_obs_path, allow_pickle=True)
    sorted_obs_keys = sorted(list(raw_obs_data.keys()), key=lambda x: int(x.split('_')[1]))
    
    n_samples = y_pred_batch.shape[0]
    L = y_pred_batch.shape[1]
    results = {}
    
    if len(sorted_obs_keys) < n_samples:
        n_samples = len(sorted_obs_keys)

    for i in range(n_samples):
        key = sorted_obs_keys[i]
        print(f"Processing {batch_name} [{i+1}/{n_samples}] - Key: {key}")
        
        fno_target = y_pred_batch[i].unsqueeze(0)      # (1, L)
        ground_truth = y_true_batch[i].cpu().numpy()   # (L,) numpy
        
        obs_item = raw_obs_data[key].item()
        # Raw Obs (N, 64) from file
        raw_obs_batch_tensor = torch.from_numpy(obs_item['y_obs']).float().to(device)
        
        t_adaptive = scheduler.get_t_start(fno_target.cpu().numpy(), verbose=True)
        
        guidance = RealDataHybridGuidance(
            fno_prediction=fno_target,
            raw_obs_batch=raw_obs_batch_tensor,
            device=device,
            weight_anchor=10.0,
            weight_chain=200.0,
            weight_diag=100.0
        )
        
        num_posterior_samples = 16
        
        posterior_samples = adaptive_refinement_loop(
            model=model,
            guide=guidance,
            shape=(num_posterior_samples, 1, L),
            device=device,
            diff_config=diff_cfg,
            init_image=fno_target,
            t_start_adaptive=t_adaptive,
            guidance_scale=guidance_scale
        )
        
        samples_np = posterior_samples.detach().cpu().numpy().squeeze()
        post_mean = samples_np.mean(axis=0)
        post_std = samples_np.std(axis=0)
        
        results[key] = {
            "posterior_samples": samples_np,
            "posterior_mean": post_mean,
            "posterior_std": post_std,
            "fno_pred": fno_target.squeeze().cpu().numpy(),
            "ground_truth": ground_truth,
            "t_start_used": float(t_adaptive)
        }
    
    np.savez(save_path, **results)
    print(f"Saved {batch_name} results to {save_path}")


# ==========================================
# 5. Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="posterior_config.json")
    args = parser.parse_args()
    
    post_cfg = load_json(args.config)
    device = get_device()
    print(f"Using device: {device}")

    adapt_cfg = post_cfg.get('adaptive', {})
    
    scheduler = AdaptiveRefinementScheduler(
        method=adapt_cfg.get('method', 'slope'),
        freq_range=tuple(adapt_cfg.get('freq_range', [1.0, 16.0])),
        slope_bounds=tuple(adapt_cfg.get('slope_bounds', [4.0, 6.0])),
        t_bounds=tuple(adapt_cfg.get('t_bounds', [0.6, 0.1]))
    )

    print("Loading ScoreNet...")
    diff_cfg = load_json(post_cfg['model']['config_path'])
    model = ScoreNet(
        channels=diff_cfg['model']['channels'],
        num_layers=diff_cfg['model']['layers'],
        L=diff_cfg['model']['L']
    ).to(device)
    model.load_state_dict(torch.load(post_cfg['model']['checkpoint'], map_location=device, weights_only=True))
    model.eval()
    
    fno_path = post_cfg['data_paths']['fno_estimations']
    print(f"Loading FNO Estimations from {fno_path}")
    fno_data = np.load(fno_path)
    
    y_pred_compare = torch.from_numpy(fno_data['y_pred_compare']).float().to(device) 
    y_true_compare = torch.from_numpy(fno_data['y_true_compare']).float().to(device) 
    
    y_pred_test = torch.from_numpy(fno_data['y_pred_test'][:32]).float().to(device)   
    y_true_test = torch.from_numpy(fno_data['y_true_test'][:32]).float().to(device)   
    
    guidance_scale = post_cfg.get('sampling', {}).get('guidance_scale', 5.0)

    # --- Run Batch 1: Compare ---
    run_batch_evaluation(
        model=model,
        scheduler=scheduler,
        diff_cfg=diff_cfg,
        device=device,
        y_pred_batch=y_pred_compare,
        y_true_batch=y_true_compare,
        raw_obs_path=post_cfg['data_paths']['raw_observations_compare'],
        save_path=post_cfg['data_paths']['save_path_compare'],
        guidance_scale=guidance_scale,
        batch_name="COMPARE_SET"
    )
    
    # --- Run Batch 2: Test ---
    run_batch_evaluation(
        model=model,
        scheduler=scheduler,
        diff_cfg=diff_cfg,
        device=device,
        y_pred_batch=y_pred_test,
        y_true_batch=y_true_test,
        raw_obs_path=post_cfg['data_paths']['raw_observations_test'],
        save_path=post_cfg['data_paths']['save_path_test'],
        guidance_scale=guidance_scale,
        batch_name="TEST_SET"
    )
    
    print("\nAll Evaluation Batches Completed Successfully.")

if __name__ == "__main__":
    main()