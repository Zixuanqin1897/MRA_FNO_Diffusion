import numpy as np
import os
import json
import argparse
from typing import Dict, List, Tuple, Optional, Callable
from scipy.interpolate import interp1d

# =========================================================
# 1. Shift Samplers (Domain: [-1.0, 1.0])
# =========================================================

def get_shift_sampler(config: Dict) -> Callable[[int], np.ndarray]:
    """Factory to create a shift sampler function."""
    sim_conf = config.get("simulation", {})
    shift_conf = sim_conf.get("shift_sampler", {"type": "uniform"})
    
    s_type = shift_conf.get("type", "uniform").lower()
    low = shift_conf.get("low", -1.0)
    high = shift_conf.get("high", 1.0)
    
    print(f"[Shift Sampler] Type: {s_type}, Range: [{low}, {high}]")

    if s_type == "uniform":
        def sampler(size):
            return np.random.uniform(low, high, size=size)
        return sampler

    elif s_type == "g1":
        # g1 is equivalent to sum of two Uniform[-0.5, 0.5]
        def sampler(size):
            u1 = np.random.uniform(-0.5, 0.5, size=size)
            u2 = np.random.uniform(-0.5, 0.5, size=size)
            return np.clip(u1 + u2, low, high) 
        return sampler

    elif s_type == "f1":
        # f1(x) = g1(x) + g1(pi * x)
        
        # 1. Define grid for numerical Inverse CDF
        x_grid = np.linspace(-1.0, 1.0, 10000)
        
        # 2. Define g1(x): Triangular on [-1, 1], peak at 0 height 1
        # Analytical: max(0, 1 - |x|)
        g1_x = np.maximum(0, 1.0 - np.abs(x_grid))
        
        # 3. Define g1(pi * x): Triangular on [-1/pi, 1/pi]
        # Analytical: max(0, 1 - |pi * x|)
        g1_pi_x = np.maximum(0, 1.0 - np.abs(np.pi * x_grid))
        
        # 4. Combine
        pdf = g1_x + g1_pi_x
        
        # 5. Normalize to be a PDF
        pdf_sum = np.trapz(pdf, x_grid)
        pdf_norm = pdf / pdf_sum
        
        # 6. Build CDF
        cdf = np.cumsum(pdf_norm)
        cdf = cdf / cdf[-1] # Ensure exactly 1.0 at end
        
        # 7. Create Interpolator
        inv_cdf_func = interp1d(cdf, x_grid, bounds_error=False, fill_value=(low, high))
        
        def sampler(size):
            u = np.random.rand(size)
            return inv_cdf_func(u)
        return sampler

    else:
        raise ValueError(f"Unknown shift sampler type: {s_type}")


# =========================================================
# 2. GP Noise Generator (FFT based)
# =========================================================

def generate_gp_noise(n_obs: int, t_grid: np.ndarray, sigma: float, lam: float) -> np.ndarray:
    """
    Generates GP noise with RBF kernel using FFT embedding.
    """
    n = t_grid.size
    dt = t_grid[1] - t_grid[0]
    
    # RBF Covariance Row
    r = np.arange(n) * dt
    cov_row = (sigma**2) * np.exp(-r**2 / (2 * lam**2))
    
    # Circulant Embedding
    L = 2 * n
    emb_row = np.concatenate([cov_row, np.zeros(L - 2*n + 1), cov_row[1:][::-1]])
    
    # Power Spectrum
    S = np.fft.fft(emb_row).real
    S = np.maximum(S, 0)
    
    # Generate Noise
    noise_white = np.random.randn(n_obs, L)
    noise_fft = np.fft.fft(noise_white, axis=1)
    colored_fft = noise_fft * np.sqrt(S)
    colored_noise = np.fft.ifft(colored_fft, axis=1).real
    
    return colored_noise[:, :n]


# =========================================================
# 3. Core Physics: Shift + Noise -> Obs -> Mean
# =========================================================

def apply_shift_and_interp(y_signal: np.ndarray, f_grid: np.ndarray, t_grid: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """
    y_signal: Signal defined on f_grid (e.g., [-1, 1])
    t_grid: Observation grid (e.g., [-2, 2])
    shifts: Random shifts
    
    Obs(t) = Signal(t - shift)
    """
    # Query points: for each obs i, we want signal at (t_grid - shift_i)
    # If (t - shift) is outside f_grid ([-1, 1]), interp returns 0.
    t_query = t_grid[None, :] - shifts[:, None]
    
    # 1D Linear Interpolation
    # f_grid must be increasing
    y_interp = np.interp(t_query.ravel(), f_grid, y_signal, left=0.0, right=0.0)
    
    return y_interp.reshape(len(shifts), len(t_grid))


def compute_cumulative_means(y_obs: np.ndarray, counts: List[int]) -> np.ndarray:
    """Compute mean of observations for different sample sizes."""
    means = []
    for c in counts:
        m = np.mean(y_obs[:c, :], axis=0)
        means.append(m)
    return np.stack(means, axis=-1)


# =========================================================
# 4. Fixed Test Functions
# =========================================================

def generate_test_functions_on_grid(f_grid: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Strict implementation of the table provided.
    f_grid is assumed to be [-1, 1] (or covering the support).
    """
    x = f_grid
    dx = x[1] - x[0]
    
    def normalize(y):
        m = np.max(np.abs(y))
        return y / (m + 1e-12)

    # --- F1: Ordinary Smooth ---
    # g1 = 1_[-0.5, 0.5] * 1_[-0.5, 0.5]
    g1 = np.maximum(0.0, 1.0 - np.abs(x))
    
    # f1(x) = g1(x) + g1(pi * x)
    # g1(pi*x) = max(0, 1 - |pi*x|)
    g1_pi = np.maximum(0.0, 1.0 - np.abs(np.pi * x))
    
    y1 = normalize(g1 + g1_pi)
    
    # --- F2: Ordinary Smooth ---
    # g2 = 1_[-0.25, 0.25] convolved 4 times.
    rect = ((x >= -0.25) & (x <= 0.25)).astype(float)
    
    # Convolution 1 (Rect * Rect)
    c1 = np.convolve(rect, rect, mode='same') * dx
    
    # Convolution 2 (c1 * Rect)
    c2 = np.convolve(c1, rect, mode='same') * dx
    
    # Convolution 3 (c2 * Rect) -> This is the 4-term convolution g2
    g2 = np.convolve(c2, rect, mode='same') * dx
    
    # f2(x) = g2(x) + g2(pi * x)
    g2_pi = np.interp(np.pi * x, x, g2, left=0.0, right=0.0)
    
    y2 = normalize(g2 + g2_pi)
    
    # --- F3: Super Smooth ---
    # g3(x) = exp(-20 x^2) cos(8x)
    # f3(x) = g3(x - 0.3)
    x_shift = x - 0.3
    g3_shifted = np.exp(-20 * x_shift**2) * np.cos(8 * x_shift)
    y3 = normalize(g3_shifted)
    
    # --- F4: Super Smooth (Mixture) ---
    # f4 = pdf(Mixture)
    # mu1 = -0.3, var = 0.02
    # mu2 = 0.5, var = 0.02
    sigma = np.sqrt(0.02)
    
    def gauss_pdf(val, mu, sig):
        return (1.0 / (np.sqrt(2 * np.pi) * sig)) * np.exp(-(val - mu)**2 / (2 * sig**2))
    
    # Assuming equal weights for mixture "pdf of mu1, mu2 mixture"
    f4_raw = 0.5 * gauss_pdf(x, -0.3, sigma) + 0.5 * gauss_pdf(x, 0.5, sigma)
    y4 = normalize(f4_raw)
    
    functions = np.stack([y1, y2, y3, y4])
    names = ["f1_ord", "f2_ord", "f3_super", "f4_super"]
    return functions, names


# =========================================================
# 5. Main Processing Routine
# =========================================================

def process_signals(
    signals_true: np.ndarray, 
    f_grid: np.ndarray,    # Signal Grid [-1, 1]
    t_grid: np.ndarray,    # Obs Grid [-2, 2]
    config: Dict,
    shift_sampler: Callable,
    counts_list: List[int],
    preview_indices: List[int] = [] 
) -> Tuple[np.ndarray, Optional[Dict]]:
    
    sim_conf = config['simulation']
    noise_conf = config['noise']
    n_obs_max = sim_conf['n_observations']
    
    n_signals, _ = signals_true.shape
    n_points_obs = t_grid.size
    
    # Output: (n_signals, n_points_obs, n_counts)
    x_out = np.zeros((n_signals, n_points_obs, len(counts_list)), dtype=np.float32)
    
    preview_data = {}
    
    print(f"Processing {n_signals} signals...")
    
    for i in range(n_signals):
        # 1. Params
        sigma = np.random.uniform(*noise_conf['sigma_range'])
        lam = np.random.uniform(*noise_conf['length_scale_range'])
        
        # 2. Shift & Noise
        shifts = shift_sampler(n_obs_max)
        noise = generate_gp_noise(n_obs_max, t_grid, sigma, lam)
        
        # 3. Obs Generation
        # apply_shift_and_interp maps f_grid (signal) to t_grid (obs)
        y_clean_shifted = apply_shift_and_interp(signals_true[i], f_grid, t_grid, shifts)
        y_obs = y_clean_shifted + noise
        
        # 4. Mean
        x_out[i] = compute_cumulative_means(y_obs, counts_list)
        
        # 5. Preview
        if i in preview_indices:
            preview_data[f"signal_{i}"] = {
                "y_true": signals_true[i], # on f_grid
                "y_obs": y_obs.astype(np.float32), # on t_grid
                "f_grid": f_grid,
                "t_grid": t_grid,
                "shifts": shifts,
                "sigma": sigma,
                "lambda": lam
            }
            
    return x_out, preview_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="data_config.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    # 1. Load Data
    print("Loading True Signals...")
    ext_conf = cfg['external_true_signals']
    root = ext_conf.get('dataset_root', '.')
    
    data_train = np.load(os.path.join(root, ext_conf['train_file']))
    data_test = np.load(os.path.join(root, ext_conf['test_file']))
    
    y_train_true = data_train['y_true']
    y_test_true = data_test['y_true']
    f_grid = data_train['x_grid'] 
    
    # Validation: Ensure signal is on [-1, 1] (or close enough)
    if np.min(f_grid) > -0.9 or np.max(f_grid) < 0.9:
        print("WARNING: Input signal grid does not seem to cover [-1, 1].")
        print(f"Current range: [{np.min(f_grid):.2f}, {np.max(f_grid):.2f}]")
    
    # 2. Setup Observation Grid (The 'Sensor' Grid)
    # Target: [-2, 2]
    res = cfg['simulation']['resolution']
    t_grid = np.linspace(-2.0, 2.0, res)
    print(f"Observation Grid: [-2, 2] with {res} points.")
    
    # 3. Setup Logic
    shift_sampler = get_shift_sampler(cfg)
    
    max_obs = cfg['simulation']['n_observations']
    counts = []
    curr = 1024
    while curr <= max_obs:
        counts.append(curr)
        curr *= 2
    if counts[-1] != max_obs: counts.append(max_obs)
    
    print(f"Stats channels will be generated for N = {counts}")

    # 4. Process Train
    x_train, _ = process_signals(
        y_train_true, f_grid, t_grid, cfg, shift_sampler, counts, 
        preview_indices=[] if cfg['simulation'].get('save_preview_npz') else []
    )
    
    # 5. Process Test
    test_preview_indices = list(range(32)) if cfg['simulation'].get('save_preview_npz') else []
    x_test, preview_test = process_signals(
        y_test_true, f_grid, t_grid, cfg, shift_sampler, counts,
        preview_indices=test_preview_indices
    )

    # 6. Process Fixed Test Functions (Strict definitions)
    # Generated on f_grid (Signal domain)
    y_fixed, fixed_names = generate_test_functions_on_grid(f_grid)
    
    x_compare, preview_fixed = process_signals(
        y_fixed, f_grid, t_grid, cfg, shift_sampler, counts,
        preview_indices=[0, 1, 2, 3] if cfg['simulation'].get('save_preview_npz') else []
    )
    
    # 7. Save
    out_file = cfg['dataset_filename']
    print(f"Saving dataset to {out_file}...")
    np.savez_compressed(
        out_file,
        x_train=x_train, y_train=y_train_true,
        x_test=x_test,   y_test=y_test_true,
        x_compare=x_compare, y_compare=y_fixed,
        t_grid=t_grid, f_grid=f_grid,
        counts_list=np.array(counts),
        compare_names=fixed_names,
        config=cfg
    )
    
    # 8. Save Preview
    if cfg['simulation'].get('save_preview_npz'):
        test_prev_file = out_file.replace(".npz", "_test_preview.npz")
        print(f"Saving Test preview (first 32) to {test_prev_file}...")
        np.savez_compressed(test_prev_file, **preview_test)
        
        comp_prev_file = out_file.replace(".npz", "_compare_preview.npz")
        print(f"Saving Compare preview to {comp_prev_file}...")
        np.savez_compressed(comp_prev_file, **preview_fixed)

    print("Done.")

if __name__ == "__main__":
    main()