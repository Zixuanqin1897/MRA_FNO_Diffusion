import torch
import torch.nn as nn
import numpy as np

class GaussianFourierProjection(nn.Module):
    """
    Embed scalar t into a vector using random Fourier features.
    """
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Random weights fixed at initialization
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        # t: (Batch,) -> (Batch, 1)
        t_proj = t[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)

class Dense(nn.Module):
    """
    Simple Linear + SiLU for time embedding.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.dense(x))

class ScoreNet(nn.Module):
    """
    Time-dependent Score Network.
    Modified for Dirichlet BCs: Uses 'zeros' padding instead of 'circular'.
    """
    def __init__(self, channels=64, num_layers=8, L=64):
        super().__init__()
        self.channels = channels
        self.L = L
        
        # Time Embedding
        self.time_embed_dim = 64 
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(self.time_embed_dim),
            Dense(self.time_embed_dim, channels)
        )
        
        self.convs = nn.ModuleList()
        
        # Input Layer: 1 -> channels
        # padding_mode='zeros' implies u=0 outside domain (Dirichlet-consistent)
        self.convs.append(nn.Conv1d(1, channels, kernel_size=3, padding=1, padding_mode='zeros'))
        
        # Hidden Layers
        for _ in range(num_layers - 2):
            self.convs.append(nn.Conv1d(channels, channels, kernel_size=3, padding=1, padding_mode='zeros'))
            
        # Output Layer: channels -> 1 (Score dimension)
        self.final_conv = nn.Conv1d(channels, 1, kernel_size=3, padding=1, padding_mode='zeros')
        
        self.act = nn.SiLU()

    def forward(self, x, t):
        """
        x: (batch, 1, L)
        t: (batch,)
        """
        # Embed time: (Batch,) -> (Batch, Channels) -> (Batch, Channels, 1)
        t_embed = self.time_embed(t)[:, :, None]
        
        h = x
        for conv in self.convs:
            h = conv(h)
            # Add time embedding to every layer feature map
            h = h + t_embed
            h = self.act(h)
            
        out = self.final_conv(h)
        return out
    
def get_sigma_schedule(t, sigma_min, sigma_max):
    """
    Exponential schedule for VE-SDE (Variance Exploding).
    sigma(t) = sigma_min * (sigma_max / sigma_min)^t
    """
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)
    return sigma_min * (sigma_max / sigma_min) ** t

def dsm_loss_fn(model, x, loss_type='l2', sigma_min=0.01, sigma_max=10.0, eps=1e-5):
    """
    Standard Denoising Score Matching Loss (MSE).
    Objective: || s_theta(x_t, t) * sigma_t + z ||^2
    """
    batch_size = x.shape[0]
    
    # 1. Sample t
    t = torch.rand(batch_size, device=x.device) * (1. - eps) + eps
    
    # 2. Sigma(t)
    sigma_t = get_sigma_schedule(t, sigma_min, sigma_max).view(batch_size, 1, 1)
    
    # 3. Perturb data
    z = torch.randn_like(x)
    x_t = x + z * sigma_t
    
    # 4. Predict Score
    score_pred = model(x_t, t)
    
    # 5. Calculate Loss Term
    # We want to minimize || score * sigma_t + z ||^2
    # Equivalently: prediction = score * sigma_t, target = -z
    # diff = prediction - target = score * sigma_t + z
    
    target_term = score_pred * sigma_t + z
    
    # Standard MSE Loss: Sum over spatial dims, Mean over batch
    # sum dim=(1, 2) assumes shape (Batch, Channel, Length)
    loss = torch.mean(torch.sum(target_term**2, dim=(1, 2)))
        
    return loss