# lib/edge_latents.py
import torch
import torch.nn.functional as F

def safe_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # logits: [..., K]
    logits = logits - logits.max(dim=dim, keepdim=True).values
    return torch.softmax(logits, dim=dim)

def gumbel_softmax_sample(logits: torch.Tensor, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
    """
    logits: [..., K]
    returns: [..., K] (soft or straight-through hard)
    """
    return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

def kl_categorical(q_probs: torch.Tensor, p_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    KL(q||p) for categorical distributions.
    q_probs, p_probs: [..., K], sum to 1 along last dim
    returns: [...] (KL per distribution)
    """
    q = torch.clamp(q_probs, eps, 1.0)
    p = torch.clamp(p_probs, eps, 1.0)
    return torch.sum(q * (torch.log(q) - torch.log(p)), dim=-1)

class TimeGridInterpolator:
    """
    Efficient linear interpolation on a fixed 1D time grid.
    Stores t_grid once and allows querying.
    """
    def __init__(self, t_grid: torch.Tensor):
        assert t_grid.ndim == 1
        self.t_grid = t_grid

    def interp(self, values_grid: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        values_grid: [T, ...]
        t: scalar tensor or shape [...]
        returns: [...]
        """
        tg = self.t_grid
        T = tg.numel()

        if T == 1:
            return values_grid[0]

        # clamp time into grid range
        t_clamped = torch.clamp(t, tg[0], tg[-1])

        # find right index: idx such that tg[idx] <= t < tg[idx+1]
        idx = torch.searchsorted(tg, t_clamped, right=True) - 1
        idx = torch.clamp(idx, 0, T - 2)

        t0 = tg[idx]
        t1 = tg[idx + 1]
        w = (t_clamped - t0) / torch.clamp(t1 - t0, min=1e-12)

        v0 = values_grid[idx]
        v1 = values_grid[idx + 1]
        return (1.0 - w) * v0 + w * v1
