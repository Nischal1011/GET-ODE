"""LG-ODE-compatible likelihood and explicit CSG-ODE evaluation metrics."""

from __future__ import annotations

from torch import Tensor


def lgode_masked_average(values: Tensor, mask: Tensor) -> Tensor:
    """Average over time per trajectory/feature, then across nodes and features."""

    if values.ndim != 5:
        raise ValueError("values must have shape [S, B, T, N, F]")
    expanded_mask = mask.unsqueeze(0).to(dtype=values.dtype)
    sums = (values * expanded_mask).sum(dim=2)
    counts = expanded_mask.sum(dim=2).clamp_min(1.0)
    return (sums / counts).mean(dim=(1, 2, 3))


def gaussian_log_likelihood(
    predictions: Tensor,
    targets: Tensor,
    mask: Tensor,
    observation_std: float,
) -> Tensor:
    squared = (predictions - targets.unsqueeze(0)).square()
    return lgode_masked_average(-squared / (2.0 * observation_std**2), mask)


def lgode_mse(predictions: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    return lgode_masked_average((predictions - targets.unsqueeze(0)).square(), mask).mean()


def masked_mse(predictions: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Element-weighted MSE, averaged across posterior trajectory samples."""

    expanded_mask = mask.unsqueeze(0).to(dtype=predictions.dtype)
    numerator = ((predictions - targets.unsqueeze(0)).square() * expanded_mask).sum()
    denominator = expanded_mask.sum().clamp_min(1.0) * predictions.shape[0]
    return numerator / denominator


def metric_bundle(
    predictions: Tensor,
    targets: Tensor,
    target_mask: Tensor,
    unobserved_mask: Tensor,
) -> dict[str, Tensor]:
    raw_all = masked_mse(predictions, targets, target_mask)
    raw_unobserved = masked_mse(predictions, targets, unobserved_mask)
    compatible = lgode_mse(predictions, targets, target_mask)
    return {
        "mse_all": raw_all,
        "mse_unobserved": raw_unobserved,
        "mse_lgode": compatible,
        "mse_table_scaled": compatible * 100.0,
    }
