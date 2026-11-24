"""Parameter-free adaptive average pooling, applied AFTER standardization."""
import torch.nn.functional as F


def apply_adaptive_average_pooling(x, output_dim=64):
    """Reduce each standardized FFT spectrum to a fixed dimension."""
    if x.ndim != 2:
        raise ValueError("pooling input must be a 2-D feature matrix")
    if not 1 <= output_dim <= x.size(1):
        raise ValueError(
            f"pooling output dimension must be in [1, {x.size(1)}], got {output_dim}"
        )
    return F.adaptive_avg_pool1d(
        x.unsqueeze(1), output_size=output_dim
    ).squeeze(1)
