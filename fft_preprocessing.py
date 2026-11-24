"""Non-DC FFT amplitude spectrum and training-only standardization."""
import torch


def build_fft_features(x):
    """Extract the non-DC FFT amplitude spectrum for each window."""
    spectrum = torch.fft.rfft(x.float(), dim=1).abs()[:, 1:]
    return spectrum, f"FFT amplitude spectrum {spectrum.shape[1]} dimensions"

def fit_feature_standardizer(x_train, std_floor=1e-6):
    """Fit global feature statistics from training nodes only."""
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False).clamp_min(std_floor)
    return mean, std

def apply_feature_standardizer(x, stats):
    mean, std = stats
    return (x - mean) / std
