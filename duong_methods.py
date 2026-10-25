"""Auditable evaluation and saliency helpers for the improved notebook."""
import numpy as np
from scipy.ndimage import gaussian_filter, convolve

def tolerance_accuracy(y_true, y_pred, threshold=0.7):
    """Fraction with absolute error strictly below threshold (boundary excluded)."""
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true)) < threshold))

def parameter_label(features):
    """All-zero design is the intercept-only baseline, irrespective of column count."""
    return f"{0 if np.all(np.asarray(features) == 0) else np.asarray(features).shape[1]}+1"

def format_correlation(value):
    return "N/A" if not np.isfinite(value) else f"{value:.3f}"

def saliency_pair(gradient):
    """A follows released code; B is one explicit interpretation of paper prose.

    B smooths the signed gradient, with a normalized discrete 5x5 Gaussian
    (sigma=1 pixel, offsets -2..2). B uses half-sample symmetric reflection.
    Sigma, sign handling and boundaries for B are assumptions, not specified
    by the paper. A matches skimage.filters.gaussian's default nearest boundary.
    """
    g = np.asarray(gradient)
    a = gaussian_filter(g*g, sigma=(5,5), truncate=3.5, mode="nearest")
    axis = np.arange(-2,3, dtype=float)
    kernel = np.exp(-(axis[:,None]**2 + axis[None,:]**2)/2)
    kernel /= kernel.sum()
    b = convolve(g, kernel, mode="reflect")
    return a,b
