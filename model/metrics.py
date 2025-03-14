import logging
from typing import Any, Dict

import numpy as np
from numpy import floating
from numpy.typing import NDArray
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)  # Get logger for metrics module


def smape(actual: NDArray[np.float32], predicted: NDArray[np.float32]) -> floating[Any]:
    """Calculate SMAPE."""
    mask = ~((actual == 0) & (predicted == 0))
    actual = actual[mask]
    predicted = predicted[mask]
    denominator = np.abs(actual) + np.abs(predicted)
    denominator = np.where(denominator == 0, 1e-8, denominator)
    smape_val = 2.0 * np.mean(np.abs(predicted - actual) / denominator)
    return smape_val * 100


def mape(actual: NDArray[np.float32], predicted: NDArray[np.float32]) -> floating[Any]:
    """Calculate MAPE, ignoring near-zero actual values to prevent huge errors."""
    threshold = 1e-4
    mask = actual >= threshold  # Only consider values above threshold
    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]
    if len(actual_filtered) == 0:
        return np.float32(0.0)
    mape_val = np.mean(np.abs((actual_filtered - predicted_filtered) / actual_filtered))
    return mape_val * 100


def rmse(actual: NDArray[np.float32], predicted: NDArray[np.float32]) -> float:
    """Calculate RMSE."""
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mae(actual: NDArray[np.float32], predicted: NDArray[np.float32]) -> floating[Any]:
    """Calculate MAE using NumPy."""
    return np.mean(np.abs(actual - predicted))


def calculate_outlier_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.95
) -> Dict[str, float]:
    """Calculate metrics specific to outlier performance."""
    threshold = np.quantile(y_true, quantile)
    outlier_mask = y_true > threshold

    if not np.any(outlier_mask):
        return {
            "outlier_mse": np.nan,
            "outlier_mae": np.nan,
            "outlier_capture_rate": np.nan,
        }

    outlier_mse = np.mean((y_pred[outlier_mask] - y_true[outlier_mask]) ** 2)
    outlier_mae = np.mean(np.abs(y_pred[outlier_mask] - y_true[outlier_mask]))
    outlier_capture = np.mean((y_pred[outlier_mask] > threshold).astype(float))

    return {
        "outlier_mse": outlier_mse,
        "outlier_mae": outlier_mae,
        "outlier_capture_rate": outlier_capture,
    }


def calculate_distribution_stats(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate distribution statistics for true and predicted values."""
    return {
        "true_min": np.min(y_true),
        "true_max": np.max(y_true),
        "pred_min": np.min(y_pred),
        "pred_max": np.max(y_pred),
        "true_95th": np.quantile(y_true, 0.95),
        "pred_95th": np.quantile(y_pred, 0.95),
        "true_mean": np.mean(y_true),
        "pred_mean": np.mean(y_pred),
        "true_std": np.std(y_true),
        "pred_std": np.std(y_pred),
    }


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive metrics including standard, outlier, and distribution statistics.
    """
    # Flatten and clean data
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]

    # Get base metrics
    base_metrics = {
        "R2": r2_score(y_true_clean, y_pred_clean),
        "RMSE": rmse(y_true_clean, y_pred_clean),
        "MAE": mae(y_true_clean, y_pred_clean),
        "MAPE": mape(y_true_clean, y_pred_clean),
        "SMAPE": smape(y_true_clean, y_pred_clean),
    }

    # Add outlier metrics
    outlier_metrics = calculate_outlier_metrics(y_true_clean, y_pred_clean)
    distribution_stats = calculate_distribution_stats(y_true_clean, y_pred_clean)

    # Combine all metrics
    all_metrics = {**base_metrics, **outlier_metrics, **distribution_stats}

    return all_metrics
