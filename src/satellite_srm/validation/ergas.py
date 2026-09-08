"""Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS) remote sensing metric."""
import numpy as np

def calculate_ergas(target: np.ndarray, prediction: np.ndarray, scale_ratio: float = 3.0) -> float:
    """
    ERGAS = 100 * (scale_ratio) * sqrt( (1/C) * sum( (RMSE_k / Mean_k)^2 ) )
    Lower value indicates higher reconstruction fidelity.
    """
    c = target.shape[0]
    sum_ratios = 0.0

    for b in range(c):
        t_b = target[b].astype(np.float64)
        p_b = prediction[b].astype(np.float64)

        rmse = np.sqrt(np.mean((t_b - p_b) ** 2))
        mean_t = np.mean(t_b)
        denom = mean_t if abs(mean_t) > 1e-6 else 1.0

        sum_ratios += (rmse / denom) ** 2

    ergas = 100.0 * (1.0 / scale_ratio) * np.sqrt(sum_ratios / c)
    return float(ergas)
