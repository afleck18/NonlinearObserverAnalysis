# ============================================================
# Learned measurement map and Jacobian
# ============================================================

import numpy as np
from Utilities.config import DuffingConfig


def f(x: float, a: float) -> float:
    return a * x

def h_phi(x: float, kappa: float) -> float:
    """
    Learned measurement map:
        h_phi(x) = x + kappa * x * exp(-x^2)
    """
    return x + kappa * x * np.exp(-x**2)

def J_h(x: float, kappa: float) -> float:
    """
    Jacobian of h_phi:
        J_h(x) = 1 + kappa * exp(-x^2) * (1 - 2x^2)
    """
    return 1.0 + kappa * np.exp(-x**2) * (1.0 - 2.0 * x**2)


def duffing_f(x: np.ndarray, t: float, cfg: DuffingConfig) -> np.ndarray:
    """
    Duffing dynamics and Jacobians

    Forced Duffing dynamics:
        x1_dot = x2
        x2_dot = -delta*x2 - alpha*x1 - beta*x1^3 + gamma*cos(omega*t)
    """
    x1, x2 = x
    dx1 = x2
    dx2 = (
        -cfg.delta * x2
        - cfg.alpha_duff * x1
        - cfg.beta_duff * (x1 ** 3)
        + cfg.gamma * np.cos(cfg.omega * t)
    )
    return np.array([dx1, dx2], dtype=float)


def jacobian_f(x: np.ndarray, t: float, cfg: DuffingConfig) -> np.ndarray:
    """Jacobian of f with respect to x."""
    
    x1, _ = x
    return np.array(
        [
            [0.0, 1.0],
            [-cfg.alpha_duff - 3.0 * cfg.beta_duff * (x1 ** 2), -cfg.delta],
        ],
        dtype=float,
    )

