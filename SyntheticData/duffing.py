import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
import matplotlib

from Utilities.config import DuffingConfig
from Utilities.measurement_map import duffing_f,jacobian_f,h_phi,J_h
from Utilities.plotting_utility import build_duffing_figure

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

cfg = DuffingConfig()

def rk4_step(
        f, 
        x: np.ndarray, 
        t: float, 
        dt: float, 
        cfg: DuffingConfig
    ) -> np.ndarray:
    """One RK4 step for the true Duffing dynamics."""

    k1 = f(x, t, cfg)
    k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt, cfg)
    k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt, cfg)
    k4 = f(x + dt * k3, t + dt, cfg)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def observer_step(
        x_hat: np.ndarray,
        y: float,
        t: float,
        dt: float,
        cfg: DuffingConfig,
        mode: str,
    ) -> tuple:
    """
    One observer step with either:
      - mode='fixed'
      - mode='normalized'
    """

    y_hat = h_phi(x_hat[0], cfg.kappa)
    innovation = y - y_hat
    Jh = np.array([[J_h(x_hat[0], cfg.kappa), 0.0]], dtype=float)

    if mode == "fixed":
        alpha_t = cfg.alpha_fixed
    elif mode == "normalized":

        denom = norm(cfg.K_tilde @ Jh, 2) + cfg.eps
        alpha_t = cfg.beta_n / denom
    else:
        raise ValueError(f"Unknown observer mode: {mode}")

    correction = alpha_t * (cfg.K_tilde.flatten() * innovation)

    # Forward Euler observer update
    x_hat_next = x_hat + dt * duffing_f(x_hat, t, cfg) + dt * correction
    return x_hat_next, alpha_t

def compute_mu(
        x_hat: np.ndarray, 
        t: float, 
        alpha_t: float, 
        cfg: DuffingConfig
    ) -> float:
    """
    Continuous-time contraction-rate proxy:
        A_t ≈ J_f(x_hat,t) - alpha_t K_tilde J_h(x_hat)
        mu_t = lambda_max(sym(A_t))
    Negative values indicate local contraction.
    """
    Jf = jacobian_f(x_hat, t, cfg)
    Jh = np.array([[J_h(x_hat[0], cfg.kappa), 0.0]], dtype=float)
    A = Jf - alpha_t * (cfg.K_tilde @ Jh)
    symA = 0.5 * (A + A.T)
    eigvals = np.linalg.eigvalsh(symA)
    return float(np.max(eigvals))

def run_simulation(cfg: DuffingConfig) -> dict:
    """Run simulation"""

    t_grid = np.arange(0.0, cfg.T + cfg.dt, cfg.dt)
    n = len(t_grid)

    x_true = np.zeros((n, 2), dtype=float)
    x_hat_fixed = np.zeros((n, 2), dtype=float)
    x_hat_norm = np.zeros((n, 2), dtype=float)

    y_meas = np.zeros(n, dtype=float)

    err_fixed = np.zeros((n, 2), dtype=float)
    err_norm = np.zeros((n, 2), dtype=float)

    err_fixed_norm = np.zeros(n, dtype=float)
    err_norm_norm = np.zeros(n, dtype=float)

    mu_fixed = np.zeros(n, dtype=float)
    mu_norm = np.zeros(n, dtype=float)

    alpha_fixed_hist = np.zeros(n, dtype=float)
    alpha_norm_hist = np.zeros(n, dtype=float)

    x_true[0] = cfg.x0_true
    x_hat_fixed[0] = cfg.x0_hat_fixed
    x_hat_norm[0] = cfg.x0_hat_norm

    rng = np.random.default_rng(0)

    for k, t in enumerate(t_grid[:-1]):
        # Measurement
        y = h_phi(x_true[k][0], cfg.kappa)
        if cfg.meas_noise_std > 0.0:
            y += rng.normal(0.0, cfg.meas_noise_std)
        y_meas[k] = y

        # Observer updates
        x_hat_fixed[k + 1], alpha_fixed_hist[k] = observer_step(
            x_hat_fixed[k], y, t, cfg.dt, cfg, mode="fixed"
        )
        x_hat_norm[k + 1], alpha_norm_hist[k] = observer_step(
            x_hat_norm[k], y, t, cfg.dt, cfg, mode="normalized"
        )

        # True state update
        x_next = rk4_step(duffing_f, x_true[k], t, cfg.dt, cfg)
        if cfg.process_noise_std > 0.0:
            x_next += rng.normal(0.0, cfg.process_noise_std, size=2)
        x_true[k + 1] = x_next

        # Errors
        err_fixed[k] = x_true[k] - x_hat_fixed[k]
        err_norm[k] = x_true[k] - x_hat_norm[k]
        err_fixed_norm[k] = norm(err_fixed[k], 2)
        err_norm_norm[k] = norm(err_norm[k], 2)

        # Contraction-rate proxy
        mu_fixed[k] = compute_mu(x_hat_fixed[k], t, alpha_fixed_hist[k], cfg)
        mu_norm[k] = compute_mu(x_hat_norm[k], t, alpha_norm_hist[k], cfg)

    # Final sample bookkeeping
    y_meas[-1] = h_phi(x_true[-1][0], cfg.kappa)

    err_fixed[-1] = x_true[-1] - x_hat_fixed[-1]
    err_norm[-1] = x_true[-1] - x_hat_norm[-1]
    err_fixed_norm[-1] = norm(err_fixed[-1], 2)
    err_norm_norm[-1] = norm(err_norm[-1], 2)

    alpha_fixed_hist[-1] = cfg.alpha_fixed
    alpha_norm_hist[-1] = cfg.beta_n / (
        norm(cfg.K_tilde @ np.array([[J_h(x_hat_norm[-1][0], cfg.kappa), 0.0]], dtype=float), 2) + cfg.eps
    )

    mu_fixed[-1] = compute_mu(x_hat_fixed[-1], t_grid[-1], alpha_fixed_hist[-1], cfg)
    mu_norm[-1] = compute_mu(x_hat_norm[-1], t_grid[-1], alpha_norm_hist[-1], cfg)

    return {
        "t": t_grid,
        "x_true": x_true,
        "x_hat_fixed": x_hat_fixed,
        "x_hat_norm": x_hat_norm,
        "y_meas": y_meas,
        "err_fixed": err_fixed,
        "err_norm": err_norm,
        "err_fixed_norm": err_fixed_norm,
        "err_norm_norm": err_norm_norm,
        "mu_fixed": mu_fixed,
        "mu_norm": mu_norm,
        "alpha_fixed_hist": alpha_fixed_hist,
        "alpha_norm_hist": alpha_norm_hist,
    }

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results = run_simulation(cfg)
    build_duffing_figure(results, cfg)

    print(f"Saved PDF to: {cfg.fig_path}")
    print(f"Final fixed error norm: {results['err_fixed_norm'][-1]:.6f}")
    print(f"Final normalized error norm: {results['err_norm_norm'][-1]:.6f}")

    print(f"Max fixed mu_t: {np.max(results['mu_fixed']):.6f}")
    print(f"Max normalized mu_t: {np.max(results['mu_norm']):.6f}")
    print(f"Mean fixed mu_t: {np.mean(results['mu_fixed']):.6f}")
    print(f"Mean normalized mu_t: {np.mean(results['mu_norm']):.6f}")
    print(f"Fraction fixed mu_t < 0: {np.mean(results['mu_fixed'] < 0.0):.4f}")
    print(f"Fraction norm mu_t < 0:  {np.mean(results['mu_norm'] < 0.0):.4f}")