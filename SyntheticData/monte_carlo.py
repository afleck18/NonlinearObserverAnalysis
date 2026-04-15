import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from Utilities.measurement_map import f, h_phi, J_h
from Utilities.config import MonteCarloConfig, MonteCarloStyle
from Utilities.plotting_utility import plot_monte_carlo

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

cfg = MonteCarloConfig()
style = MonteCarloStyle()
rng = np.random.default_rng(7)

def simulate_trial(
        cfg: MonteCarloConfig, 
        normalized: bool
    ) -> np.ndarray:

    """
    Single Monte Carlo run
    """

    x = np.zeros(cfg.T + 1)
    xhat = np.zeros(cfg.T + 1)

    # Randomized initial conditions
    x[0] = 1.5 + rng.random()
    xhat[0] = -0.5 + rng.random()

    for t in range(cfg.T):
        w_t = cfg.sigma_w * rng.normal()
        v_t = cfg.sigma_v * rng.normal()

        x[t + 1] = f(x[t],cfg.a) + w_t
        y_t = h_phi(x[t], cfg.kappa_rep) + v_t

        if normalized:
            alpha_t = cfg.beta / (abs(cfg.K_tilde * J_h(xhat[t], cfg.kappa_rep)) + cfg.eps)
        else:
            alpha_t = cfg.alpha_fixed

        innovation = y_t - h_phi(xhat[t], cfg.kappa_rep)
        xhat[t + 1] = f(xhat[t],cfg.a) + alpha_t * cfg.K_tilde * innovation

    return np.abs(x - xhat)

def empirical_contraction_rate(error:np.ndarray) -> float:

    """
    Empirical contraction rate
    Overall long-horizon contraction measure
    """
    
    e0 = np.linalg.norm(error[0])
    eT = np.linalg.norm(error[-1])

    if e0 < 1e-8 or eT < 1e-12:
        return np.nan

    horizon = len(error) - 1
    
    return (eT / e0) ** (1.0 / horizon)

def monte_carlo_runs(cfg: MonteCarloConfig) -> dict:

    """
    Monte Carlo runs
    """

    errors_fixed = []
    errors_norm = []

    for _ in range(cfg.N):
        errors_fixed.append(simulate_trial(cfg, normalized=False))
        errors_norm.append(simulate_trial(cfg, normalized=True))

    errors_fixed = np.array(errors_fixed)
    errors_norm = np.array(errors_norm)

    fixed_mean = np.mean(errors_fixed, axis=0)
    norm_mean = np.mean(errors_norm, axis=0)

    fixed_std = np.std(errors_fixed, axis=0)
    norm_std = np.std(errors_norm, axis=0)

    # Lower / upper confidence-style bands
    fixed_low = np.maximum(fixed_mean - fixed_std, 1e-12)
    fixed_high = fixed_mean + fixed_std

    norm_low = np.maximum(norm_mean - norm_std, 1e-12)
    norm_high = norm_mean + norm_std

    monte_carlo_results = {
        "fixed_mean": fixed_mean,
        "fixed_low": fixed_low,
        "fixed_high": fixed_high,
        "norm_mean": norm_mean,
        "norm_low": norm_low,
        "norm_high": norm_high,
        "errors_fixed": errors_fixed,
        "errors_norm": errors_norm
    }

    return monte_carlo_results

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # ------------------------------------------------------------
    # Monte Carlo Calculation and Plotting
    # ------------------------------------------------------------
    monte_carlo_results = monte_carlo_runs(cfg)
    t = np.arange(cfg.T + 1)
    envelope = np.exp(-0.23 * t)

    plot_monte_carlo(
        cfg,
        style,
        t,
        envelope,
        monte_carlo_results
    )

    # ------------------------------------------------------------
    # Table metrics for LaTeX insertion
    # ------------------------------------------------------------

    tail = 15

    # Final RMS
    fixed_final_rms = float(np.sqrt(np.mean(monte_carlo_results["errors_fixed"][:, -tail:]**2)))
    norm_final_rms = float(np.sqrt(np.mean(monte_carlo_results["errors_norm"][:, -tail:]**2)))

    # Peak error
    fixed_peak = float(np.max(monte_carlo_results["errors_fixed"]))
    norm_peak = float(np.max(monte_carlo_results["errors_norm"]))

    # Steady-state mean
    fixed_ss_mean = float(np.mean(monte_carlo_results["errors_fixed"][:, -tail:]))
    norm_ss_mean = float(np.mean(monte_carlo_results["errors_norm"][:, -tail:]))

    # Empirical contraction rates
    fixed_rho_trials = np.array([
        empirical_contraction_rate(err) for err in monte_carlo_results["errors_fixed"]
    ], dtype=float)

    norm_rho_trials = np.array([
        empirical_contraction_rate(err) for err in monte_carlo_results["errors_norm"]
    ], dtype=float)

    fixed_rho_emp = float(np.nanmean(fixed_rho_trials))
    norm_rho_emp = float(np.nanmean(norm_rho_trials))

    fixed_rho_std = float(np.nanstd(fixed_rho_trials))
    norm_rho_std = float(np.nanstd(norm_rho_trials))

    print(f"Saved PDF to: {cfg.fig_path}")

    print("Fixed final RMS:", fixed_final_rms)
    print("Normalized final RMS:", norm_final_rms)
    print("Fixed peak error:", fixed_peak)
    print("Normalized peak error:", norm_peak)
    print("Fixed steady-state mean:", fixed_ss_mean)
    print("Normalized steady-state mean:", norm_ss_mean)
    print("Fixed empirical contraction rate:", fixed_rho_emp)
    print("Normalized empirical contraction rate:", norm_rho_emp)
    print("Fixed empirical contraction rate std:", fixed_rho_std)
    print("Normalized empirical contraction rate std:", norm_rho_std)