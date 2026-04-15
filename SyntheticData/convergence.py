import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from Utilities.config import ConvergenceConfig,convergenceStyle
from Utilities.measurement_map import f, h_phi, J_h
from Utilities.plotting_utility import plot_convergence_panels,plot_convergence_probability

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

cfg = ConvergenceConfig()
rng = np.random.default_rng(7)

def simulate_observer(
    kappa: float,
    cfg: ConvergenceConfig,
    normalized: bool = False,
    x0_: float | None = None,
    xhat0_: float | None = None
) -> dict:
    """Simulation"""

    if x0_ is None:
        x0_ = cfg.x0
    if xhat0_ is None:
        xhat0_ = cfg.xhat0

    x = np.zeros(cfg.T + 1)
    xhat = np.zeros(cfg.T + 1)
    e = np.zeros(cfg.T + 1)
    gamma = np.zeros(cfg.T)

    x[0] = x0_
    xhat[0] = xhat0_
    e[0] = x0_ - xhat0_

    for t in range(cfg.T):
        # True system
        x[t + 1] = f(x[t],cfg.a)

        # Measurement
        y = h_phi(x[t], kappa)

        # Gain
        if normalized:
            alpha_t = cfg.beta / (abs(cfg.K_tilde * J_h(xhat[t], kappa)) + cfg.eps)
        else:
            alpha_t = cfg.alpha_fixed

        # Observer update
        innovation = y - h_phi(xhat[t], kappa)
        xhat[t + 1] = f(xhat[t],cfg.a) + alpha_t * cfg.K_tilde * innovation

        # Error
        e[t + 1] = x[t + 1] - xhat[t + 1]

        # Local contraction certificate
        x_mid = 0.5 * (x[t] + xhat[t])
        A_t = cfg.a - alpha_t * cfg.K_tilde * J_h(x_mid, kappa)
        gamma_raw = abs(A_t)

        # Mild smoothing for cleaner theory-facing display
        if t == 0:
            gamma[t] = gamma_raw
        else:
            gamma[t] = cfg.smooth * gamma[t - 1] + (1.0 - cfg.smooth) * gamma_raw

    return {
        "x": x,
        "xhat": xhat,
        "e": e,
        "gamma": gamma,
    }

def simulation_comp (cfg:ConvergenceConfig) -> dict:
    """Nominal simulations for Figure 1"""

    fixed_runs = {k: simulate_observer(k,cfg, normalized=False) for k in cfg.kappa_values}
    low_fixed = simulate_observer(cfg.kappa_low,cfg, normalized=False)
    high_fixed = simulate_observer(cfg.kappa_high,cfg, normalized=False)
    high_norm = simulate_observer(cfg.kappa_high,cfg, normalized=True)

    simulation_results = {
        "fixed_runs": fixed_runs,
        "low_fixed": low_fixed,
        "high_fixed": high_fixed,
        "high_norm": high_norm
    }

    return simulation_results

def monte_carlo_validation(
        cfg: ConvergenceConfig, 
        rng: np.random._generator.Generator
    ) -> dict:
    """
    Monte Carlo validation (noise + randomized ICs)
    """

    sigma_w = 0.01
    sigma_v = 0.01

    N = 200
    epsilon_conv = 1e-3

    kappa_sweep = np.linspace(0.1, 1.5, 12)

    conv_fixed = []
    conv_norm = []

    rho_fixed = []
    rho_norm = []

    for kappa in kappa_sweep:

        conv_f = []
        conv_n = []

        rho_f = []
        rho_n = []

        for _ in range(N):

            x = np.zeros(cfg.T+1)
            xhat_f = np.zeros(cfg.T+1)
            xhat_n = np.zeros(cfg.T+1)

            x[0] = 1.2 + 0.6 * rng.random()
            xhat_f[0] = -0.5 + rng.random()
            xhat_n[0] = xhat_f[0]

            e_f = []
            e_n = []

            for t in range(cfg.T):

                # process noise
                w = sigma_w * rng.normal()

                x[t+1] = f(x[t],cfg.a) + w

                # measurement noise
                v = sigma_v * rng.normal()
                y = h_phi(x[t], kappa) + v

                # fixed observer
                alpha_f = cfg.alpha_fixed
                xhat_f[t+1] = f(xhat_f[t],cfg.a) + alpha_f * cfg.K_tilde * (
                    y - h_phi(xhat_f[t], kappa)
                )

                # normalized observer
                alpha_n = cfg.beta / (
                    abs(cfg.K_tilde * J_h(xhat_n[t], kappa)) + cfg.eps
                )

                xhat_n[t+1] = f(xhat_n[t],cfg.a) + alpha_n * cfg.K_tilde * (
                    y - h_phi(xhat_n[t], kappa)
                )

                e_f.append(abs(x[t+1] - xhat_f[t+1]) + 1e-12)
                e_n.append(abs(x[t+1] - xhat_n[t+1]) + 1e-12)

            e_f = np.array(e_f)
            e_n = np.array(e_n)

            # convergence
            conv_f.append(e_f[-1] < epsilon_conv)
            conv_n.append(e_n[-1] < epsilon_conv)

            # empirical contraction rate
            rho_f.append(np.exp(np.mean(np.log(e_f[1:] / e_f[:-1]))))
            rho_n.append(np.exp(np.mean(np.log(e_n[1:] / e_n[:-1]))))

        conv_fixed.append(np.mean(conv_f))
        conv_norm.append(np.mean(conv_n))

        rho_fixed.append(np.mean(rho_f))
        rho_norm.append(np.mean(rho_n))

    conv_fixed = np.array(conv_fixed)
    conv_norm = np.array(conv_norm)

    rho_fixed = np.array(rho_fixed)
    rho_norm = np.array(rho_norm)

    results = {
        "kappa_sweep": kappa_sweep,
        "conv_fixed": conv_fixed,
        "conv_norm": conv_norm,
        "rho_fixed": rho_fixed,
        "rho_norm": rho_norm
    }

    return results

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    simulation_results = simulation_comp(cfg)
    plot_convergence_panels(
        cfg,
        convergenceStyle,
        simulation_results
    )
    
    monte_carlo_results = monte_carlo_validation (cfg, rng)
    plot_convergence_probability(convergenceStyle, monte_carlo_results)

    print(f"Saved PDF to: {cfg.fig_path}")
