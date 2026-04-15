import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from Utilities.measurement_map import h_phi,J_h
from Utilities.config import ContractionRMSEConfig, ContractionRMSEStyle
from Utilities.plotting_utility import plot_contraction_rmse

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

cfg = ContractionRMSEConfig()
style = ContractionRMSEStyle()

def simulate_scalar_variant(
    kappa: float,
    cfg: ContractionRMSEConfig,
    gain_mode: str = "fixed"
) -> dict:
    
    """
    Simulate one observer variant on a forced scalar system.

    True system:
        x_{t+1} = a x_t + u_t + w_t
        y_t     = h_phi(x_t) + v_t

    Observer:
        xhat_{t+1} = a xhat_t + u_t + alpha_t * K * (y_t - h_phi(xhat_t))
    """

    rng = np.random.default_rng(cfg.seed)

    x = np.zeros(cfg.T + 1)
    xhat = np.zeros(cfg.T + 1)
    y = np.zeros(cfg.T)
    u_hist = np.zeros(cfg.T)
    alpha_hist = np.zeros(cfg.T)
    sens_hist = np.zeros(cfg.T)

    x[0] = cfg.x0
    xhat[0] = cfg.xhat0

    for t in range(cfg.T):
        # Keep the trajectory revisiting the sensitive region near x = 0
        u_t = cfg.forcing_amp * np.sin(2.0 * np.pi * t / cfg.forcing_period)
        u_hist[t] = u_t

        w_t = cfg.process_noise_std * rng.standard_normal()
        v_t = cfg.meas_noise_std * rng.standard_normal()

        # Current measurement and sensitivity
        y[t] = h_phi(x[t], kappa) + v_t
        sens_hist[t] = np.abs(J_h(x[t], kappa))

        # Gain selection
        if gain_mode == "fixed":
            alpha_t = cfg.alpha_fixed
        elif gain_mode == "normalized":
            alpha_nom = cfg.beta / (np.abs(cfg.K * J_h(xhat[t], kappa)) + cfg.eps)
            alpha_t = np.clip(alpha_nom, 0.15, cfg.alpha_fixed)
        else:
            raise ValueError("gain_mode must be 'fixed' or 'normalized'")

        alpha_hist[t] = alpha_t

        # Observer update
        innovation = y[t] - h_phi(xhat[t], kappa)
        xhat[t + 1] = cfg.Lf * xhat[t] + u_t + alpha_t * cfg.K * innovation

        # True state update
        x[t + 1] = cfg.Lf * x[t] + u_t + w_t

    err = x - xhat

    return {
        "x": x,
        "xhat": xhat,
        "y": y,
        "u_hist": u_hist,
        "err": err,
        "abs_err": np.abs(err),
        "alpha_hist": alpha_hist,
        "sens_hist": sens_hist,
    }

def simulation_comp(cfg:ContractionRMSEConfig) -> dict:

    """
    Simulate all observer variants on forced scalar systems.

    Returns compilation of simulations for high and low, fixed
    and normalized observers.
    """
    sim_fixed_low = simulate_scalar_variant(
        kappa=cfg.kappa_low,
        cfg=cfg,
        gain_mode="fixed"
    )

    sim_fixed_high = simulate_scalar_variant(
        kappa=cfg.kappa_high,
        cfg=cfg,
        gain_mode="fixed"
    )

    sim_norm_high = simulate_scalar_variant(
        kappa=cfg.kappa_high,
        cfg=cfg,
        gain_mode="normalized"
    )

    sim_results = {
        "sim_fixed_low":sim_fixed_low,
        "sim_fixed_high":sim_fixed_high,
        "sim_norm_high":sim_norm_high
    }

    return sim_results

def rolling_rmse(
        err: list, 
        window: int = 33, 
        start_index: int = 0
    ) -> np.ndarray:

    """
    Rolling RMSE over a trailing window.

    Values before start_index are set to NaN so the plotted curve
    does not inherit the startup transient.
    """

    err = np.asarray(err, dtype=float)
    out = np.full_like(err, np.nan, dtype=float)
    sq = err**2

    for i in range(max(start_index, 0), len(err)):
        start = max(start_index, i - window + 1)
        out[i] = np.sqrt(np.mean(sq[start:i + 1]))

    return out

def rmse_comp(
        cfg: ContractionRMSEConfig,
        sim_results: dict
    ) -> dict:

    """
    Rolling RMSE over all simulations.

    Returns compilation of RMSE arrays from simulations for high and low, fixed
    and normalized observers.
    """

    rmse_fixed_low = rolling_rmse(
        sim_results["sim_fixed_low"]["err"], window=cfg.rolling_window, start_index=cfg.t_start_panel_b
    )
    rmse_fixed_high = rolling_rmse(
        sim_results["sim_fixed_high"]["err"], window=cfg.rolling_window, start_index=cfg.t_start_panel_b
    )
    rmse_norm_high = rolling_rmse(
        sim_results["sim_norm_high"]["err"], window=cfg.rolling_window, start_index=cfg.t_start_panel_b
    )

    rmse_results = {
        "rmse_fixed_low": rmse_fixed_low, 
        "rmse_fixed_high": rmse_fixed_high, 
        "rmse_norm_high": rmse_norm_high
    }

    return rmse_results

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Panel (a): display-only theoretical curves
    # --------------------------------------------------------
    x_grid = np.linspace(cfg.x_min, cfg.x_max, cfg.num_points)
    sens_display = np.abs(J_h(x_grid, cfg.kappa_display))
    gamma_display = cfg.Lf_display + cfg.alpha_display * np.abs(cfg.K_display) * sens_display
    S_crit_display = (1.0 - cfg.Lf_display) / (cfg.alpha_display * np.abs(cfg.K_display))

    # --------------------------------------------------------
    # Panel (b): simulation curves
    # --------------------------------------------------------
    sim_results = simulation_comp(cfg)
    t = np.arange(cfg.T + 1)
    rmse_results = rmse_comp(cfg,sim_results)

    plot_contraction_rmse(
        cfg,
        style,
        x_grid,
        sens_display,
        gamma_display,
        S_crit_display,
        t,
        rmse_results
    )


    # --------------------------------------------------------
    # Summary metrics
    # --------------------------------------------------------
    full_rmse_fixed_low = float(np.sqrt(np.mean(sim_results["sim_fixed_low"]["err"]**2)))
    full_rmse_fixed_high = float(np.sqrt(np.mean(sim_results["sim_fixed_high"]["err"]**2)))
    full_rmse_norm_high = float(np.sqrt(np.mean(sim_results["sim_norm_high"]["err"]**2)))

    full_improvement_pct = 100.0 * (
        full_rmse_fixed_high - full_rmse_norm_high
    ) / full_rmse_fixed_high

    post_start = cfg.t_start_panel_b
    post_rmse_fixed_low = float(np.sqrt(np.mean(sim_results["sim_fixed_low"]["err"][post_start:]**2)))
    post_rmse_fixed_high = float(np.sqrt(np.mean(sim_results["sim_fixed_high"]["err"][post_start:]**2)))
    post_rmse_norm_high = float(np.sqrt(np.mean(sim_results["sim_norm_high"]["err"][post_start:]**2)))

    post_improvement_pct = 100.0 * (
        post_rmse_fixed_high - post_rmse_norm_high
    ) / post_rmse_fixed_high

    print(f"Saved PDF to: {cfg.fig_path}")

    print("\nRMSE summary:")
    print(f"  Full RMSE | Fixed, low sensitivity:        {full_rmse_fixed_low:.4f}")
    print(f"  Full RMSE | Fixed, high sensitivity:       {full_rmse_fixed_high:.4f}")
    print(f"  Full RMSE | Normalized, high sensitivity:  {full_rmse_norm_high:.4f}")
    print(f"  Full-horizon improvement:                  {full_improvement_pct:.2f}%")

    print(f"  Post-transient RMSE | Fixed, low sensitivity:        {post_rmse_fixed_low:.4f}")
    print(f"  Post-transient RMSE | Fixed, high sensitivity:       {post_rmse_fixed_high:.4f}")
    print(f"  Post-transient RMSE | Normalized, high sensitivity:  {post_rmse_norm_high:.4f}")
    print(f"  Post-transient improvement:                         {post_improvement_pct:.2f}%")

    print("\nDiagnostics:")
    print(f"  alpha_fixed = {cfg.alpha_fixed:.4f}")
    print(
        f"  normalized alpha range = "
        f"[{np.min(sim_results['sim_norm_high']['alpha_hist']):.4f}, {np.max(sim_results['sim_norm_high']['alpha_hist']):.4f}]"
    )
    print(
        f"  high-kappa sensitivity range = "
        f"[{np.min(sim_results['sim_fixed_high']['sens_hist']):.4f}, {np.max(sim_results['sim_fixed_high']['sens_hist']):.4f}]"
    )