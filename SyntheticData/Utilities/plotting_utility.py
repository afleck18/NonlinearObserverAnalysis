import numpy as np
import matplotlib.pyplot as plt
from Utilities.config import ConvergenceConfig,convergenceStyle
from Utilities.config import DuffingConfig
from Utilities.config import ContractionRMSEConfig,ContractionRMSEStyle
from Utilities.config import MonteCarloConfig,MonteCarloStyle

def plot_convergence_panels(
        cfg: ConvergenceConfig,
        style: convergenceStyle,
        simulation_results: dict
    ):
    """Figure 1: Mechanism / theory-facing figure"""
    
    fig1, axs = plt.subplots(2, 2, figsize=(10, 7))
    axs = axs.ravel()

    # ------------------------------------------------------------
    # (a) Sensitivity-dependent error trajectories
    # ------------------------------------------------------------
    for kappa in cfg.kappa_values:
        axs[0].semilogy(
            np.abs(simulation_results["fixed_runs"][kappa]["e"]) + 1e-12,
            label=fr"$\kappa={kappa}$",
            **style.STYLE_MAP_A[kappa]
        )

    axs[0].set_title("(a) Sensitivity-dependent error trajectories", fontsize=11)
    axs[0].set_ylabel(r"$\|e_t\|$")
    axs[0].set_xlabel("Time step")
    axs[0].legend(frameon=False)
    axs[0].grid(alpha=0.18)

    # ------------------------------------------------------------
    # (b) Fixed-gain contraction certificate
    # ------------------------------------------------------------
    tt = np.arange(cfg.t_plot_start, cfg.T)

    axs[1].plot(
        tt,
        simulation_results["low_fixed"]["gamma"][cfg.t_plot_start:],
        color=style.COL_FIXED,
        linewidth=style.LW_MAIN,
        label=fr"fixed, $\kappa={cfg.kappa_low}$"
    )
    axs[1].plot(
        tt,
        simulation_results["high_fixed"]["gamma"][cfg.t_plot_start:],
        color=style.COL_NORM,
        linewidth=style.LW_MAIN,
        label=fr"fixed, $\kappa={cfg.kappa_high}$"
    )
    axs[1].axhline(
        1.0,
        linestyle="--",
        color=style.COL_FIXED,
        linewidth=style.LW_REF,
        label=r"$\gamma_t=1$"
    )

    axs[1].set_title("(b) Fixed-gain contraction certificate", fontsize=11)
    axs[1].set_ylabel(r"$\gamma_t$")
    axs[1].set_xlabel("Time step")
    axs[1].legend(frameon=False)
    axs[1].grid(alpha=0.18)

    # ------------------------------------------------------------
    # (c) Normalized contraction certificate
    # ------------------------------------------------------------
    axs[2].plot(
        tt,
        simulation_results["high_fixed"]["gamma"][cfg.t_plot_start:],
        color=style.COL_FIXED,
        linewidth=style.LW_MAIN,
        label="fixed gain"
    )
    axs[2].plot(
        tt,
        simulation_results["high_norm"]["gamma"][cfg.t_plot_start:],
        color=style.COL_NORM,
        linewidth=style.LW_MAIN,
        label="normalized gain"
    )
    axs[2].axhline(
        1.0,
        linestyle="--",
        color=style.COL_FIXED,
        linewidth=style.LW_REF,
        label=r"$\gamma_t=1$"
    )

    axs[2].set_title("(c) Normalized contraction certificate", fontsize=11)
    axs[2].set_ylabel(r"$\gamma_t$")
    axs[2].set_xlabel("Time step")
    axs[2].legend(frameon=False)
    axs[2].grid(alpha=0.18)

    # ------------------------------------------------------------
    # (d) High-sensitivity error comparison
    # ------------------------------------------------------------
    axs[3].semilogy(
        np.abs(simulation_results["high_fixed"]["e"]) + 1e-12,
        color=style.COL_FIXED,
        linewidth=style.LW_MAIN,
        label="fixed gain"
    )
    axs[3].semilogy(
        np.abs(simulation_results["high_norm"]["e"]) + 1e-12,
        color=style.COL_NORM,
        linewidth=style.LW_MAIN,
        label="normalized gain"
    )

    # Theory-facing exponential reference envelope
    env = np.exp(-0.23 * np.arange(cfg.T))
    axs[3].semilogy(
        env,
        linestyle="--",
        color=style.COL_REF,
        linewidth=style.LW_REF,
        alpha=style.ALPHA_ENV,
        label="exp envelope"
    )

    axs[3].set_title("(d) High-sensitivity error comparison", fontsize=11)
    axs[3].set_ylabel(r"$\|e_t\|$")
    axs[3].set_xlabel("Time step")
    axs[3].legend(frameon=False)
    axs[3].grid(alpha=0.18)

    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if cfg.save_fig:
        plt.savefig(
            cfg.fig_path,
            bbox_inches="tight",
            pad_inches=0.02,
            format="pdf"
        )
    plt.show()

def plot_convergence_probability(
        style: convergenceStyle,
        monte_carlo_results:dict
    ):
    """Plot Duffing Figure"""

    fig, axs = plt.subplots(1,2, figsize=(8,3.2))

    # ------------------------------------------------------------
    # convergence probability
    # ------------------------------------------------------------

    axs[0].plot(
        monte_carlo_results["kappa_sweep"],
        monte_carlo_results["conv_fixed"],
        color=style.COL_FIXED,
        linewidth=1.4,
        label="fixed"
    )

    axs[0].plot(
        monte_carlo_results["kappa_sweep"],
        monte_carlo_results["conv_norm"],
        color=style.COL_NORM,
        linewidth=1.4,
        label="normalized"
    )

    axs[0].set_title("Convergence probability", fontsize=10)
    axs[0].set_xlabel(r"$\kappa$")
    axs[0].set_ylabel("Probability")
    axs[0].set_ylim(-0.02,1.02)
    axs[0].legend(frameon=False)
    axs[0].grid(alpha=0.12)


    # ------------------------------------------------------------
    # empirical contraction rate
    # ------------------------------------------------------------

    axs[1].plot(
        monte_carlo_results["kappa_sweep"],
        monte_carlo_results["rho_fixed"],
        color=style.COL_FIXED,
        linewidth=1.4,
        label="fixed"
    )

    axs[1].plot(
        monte_carlo_results["kappa_sweep"],
        monte_carlo_results["rho_norm"],
        color=style.COL_NORM,
        linewidth=1.4,
        label="normalized"
    )

    axs[1].axhline(
        1.0,
        linestyle="--",
        color="0.6",
        linewidth=1.0
    )

    axs[1].set_title("Empirical contraction rate", fontsize=10)
    axs[1].set_xlabel(r"$\kappa$")
    axs[1].set_ylabel(r"$\rho_{\mathrm{emp}}$")
    axs[1].grid(alpha=0.12)

    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

def smooth_signal(x: np.ndarray, window: int = 7) -> np.ndarray:
    """
    Simple moving-average smoothing.
    Used only for plotting the contraction proxy to reduce visual jitter.
    """

    if window <= 1:
        return x.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")

def build_duffing_figure(results: dict, cfg: DuffingConfig):
    """Plot Duffing Figure"""

    t = results["t"]
    x_true = results["x_true"]
    x_hat_fixed = results["x_hat_fixed"]
    x_hat_norm = results["x_hat_norm"]

    mu_fixed = results["mu_fixed"]
    mu_norm = results["mu_norm"]

    # Light smoothing for presentation only
    mu_fixed_s = smooth_signal(mu_fixed, cfg.mu_smoothing_window)
    mu_norm_s = smooth_signal(mu_norm, cfg.mu_smoothing_window)

    fig = plt.figure(figsize=(10, 4.4))

    # --------------------------------------------------------
    # (a) Phase portrait
    # --------------------------------------------------------
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x_true[:, 0], x_true[:, 1], linewidth=2.0, label="True")
    ax1.plot(x_hat_fixed[:, 0], x_hat_fixed[:, 1], linewidth=1.6, label="Fixed gain")
    ax1.plot(x_hat_norm[:, 0], x_hat_norm[:, 1], linewidth=1.6, label="Normalized")

    ax1.set_xlabel(r"$x_1$")
    ax1.set_ylabel(r"$x_2$")
    ax1.set_title("(a) Duffing phase portrait")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(True, alpha=0.25)

    # --------------------------------------------------------
    # (b) Continuous-time contraction-rate proxy
    # --------------------------------------------------------
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t, mu_fixed_s, linewidth=2.0, label="Fixed gain")
    ax2.plot(t, mu_norm_s, linewidth=2.0, label="Normalized")
    ax2.axhline(0.0, linestyle="--", linewidth=1.2, label=r"$\mu_t = 0$")

    ax2.set_xlabel("Time")
    ax2.set_ylabel(r"$\mu_t=\lambda_{\max}(\mathrm{sym}(A_t))$")
    ax2.set_title("(b) Continuous-time contraction-rate proxy")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    plt.show()

    if cfg.save_fig:
        fig.savefig(cfg.fig_path, dpi=300, bbox_inches="tight")

def plot_contraction_rmse(
        cfg: ContractionRMSEConfig,
        style: ContractionRMSEStyle,
        x_grid: dict,
        sens_display: dict,
        gamma_display:dict,
        S_crit_display:dict,
        t: dict,
        rmse_results: dict,
    ):
    """Plot contraction RMSE"""

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=False)

    # =========================
    # (a) Sensitivity + contraction certificate
    # =========================
    ax = axes[0]

    ax.plot(
        x_grid,
        sens_display,
        color=style.COLOR_FIXED,
        linewidth=style.LW_MAIN,
        label=r"$|J_h(x)|$"
    )

    ax.plot(
        x_grid,
        gamma_display,
        color=style.COLOR_HIGH,
        linewidth=style.LW_MAIN,
        label=r"$\gamma(x)$"
    )

    ax.axhline(
        S_crit_display,
        linestyle="--",
        linewidth=style.LW_REF,
        color=style.COLOR_FIXED,
        alpha=0.8
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=style.LW_REF,
        color="gray",
        alpha=0.8
    )

    ax.set_xlim(cfg.x_min, cfg.x_max)
    ax.set_ylabel("Value")
    ax.set_title("(a) Sensitivity and contraction certificate", loc="left", pad=4)
    ax.grid(True, alpha=0.12)

    ymin = min(np.min(sens_display), np.min(gamma_display))
    ymax = max(np.max(sens_display), np.max(gamma_display))
    yspan = ymax - ymin

    # Data-based placement chosen away from the curves
    x_scrit = 0.78
    ax.text(
        x_scrit,
        S_crit_display + 0.08 * yspan,
        r"$S_{\mathrm{crit}}$",
        fontsize=7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.2),
    )

    x_gamma = 1.12
    ax.text(
        x_gamma,
        1.0 - 0.11 * yspan,
        r"$\gamma(x)=1$",
        fontsize=7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.2),
    )

    ax.legend(frameon=False, loc="upper right")

    # =========================
    # (b) Observer variants
    # =========================
    ax = axes[1]

    ax.plot(
        t,
        rmse_results["rmse_fixed_low"],
        color=style.COLOR_FIXED,
        linewidth=style.LW_MAIN,
        label="Fixed, low sensitivity",
    )

    ax.plot(
        t,
        rmse_results["rmse_fixed_high"],
        color=style.COLOR_HIGH,
        linewidth=style.LW_MAIN,
        label="Fixed, high sensitivity",
    )

    ax.plot(
        t,
        rmse_results["rmse_norm_high"],
        color=style.COLOR_NORM,
        linewidth=style.LW_MAIN,
        label="Normalized, high sensitivity",
    )

    ax.set_xlim(cfg.t_start_panel_b, cfg.T)
    ax.set_ylim(*cfg.panel_b_ylim)
    ax.set_xlabel(r"Time step $t$")
    ax.set_ylabel("Rolling RMSE")
    ax.set_title("(b) Observer performance under sensitivity variation", loc="left", pad=4)
    ax.grid(True, alpha=0.12)
    ax.legend(frameon=False, loc="upper right")
    ax.legend(fontsize=6)

    # Save first, then show
    fig.tight_layout(pad=0.7)
    if cfg.save_fig:   
        fig.savefig(cfg.fig_path, bbox_inches="tight")
    plt.show(block=True)
    plt.close(fig)

def plot_monte_carlo(
        cfg: MonteCarloConfig,
        style: MonteCarloStyle,
        t: np.ndarray,
        envelope: np.ndarray,
        monte_carlo_results:dict
    ):
    """Plot Monte Carlo"""

    fig, ax = plt.subplots(figsize=(4.8, 3.2))

    # Theory envelope
    ax.semilogy(
        t,
        envelope,
        "--",
        color=style.COL_ENV,
        linewidth=style.LW_ENV,
        alpha=0.30,
        label="theory envelope"
    )

    # Fixed mean + band
    ax.semilogy(
        t,
        monte_carlo_results["fixed_mean"],
        color=style.COL_FIXED,
        linewidth=style.LW_MAIN,
        label="fixed gain"
    )
    ax.fill_between(
        t,
        monte_carlo_results["fixed_low"],
        monte_carlo_results["fixed_high"],
        color=style.COL_FIXED,
        alpha=style.ALPHA_BAND,
        linewidth=0
    )

    # Normalized mean + band
    ax.semilogy(
        t,
        monte_carlo_results["norm_mean"],
        color=style.COL_NORM,
        linewidth=style.LW_MAIN,
        label="normalized gain"
    )
    ax.fill_between(
        t,
        monte_carlo_results["norm_low"],
        monte_carlo_results["norm_high"],
        color=style.COL_NORM,
        alpha=style.ALPHA_BAND,
        linewidth=0
    )

    ax.set_title(r"Monte Carlo error trace ($\kappa = 1.8$)", fontsize=9)
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"$|e_t|$")

    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.grid(alpha=0.12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if cfg.save_fig:
        plt.savefig(
            cfg.fig_path,
            bbox_inches="tight",
            pad_inches=0.02,
            dpi=300
        )
    plt.show()