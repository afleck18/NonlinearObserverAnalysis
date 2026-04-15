import numpy as np

# ============================================================
# Convergence Parameters
# ============================================================
class ConvergenceConfig:
    T = 60

    # System / observer parameters
    a = 0.8
    K_tilde = 1.0

    alpha_fixed = 0.80
    beta = 0.30
    eps = 1e-3

    # Nominal initial condition for main figure
    x0 = 1.5
    xhat0 = 0.0

    # Sensitivity levels for main mechanism figure
    kappa_values = [0.2, 0.6, 1.0, 1.4]
    kappa_low = 0.2
    kappa_high = 1.4

    # Trim early transient for contraction-certificate panels
    t_plot_start = 2

    # Smooth the contraction certificate slightly for cleaner CDC/TAC-style curves
    smooth = 0.85

    # Plot saving
    save_fig: bool = True
    fig_path: str = "SyntheticData/Plots/convergence_panels.pdf"

class convergenceStyle:
    # ============================================================
    # Plot style
    # ============================================================

    LW_MAIN = 1.35
    LW_REF = 1.0
    ALPHA_ENV = 0.50
    MS = 3.5

    # Color-blind-friendly restrained palette (Okabe-Ito inspired)
    STYLE_MAP_A = {
        0.2: dict(color="#0072B2", linestyle="-",  linewidth=LW_MAIN),   # blue
        0.6: dict(color="#E69F00", linestyle="--", linewidth=LW_MAIN),   # orange
        1.0: dict(color="#009E73", linestyle="-.", linewidth=LW_MAIN),   # green
        1.4: dict(color="#8c2d2d", linestyle="-",  linewidth=LW_MAIN),   # muted dark red
    }

    COL_FIXED = "#222222"   # near-black
    COL_NORM = "#56B3E8"    # teal
    #COL_REF = "0.65"        # soft gray
    COL_REF = "#FF27A5"

# ============================================================
# Duffing Configuration
# ============================================================
class DuffingConfig:
    # Time
    dt: float = 0.01
    T: float = 40.0

    # Duffing parameters
    delta: float = 0.3
    alpha_duff: float = -1.0
    beta_duff: float = 1.0
    gamma: float = 0.30
    omega: float = 1.2

    # Measurement geometry parameter
    kappa: float = 2.0

    # Observer gains
    alpha_fixed: float = 0.35
    beta_n: float = 0.8
    eps: float = 1e-3

    # Observer gain vector Ktilde in R^{2 x 1}
    K_tilde = np.array(
        [
            [1.0],
            [0.6],
        ],
        dtype=float,
    )

    # Noise
    process_noise_std: float = 0.0
    meas_noise_std: float = 0.0

    # Initial conditions
    x0_true = np.array([1.2, -0.4], dtype=float)
    x0_hat_fixed = np.array([-1.0, 0.8], dtype=float)
    x0_hat_norm = np.array([-1.0, 0.8], dtype=float)

    # Plot smoothing (presentation only)
    mu_smoothing_window: int = 7

    # Plot saving
    save_fig: bool = True
    fig_path: str = "SyntheticData/Plots/fig_duffing_validation.pdf"

# ============================================================
# Contraction RMSE Configuration
# ============================================================
class ContractionRMSEConfig:

    # Panel (a): display-only parameters
    x_min=-2.5
    x_max=2.5
    num_points=1400
    kappa_display=1.2
    Lf_display=0.80
    alpha_display=0.18
    K_display=1.0

    # Panel (b): simulation parameters
    kappa_high=2.7
    kappa_low=0.35
    Lf=0.985
    alpha_fixed=0.62
    beta=0.18
    eps=1e-3
    K=1.0
    T=220
    x0=0.05
    xhat0=-1.4
    forcing_amp=0.18
    forcing_period=18
    process_noise_std=0.01
    meas_noise_std=0.08
    rolling_window=33
    t_start_panel_b=20
    panel_b_ylim=(0.0, 0.16)
    seed=7

    # Plot saving
    save_fig: bool = True
    fig_path: str = "SyntheticData/Plots/fig_synthetic_geometry_contraction_rmse.pdf"

class ContractionRMSEStyle:
    # CDC-style consistent palette (match Monte Carlo + convergence)
    COLOR_FIXED = "#4C78A8"        # blue
    COLOR_HIGH = "#F58518"         # orange
    COLOR_NORM = "#54A24B"         # green
    COLOR_TRUE = "black"

    LW_MAIN = 1
    LW_REF = 0.7

# ============================================================
# Monte Carlo Configuration
# ============================================================
class MonteCarloConfig:
    T = 60
    N = 200

    sigma_w = 0.01
    sigma_v = 0.01

    # Increased sensitivity for stronger geometry effect
    kappa_rep = 1.8

    a = 0.8
    K_tilde = 1.0

    # Increased fixed gain so the fixed observer degrades more clearly
    alpha_fixed = 0.95
    beta = 0.30
    eps = 1e-3

    # Plot saving
    save_fig: bool = True
    fig_path: str = "SyntheticData/Plots/monte_carlo_trace.pdf"

class MonteCarloStyle:
    LW_MAIN = 1.6
    LW_ENV = 0.9
    ALPHA_BAND = 0.10

    COL_FIXED = "0.15"      # dark gray / near-black
    COL_NORM = "#8c2d2d"    # muted dark red
    COL_ENV = "#FF27A5"         # light pink
