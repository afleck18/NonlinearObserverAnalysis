import numpy as np
from typing import List

from src.config import Config, TrackingRecord, ObserverOutputs
from src.utils.utils import stable_norm

# ============================================================
# Observers
# ============================================================

class FixedGainObserver:
    def __init__(self, cfg: Config):
        self.F = cfg.F.copy()
        self.H = cfg.H.copy()
        self.K_tilde = cfg.K_tilde.copy()
        self.alpha = float(cfg.alpha_fixed)
        self.xhat = None

    def initialize(self, y0):
        self.xhat = np.array([y0[0], y0[1], 0.0, 0.0], dtype=float)

    def step(self, y):
        if self.xhat is None:
            self.initialize(y)
        innovation = y - self.H @ self.xhat
        self.xhat = self.F @ self.xhat + self.alpha * (self.K_tilde @ innovation)
        return self.xhat.copy()

    def reset(self):
        self.xhat = None

    def certificate_proxy(self, sensitivity_proxy):
        F_norm = stable_norm(self.F)
        K_norm = stable_norm(self.K_tilde)
        return F_norm + self.alpha * K_norm * sensitivity_proxy

class NormalizedObserver:
    def __init__(self, cfg: Config):
        self.F = cfg.F.copy()
        self.H = cfg.H.copy()
        self.K_tilde = cfg.K_tilde.copy()
        self.beta = float(cfg.beta)
        self.epsilon = float(cfg.epsilon)
        self.xhat = None

    def initialize(self, y0):
        self.xhat = np.array([y0[0], y0[1], 0.0, 0.0], dtype=float)

    def alpha_t(self, sensitivity_proxy):
        K_norm = stable_norm(self.K_tilde)
        return self.beta / (K_norm * sensitivity_proxy + self.epsilon)

    def step(self, y, sensitivity_proxy):
        if self.xhat is None:
            self.initialize(y)
        innovation = y - self.H @ self.xhat
        alpha = self.alpha_t(sensitivity_proxy)
        self.xhat = self.F @ self.xhat + alpha * (self.K_tilde @ innovation)
        return self.xhat.copy()

    def reset(self):
        self.xhat = None

    def certificate_proxy(self, sensitivity_proxy):
        F_norm = stable_norm(self.F)
        K_norm = stable_norm(self.K_tilde)
        alpha = self.alpha_t(sensitivity_proxy)
        return F_norm + alpha * K_norm * sensitivity_proxy

def run_observers(cfg: Config, records: List[TrackingRecord]) -> ObserverOutputs:
    fixed_obs = FixedGainObserver(cfg)
    norm_obs = NormalizedObserver(cfg)

    fixed_centers = []
    norm_centers = []
    fixed_errors = []
    norm_errors = []
    fixed_gamma_proxy = []
    norm_gamma_proxy = []
    sensitivity_proxy_list = []
    valid_frame_indices = []

    missing_count = 0
    initialized = False

    for rec in records:
        if rec.tracker_center is None or rec.gt_center is None:
            missing_count += 1
            if missing_count > cfg.max_missing_frames_before_reset:
                fixed_obs.reset()
                norm_obs.reset()
                initialized = False
            continue

        missing_count = 0
        s_t = rec.sensitivity_proxy if rec.sensitivity_proxy is not None else 1.0

        if not initialized:
            fixed_obs.initialize(rec.tracker_center)
            norm_obs.initialize(rec.tracker_center)
            initialized = True

        xhat_fix = fixed_obs.step(rec.tracker_center)
        xhat_norm = norm_obs.step(rec.tracker_center, s_t)

        p_fix = xhat_fix[:2].copy()
        p_norm = xhat_norm[:2].copy()

        fixed_centers.append(p_fix)
        norm_centers.append(p_norm)
        fixed_errors.append(stable_norm(p_fix - rec.gt_center))
        norm_errors.append(stable_norm(p_norm - rec.gt_center))
        fixed_gamma_proxy.append(fixed_obs.certificate_proxy(s_t))
        norm_gamma_proxy.append(norm_obs.certificate_proxy(s_t))
        sensitivity_proxy_list.append(float(s_t))
        valid_frame_indices.append(rec.frame_idx)

    return ObserverOutputs(
        fixed_centers=fixed_centers,
        norm_centers=norm_centers,
        fixed_errors=fixed_errors,
        norm_errors=norm_errors,
        fixed_gamma_proxy=fixed_gamma_proxy,
        norm_gamma_proxy=norm_gamma_proxy,
        sensitivity_proxy=sensitivity_proxy_list,
        valid_frame_indices=valid_frame_indices,
    )
