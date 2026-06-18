import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from scipy.ndimage import gaussian_filter1d

try:
    from scipy.stats import wilcoxon
    SCIPY_WILCOXON_AVAILABLE = True
except Exception:
    SCIPY_WILCOXON_AVAILABLE = False

from src.config import Config,TrackingRecord,ObserverOutputs

# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def get_video_output_dir(cfg: Config, video_name: str) -> str:
    out_dir = os.path.join(cfg.output_dir, video_name)
    ensure_dir(out_dir)
    return out_dir

def stable_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))

def rmse(errors: np.ndarray) -> float:
    errors = np.asarray(errors, dtype=float)
    if errors.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(errors ** 2)))

def percentage_improvement(fixed_rmse: float, norm_rmse: float) -> float:
    if not np.isfinite(fixed_rmse) or fixed_rmse <= 0:
        return float("nan")
    return 100.0 * (fixed_rmse - norm_rmse) / fixed_rmse

def rolling_rmse(errors: np.ndarray, window: int = 25) -> np.ndarray:
    errors = np.asarray(errors, dtype=float)
    if errors.size == 0:
        return np.array([], dtype=float)
    out = np.zeros_like(errors)
    for i in range(len(errors)):
        start = max(0, i - window + 1)
        out[i] = np.sqrt(np.mean(errors[start:i + 1] ** 2))
    return out

def xywh_to_xyxy(box_xywh: np.ndarray) -> Tuple[float, float, float, float]:
    cx, cy, w, h = box_xywh
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2

def iou_xywh(box1: np.ndarray, box2: np.ndarray) -> float:
    x11, y11, x12, y12 = xywh_to_xyxy(box1)
    x21, y21, x22, y22 = xywh_to_xyxy(box2)

    ix1 = max(x11, x21)
    iy1 = max(y11, y21)
    ix2 = min(x12, x22)
    iy2 = min(y12, y22)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area1 = max(0.0, x12 - x11) * max(0.0, y12 - y11)
    area2 = max(0.0, x22 - x21) * max(0.0, y22 - y21)
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union

def color_for_track_id(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(abs(track_id) % 100000)
    vals = rng.integers(50, 255, size=3)
    return int(vals[0]), int(vals[1]), int(vals[2])

def clip_box_to_image(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    return x1, y1, x2, y2

def bootstrap_ci(
    values: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_boot: int = 10000,
    ci: float = 95.0,
    random_seed: int = 0,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = values.size
    if n == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(random_seed)
    boot_stats = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[b] = statistic_fn(sample)

    alpha = (100.0 - ci) / 2.0
    lower = float(np.percentile(boot_stats, alpha))
    upper = float(np.percentile(boot_stats, 100.0 - alpha))
    return lower, upper

def paired_cohens_dz(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size < 2:
        return float("nan")
    sd = np.std(differences, ddof=1)
    if sd <= 0:
        return float("nan")
    return float(np.mean(differences) / sd)

def hedges_g_paired_from_differences(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]

    n = differences.size
    if n < 3:
        return float("nan")

    mean_diff = np.mean(differences)
    sd_diff = np.std(differences, ddof=1)
    if sd_diff <= 0:
        return float("nan")

    nu = n - 1
    J = 1.0 - 3.0 / (4.0 * nu - 1.0)
    return float(J * (mean_diff / sd_diff))

def wilcoxon_signed_rank_test(differences: np.ndarray) -> Dict[str, float]:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]

    if differences.size < 2 or not SCIPY_WILCOXON_AVAILABLE:
        return {
            "wilcoxon_statistic": float("nan"),
            "wilcoxon_pvalue": float("nan"),
        }

    nonzero = differences[differences != 0]
    if nonzero.size < 1:
        return {
            "wilcoxon_statistic": float("nan"),
            "wilcoxon_pvalue": float("nan"),
        }

    try:
        stat, pval = wilcoxon(nonzero, alternative="greater")
        return {
            "wilcoxon_statistic": float(stat),
            "wilcoxon_pvalue": float(pval),
        }
    except Exception:
        return {
            "wilcoxon_statistic": float("nan"),
            "wilcoxon_pvalue": float("nan"),
        }

def contiguous_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return inclusive-exclusive index intervals [start, end) where mask is True.
    """
    mask = np.asarray(mask, dtype=bool)
    runs: List[Tuple[int, int]] = []
    start = None

    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            runs.append((start, i))
            start = None

    if start is not None:
        runs.append((start, len(mask)))
    return runs

def smooth_and_clip_sensitivity(
    sensitivity: np.ndarray,
    clip_percentile: float = 95.0,
    sigma: float = 4.0,
) -> np.ndarray:
    s = np.asarray(sensitivity, dtype=float).copy()
    if s.size == 0:
        return s

    finite = np.isfinite(s)
    if not np.any(finite):
        return np.zeros_like(s)

    fill_value = float(np.nanmedian(s[finite]))
    s[~finite] = fill_value

    clip_val = np.percentile(s, clip_percentile)
    s = np.clip(s, 0.0, clip_val)

    if sigma > 0:
        s = gaussian_filter1d(s, sigma=sigma)

    return s

def compute_empirical_sensitivity_threshold(
    sensitivity_smoothed: np.ndarray,
    fixed_errors: np.ndarray,
    norm_errors: np.ndarray,
    min_improvement_px: float = 0.5,
    fallback_percentile: float = 75.0,
) -> float:
    """
    Define an empirical sensitivity threshold from frames where normalization
    meaningfully improves rolling error.
    """
    sensitivity_smoothed = np.asarray(sensitivity_smoothed, dtype=float)
    fixed_errors = np.asarray(fixed_errors, dtype=float)
    norm_errors = np.asarray(norm_errors, dtype=float)

    T = min(len(sensitivity_smoothed), len(fixed_errors), len(norm_errors))
    if T == 0:
        return float("nan")

    delta = fixed_errors[:T] - norm_errors[:T]
    mask = delta > min_improvement_px

    if np.any(mask):
        return float(np.median(sensitivity_smoothed[:T][mask]))
    return float(np.percentile(sensitivity_smoothed[:T], fallback_percentile))

# ============================================================
# Representative frame selection
# ============================================================

def choose_representative_frame(
    cfg: Config,
    records: List[TrackingRecord],
    outputs: ObserverOutputs,
    selected_track_id: Optional[int] = None,
    per_track_representative_frames: Optional[Dict[int, int]] = None,
) -> int:
    if cfg.representative_frame_index is not None:
        return int(np.clip(cfg.representative_frame_index, 0, max(0, len(records) - 1)))

    if per_track_representative_frames is None:
        per_track_representative_frames = {}

    if (
        cfg.frame_selection_mode == "manual"
        and selected_track_id is not None
        and selected_track_id in per_track_representative_frames
    ):
        requested_frame = int(per_track_representative_frames[selected_track_id])

        for i, rec in enumerate(records):
            if rec.frame_idx == requested_frame:
                return i

        frame_indices = np.array([rec.frame_idx for rec in records], dtype=int)
        nearest_i = int(np.argmin(np.abs(frame_indices - requested_frame)))
        return nearest_i

    candidates: List[Tuple[float, int]] = []
    obs_idx = 0

    for i, rec in enumerate(records):
        if rec.tracker_center is None or rec.gt_center is None:
            continue
        if rec.sensitivity_proxy is None:
            continue
        if obs_idx >= len(outputs.fixed_errors):
            break

        score = rec.sensitivity_proxy * (outputs.fixed_errors[obs_idx] + 1e-3)
        candidates.append((score, i))
        obs_idx += 1

    if not candidates:
        return 0

    candidates.sort(reverse=True)
    best_rank = min(19, len(candidates) - 1)
    return candidates[best_rank][1]

# ============================================================
# Candidate frame inspection
# ============================================================

def print_candidate_frame_windows(
    records: List[TrackingRecord],
    selected_track_id: int,
    num_examples: int = 8,
) -> None:
    valid = [rec.frame_idx for rec in records if rec.tracker_center is not None and rec.gt_center is not None]
    if len(valid) == 0:
        print(f"[INFO] No valid frames for track {selected_track_id}")
        return

    if len(valid) <= num_examples:
        print(f"[INFO] Valid frames for track {selected_track_id}: {valid}")
        return

    idxs = np.linspace(0, len(valid) - 1, num_examples).astype(int)
    sampled = [valid[i] for i in idxs]
    print(f"[INFO] Candidate frames for track {selected_track_id}: {sampled}")

def get_candidate_frame_indices(
    records: List[TrackingRecord],
    num_examples: int = 8,
) -> List[int]:
    valid_record_indices = [
        i for i, rec in enumerate(records)
        if rec.tracker_center is not None and rec.gt_center is not None
    ]

    if len(valid_record_indices) == 0:
        return []

    if len(valid_record_indices) <= num_examples:
        return valid_record_indices

    sample_positions = np.linspace(0, len(valid_record_indices) - 1, num_examples).astype(int)
    return [valid_record_indices[pos] for pos in sample_positions]