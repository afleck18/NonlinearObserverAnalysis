import json
import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d, binary_closing

from src.config import Config, TrackingRecord, ObserverOutputs
from src.tracking import normalized_center_error, compute_normalized_precision_curve, build_observer_box_from_center
from src.utils.utils import (
    ensure_dir, 
    rmse, 
    rolling_rmse, 
    percentage_improvement,
    clip_box_to_image,
    xywh_to_xyxy,
    smooth_and_clip_sensitivity, 
    compute_empirical_sensitivity_threshold, 
    contiguous_true_runs, 
    choose_representative_frame, 
    get_candidate_frame_indices
)

# ============================================================
# Main figure
# ============================================================
def build_four_panel_figure(
    cfg: Config,
    records: List[TrackingRecord],
    outputs: ObserverOutputs,
    selected_track_id: int,
    video_output_dir: str,
    per_track_representative_frames: Optional[Dict[int, int]] = None,
) -> None:
    ensure_dir(cfg.output_dir)

    # Prepare aligned arrays for plotting
    valid_records = [r for r in records if r.tracker_center is not None and r.gt_center is not None]
    T = min(
        len(valid_records),
        len(outputs.fixed_errors),
        len(outputs.norm_errors),
        len(outputs.sensitivity_proxy),
    )
    t = np.arange(T)

    fixed_errors = np.array(outputs.fixed_errors[:T], dtype=float)
    norm_errors = np.array(outputs.norm_errors[:T], dtype=float)
    sensitivity = np.array(outputs.sensitivity_proxy[:T], dtype=float)

    fixed_rmse = rmse(fixed_errors)
    norm_rmse = rmse(norm_errors)
    improvement = percentage_improvement(fixed_rmse, norm_rmse)

    # Representative frame
    rep_idx = choose_representative_frame(
        cfg=cfg,
        records=records,
        outputs=outputs,
        selected_track_id=selected_track_id,
        per_track_representative_frames=per_track_representative_frames,
    )
    rep = records[rep_idx]
    rep_img = rep.frame_bgr.copy()

    fixed_pt = None
    norm_pt = None

    if rep.tracker_box_xywh is not None:
        rep_img = draw_box_xywh(rep_img, rep.tracker_box_xywh, (255, 0, 0), "Trk")
    if rep.gt_box_xywh is not None:
        rep_img = draw_box_xywh(rep_img, rep.gt_box_xywh, (0, 255, 0), "GT")
    if rep.gt_center is not None:
        rep_img = draw_point(rep_img, rep.gt_center, (0, 255, 0), "GT")

    frame_to_obs_idx = {frame_idx: i for i, frame_idx in enumerate(outputs.valid_frame_indices[:T])}
    if rep.frame_idx in frame_to_obs_idx:
        oi = frame_to_obs_idx[rep.frame_idx]
        if oi < len(outputs.fixed_centers) and outputs.fixed_centers[oi] is not None:
            fixed_pt = outputs.fixed_centers[oi]
            rep_img = draw_point(rep_img, fixed_pt, (0, 0, 255), "Fxd")
        if oi < len(outputs.norm_centers) and outputs.norm_centers[oi] is not None:
            norm_pt = outputs.norm_centers[oi]
            rep_img = draw_point(rep_img, norm_pt, (0, 165, 255), "Nrm")

    if getattr(cfg, "use_tight_crop_panel_a", False):
        rep_img = make_tight_crop_panel_a(cfg, rep_img, rep, fixed_pt, norm_pt)

    rep_img_rgb = cv2.cvtColor(rep_img, cv2.COLOR_BGR2RGB)

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Representative frame
    ax = axes[0, 0]
    ax.imshow(rep_img_rgb)
    ax.set_title("(a) Representative frame")
    ax.axis("off")

    # (b) Trajectory error over time
    S_t = sensitivity
    S_t = np.clip(S_t, 0, np.percentile(S_t, 95))
    S_t = gaussian_filter1d(S_t, sigma=4)
    ax = axes[0, 1]
    
    # shaded sensitivity regions
    threshold = np.percentile(S_t, 95)
    #s_crit = (1 - cfg.Lf) / (cfg.alpha_fixed * np.linalg.norm(cfg.K_tilde))
    high = S_t > threshold
    
    start = None
    for i in range(len(high)):
        if high[i] and start is None:
            start = i
        elif not high[i] and start is not None:
            ax.axvspan(start, i, color='gray', alpha=0.12)
            start = None

    # handle end
    if start is not None:
        ax.axvspan(start, len(high), color='gray', alpha=0.12)

    # raw errors (faint)
    ax.plot(fixed_errors, color='tab:blue', alpha=0.2)
    ax.plot(norm_errors,  color='tab:orange', alpha=0.2)


    # rolling RMSE
    rmse_fixed = rolling_rmse(fixed_errors, window=25)
    rmse_norm  = rolling_rmse(norm_errors, window=25)

    ax.plot(rmse_fixed, color='tab:blue', linewidth=2.5,
            label='Fixed gain (rolling RMSE)')

    ax.plot(rmse_norm, color='tab:orange', linewidth=2.5,
            label='Normalized gain (rolling RMSE)')

    ax.set_xlabel("Frame")
    ax.set_ylabel("Center error (pixels)")
    ax.set_title("Observer trajectory error over time")
    ax.legend()
    ax.grid(False,alpha=0.1)

    # (c) Sensitivity proxy

    
    delta = rmse_fixed - rmse_norm
    mask = delta > 0.5   # example margin in pixels
    S_crit_emp = np.median(S_t[mask])

    ax = axes[1, 0]
    ax.plot(t, S_t)
    ax.axhline(S_crit_emp, color='red', linestyle='--', linewidth=2,
           label='Empirical sensitivity threshold')

    min_width = 15   # minimum frames to shade

    high = S_t > S_crit_emp

    # remove small gaps (merge nearby regions)
    high = binary_closing(high, structure=np.ones(20))
    # minimum width
    min_width = 20

    start = None
    for i in range(len(high)):
        if high[i] and start is None:
            start = i
        elif not high[i] and start is not None:
            if i - start >= min_width:
                ax.axvspan(start, i, color='gray', alpha=0.18)
            start = None

    if start is not None:
        if len(high) - start >= min_width:
            ax.axvspan(start, len(high), color='gray', alpha=0.18)


    ax.set_title("(c) Sensitivity proxy over time")
    ax.set_xlabel("Frame")
    ax.set_ylabel(r"$S_t$")
    ax.grid(True, alpha=0.3)
   
    # (d) RMSE comparison
    ax = axes[1, 1]
    bars = ax.bar(["Fixed", "Normalized"], [fixed_rmse, norm_rmse])
    ax.set_title("(d) RMSE comparison")
    ax.set_ylabel("RMSE (pixels)")
    ax.grid(True, axis="y", alpha=0.3)

    ymax = max(fixed_rmse, norm_rmse) * 1.2 if np.isfinite(max(fixed_rmse, norm_rmse)) else 1.0
    ax.set_ylim(0, ymax)
    ax.text(
        0.5,
        ymax * 0.9,
        f"Improvement: {improvement:.1f}%",
        ha="center",
        va="center",
        fontsize=11,
    )

    fig.suptitle("UOT32 Real-Data Validation", fontsize=14)
    fig.tight_layout()

    out_png = os.path.join(video_output_dir, f"four_panel_id_{selected_track_id}.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "fixed_rmse": fixed_rmse,
        "normalized_rmse": norm_rmse,
        "improvement_percent": improvement,
        "num_valid_frames": int(T),
        "representative_frame_index": int(rep_idx),
    }

    print(f"Saved figure to: {out_png}")
    print(json.dumps(metrics, indent=2))

def build_five_panel_figure(
    cfg: Config,
    records: List[TrackingRecord],
    outputs: ObserverOutputs,
    selected_track_id: int,
    video_output_dir: str,
    per_track_representative_frames: Optional[Dict[int, int]] = None,
) -> Dict:
    """
    Build an IEEE/CDC-friendly portrait 5-panel figure:

        Row 1: (a) Representative frame      (d) RMSE
        Row 2: (b) Center error over time    [full width]
        Row 3: (c) Sensitivity proxy         (e) Normalized precision

    Expected box format throughout this function:
        [cx, cy, w, h]
    """
    ensure_dir(video_output_dir)

    # ------------------------------------------------------------------
    # IEEE/CDC-friendly plotting defaults
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })


    # ------------------------------------------------------------------
    # Validate usable time horizon
    # ------------------------------------------------------------------
    T = min(
        len(outputs.fixed_errors),
        len(outputs.norm_errors),
        len(outputs.sensitivity_proxy),
        len(outputs.valid_frame_indices),
    )

    if T == 0:
        print(f"[WARNING] No valid data for selected track {selected_track_id}.")
        return {
            "selected_track_id": selected_track_id,
            "fixed_rmse": float("nan"),
            "normalized_rmse": float("nan"),
            "improvement_percent": float("nan"),
            "num_valid_frames": 0,
            "representative_frame_index": -1,
            "improved": 0,
            "empirical_sensitivity_threshold": float("nan"),
            "norm_precision_auc_fixed": float("nan"),
            "norm_precision_auc_normalized": float("nan"),
            "norm_precision_at_01_fixed": float("nan"),
            "norm_precision_at_01_normalized": float("nan"),
            "norm_precision_num_frames": 0,
        }

    # ------------------------------------------------------------------
    # Pull core arrays
    # ------------------------------------------------------------------
    fixed_errors = np.asarray(outputs.fixed_errors[:T], dtype=float)
    norm_errors = np.asarray(outputs.norm_errors[:T], dtype=float)
    sensitivity_raw = np.asarray(outputs.sensitivity_proxy[:T], dtype=float)
    t = np.arange(T)

    fixed_rmse = rmse(fixed_errors)
    norm_rmse = rmse(norm_errors)
    improvement_percent = percentage_improvement(fixed_rmse, norm_rmse)

    # ------------------------------------------------------------------
    # Smoothed sensitivity and rolling error
    # ------------------------------------------------------------------
    S_t = smooth_and_clip_sensitivity(
        sensitivity_raw,
        clip_percentile=95.0,
        sigma=4.0,
    )

    rmse_fixed_roll = rolling_rmse(fixed_errors, window=cfg.rolling_window)
    rmse_norm_roll = rolling_rmse(norm_errors, window=cfg.rolling_window)

    S_crit_emp = compute_empirical_sensitivity_threshold(
        sensitivity_smoothed=S_t,
        fixed_errors=rmse_fixed_roll,
        norm_errors=rmse_norm_roll,
        min_improvement_px=0.5,
        fallback_percentile=75.0,
    )

    # ------------------------------------------------------------------
    # High-sensitivity intervals (lightly shaded)
    # ------------------------------------------------------------------
    high = np.isfinite(S_t) & np.isfinite(S_crit_emp) & (S_t > S_crit_emp)
    high = binary_closing(high, structure=np.ones(20))
    high_runs = [
        (start, end)
        for (start, end) in contiguous_true_runs(high)
        if (end - start) >= 20
    ]

    # ------------------------------------------------------------------
    # Choose representative frame and build panel (a) image
    # ------------------------------------------------------------------
    rep_idx = choose_representative_frame(
        cfg=cfg,
        records=records,
        outputs=outputs,
        selected_track_id=selected_track_id,
        per_track_representative_frames=per_track_representative_frames,
    )
    rep = records[rep_idx]
    rep_img = rep.frame_bgr.copy()

    fixed_pt = None
    norm_pt = None

    if rep.tracker_box_xywh is not None:
        rep_img = draw_box_xywh(rep_img, rep.tracker_box_xywh, (255, 0, 0), "Trk")
    if rep.gt_box_xywh is not None:
        rep_img = draw_box_xywh(rep_img, rep.gt_box_xywh, (0, 255, 0), "GT")
    if rep.gt_center is not None:
        rep_img = draw_point(rep_img, rep.gt_center, (0, 255, 0), "GT")

    frame_to_obs_idx = {frame_idx: i for i, frame_idx in enumerate(outputs.valid_frame_indices[:T])}
    if rep.frame_idx in frame_to_obs_idx:
        oi = frame_to_obs_idx[rep.frame_idx]
        if oi < len(outputs.fixed_centers) and outputs.fixed_centers[oi] is not None:
            fixed_pt = outputs.fixed_centers[oi]
            rep_img = draw_point(rep_img, fixed_pt, (0, 0, 255), "Fxd")
        if oi < len(outputs.norm_centers) and outputs.norm_centers[oi] is not None:
            norm_pt = outputs.norm_centers[oi]
            rep_img = draw_point(rep_img, norm_pt, (0, 165, 255), "Nrm")

    if getattr(cfg, "use_tight_crop_panel_a", False):
        rep_img = make_tight_crop_panel_a(cfg, rep_img, rep, fixed_pt, norm_pt)

    rep_img_rgb = cv2.cvtColor(rep_img, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # Figure layout: portrait IEEE/CDC-friendly
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(6.9, 8.8))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.95, 1.25, 1.0],
        width_ratios=[1.0, 1.0],
        hspace=0.38,
        wspace=0.28,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])

    ax_b = fig.add_subplot(gs[1, :])

    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])

    for ax in [ax_b, ax_c, ax_d, ax_e]:
        ieee_axis(ax)
    # ------------------------------------------------------------------
    # (a) Representative frame
    # ------------------------------------------------------------------
    ax_a.imshow(rep_img_rgb)
    ax_a.set_title("(a) Representative frame")
    ax_a.axis("off")

    # ------------------------------------------------------------------
    # (b) Center error over time
    # ------------------------------------------------------------------
    for start, end in high_runs:
        ax_b.axvspan(start, end, color="gray", alpha=0.08, zorder=0)

    # faint raw traces
    ax_b.plot(t, fixed_errors, alpha=0.12, linewidth=0.8, label="_nolegend_")
    ax_b.plot(t, norm_errors, alpha=0.12, linewidth=0.8, label="_nolegend_")

    # prominent rolling curves
    ax_b.plot(
        t,
        rmse_fixed_roll,
        linewidth=2.0,
        color="tab:green",
        label="Fixed gain (rolling RMSE)",
        zorder=3,
    )
    ax_b.plot(
        t,
        rmse_norm_roll,
        linewidth=2.0,
        color="tab:red",
        label="Normalized gain (rolling RMSE)",
        zorder=4,
    )

    ax_b.set_title("(b) Center error", loc="left")
    ax_b.set_xlabel("Frame")
    ax_b.set_ylabel("Center error (pixels)")
    ax_b.legend(loc="upper right", framealpha=0.95)
    ax_b.grid(False)
    ax_b.tick_params(width=0.8, length=3)

    # ------------------------------------------------------------------
    # (c) Sensitivity proxy
    # ------------------------------------------------------------------
    for start, end in high_runs:
        ax_c.axvspan(start, end, color="tab:blue", alpha=0.08, zorder=0)

    ax_c.plot(t, S_t, linewidth=1.8, color="tab:blue")
    if np.isfinite(S_crit_emp):
        ax_c.axhline(
            S_crit_emp,
            linestyle="--",
            linewidth=1.2,
            color="tab:blue",
            alpha=0.9,
            label="Empirical sensitivity threshold",
        )

    ax_c.set_title("(c) Sensitivity")
    ax_c.set_xlabel("Frame")
    ax_c.set_ylabel(r"$S_t$")
    ax_c.grid(True, alpha=0.15)
    ax_c.legend(loc="lower left", framealpha=0.95)
    ax_c.tick_params(width=0.8, length=3)

    if np.isfinite(S_crit_emp):
        ax_c.text(
            0.03,
            0.93,
            f"Threshold = {S_crit_emp:.2f}",
            transform=ax_c.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="0.75"),
        )

    # ------------------------------------------------------------------
    # (d) RMSE comparison
    # ------------------------------------------------------------------
    bars = ax_d.bar(
        ["Fixed", "Normalized"],
        [fixed_rmse, norm_rmse],
        color=["tab:blue", "tab:blue"],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6
    )
    ax_d.set_title("(d) RMSE")
    ax_d.set_ylabel("RMSE (pixels)")
    ax_d.grid(True, axis="y", alpha=0.15)
    ax_d.tick_params(width=0.8, length=3)

    ymax = max(fixed_rmse, norm_rmse)
    ymax = 1.22 * ymax if np.isfinite(ymax) and ymax > 0 else 1.0
    ax_d.set_ylim(0.0, ymax)

    ax_d.text(
        0.5,
        0.90 * ymax,
        f"Improvement: {improvement_percent:.1f}%",
        ha="center",
        va="center",
        fontsize=7.5,
    )

    # ============================================================
    # (e) Normalized precision plot
    # ============================================================

    # Helper: build map from frame index -> record
    frame_to_record = {rec.frame_idx: rec for rec in records}

    # Optional explicit observer boxes if available
    has_fixed_boxes = hasattr(outputs, "fixed_boxes_xywh") and outputs.fixed_boxes_xywh is not None
    has_norm_boxes = hasattr(outputs, "norm_boxes_xywh") and outputs.norm_boxes_xywh is not None

    gt_boxes = []
    fixed_boxes = []
    norm_boxes = []

    for obs_idx, frame_idx in enumerate(outputs.valid_frame_indices[:T]):
        rec = frame_to_record.get(frame_idx, None)
        if rec is None:
            continue
        if rec.gt_box_xywh is None:
            continue

        gt_box = np.asarray(rec.gt_box_xywh, dtype=float)

        # Fixed observer box
        fixed_box = None
        if has_fixed_boxes and obs_idx < len(outputs.fixed_boxes_xywh):
            if outputs.fixed_boxes_xywh[obs_idx] is not None:
                fixed_box = np.asarray(outputs.fixed_boxes_xywh[obs_idx], dtype=float)
        elif obs_idx < len(outputs.fixed_centers):
            if outputs.fixed_centers[obs_idx] is not None:
                fixed_box = build_observer_box_from_center(rec, outputs.fixed_centers[obs_idx])

        # Normalized observer box
        norm_box = None
        if has_norm_boxes and obs_idx < len(outputs.norm_boxes_xywh):
            if outputs.norm_boxes_xywh[obs_idx] is not None:
                norm_box = np.asarray(outputs.norm_boxes_xywh[obs_idx], dtype=float)
        elif obs_idx < len(outputs.norm_centers):
            if outputs.norm_centers[obs_idx] is not None:
                norm_box = build_observer_box_from_center(rec, outputs.norm_centers[obs_idx])

        if fixed_box is None or norm_box is None:
            continue

        gt_boxes.append(gt_box)
        fixed_boxes.append(fixed_box)
        norm_boxes.append(norm_box)

    # Default metrics in case nothing valid is available
    fixed_norm_prec_auc = float("nan")
    norm_norm_prec_auc = float("nan")
    norm_prec_at_01_fixed = float("nan")
    norm_prec_at_01_norm = float("nan")
    n_norm_prec_frames = 0

    if len(gt_boxes) == 0:
        ax_e.text(
            0.5,
            0.5,
            "No valid boxes available\nfor normalized precision plot",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax_e.set_title("(e) Normalized precision", loc="left")
        ax_e.axis("off")
    else:
        gt_boxes = np.asarray(gt_boxes, dtype=float)
        fixed_boxes = np.asarray(fixed_boxes, dtype=float)
        norm_boxes = np.asarray(norm_boxes, dtype=float)

        # Normalized center error using GT box diagonal
        fixed_norm_errs = np.array(
            [normalized_center_error(fb, gb, normalization="diag") for fb, gb in zip(fixed_boxes, gt_boxes)],
            dtype=float,
        )
        norm_norm_errs = np.array(
            [normalized_center_error(nb, gb, normalization="diag") for nb, gb in zip(norm_boxes, gt_boxes)],
            dtype=float,
        )

        # Thresholds for the displayed curve
        norm_thresholds = np.linspace(0.0, 0.30, 61)

        fixed_norm_prec_curve = compute_normalized_precision_curve(
            fixed_norm_errs,
            norm_thresholds,
        )
        norm_norm_prec_curve = compute_normalized_precision_curve(
            norm_norm_errs,
            norm_thresholds,
        )

        # AUC over displayed range [0, 0.30]
        if hasattr(np, "trapezoid"):
            fixed_norm_prec_auc = float(np.trapezoid(fixed_norm_prec_curve, norm_thresholds) / 0.30)
            norm_norm_prec_auc = float(np.trapezoid(norm_norm_prec_curve, norm_thresholds) / 0.30)
        else:
            fixed_norm_prec_auc = float(np.trapezoid(fixed_norm_prec_curve, norm_thresholds) / 0.30)
            norm_norm_prec_auc = float(np.trapezoid(norm_norm_prec_curve, norm_thresholds) / 0.30)

        # Precision at 0.10
        thr = 0.10
        idx_01 = int(np.argmin(np.abs(norm_thresholds - thr)))
        norm_prec_at_01_fixed = float(fixed_norm_prec_curve[idx_01])
        norm_prec_at_01_norm = float(norm_norm_prec_curve[idx_01])

        n_norm_prec_frames = int(np.sum(np.isfinite(fixed_norm_errs) & np.isfinite(norm_norm_errs)))

        # Plot curves
        ax_e.plot(
            norm_thresholds,
            fixed_norm_prec_curve,
            linewidth=2.0,
            color="tab:blue",
            label="Fixed gain",
        )
        ax_e.plot(
            norm_thresholds,
            norm_norm_prec_curve,
            linewidth=2.0,
            color="tab:orange",
            label="Normalized gain",
        )

        # Threshold line
        ax_e.axvline(
            thr,
            linestyle="--",
            linewidth=1.0,
            color="gray",
            alpha=0.8,
        )
        ax_e.text(
            0.102,
            0.15,
            "@0.10",
            rotation=90,
            fontsize=7,
            color="gray",
            va="bottom"
        )

        # Threshold label at top
        ax_e.text(
            thr,
            0.995,
            "@0.10",
            transform=ax_e.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
            color="gray",
        )

        # Metrics box at top-left
        ax_e.text(
            0.02,
            0.98,
            "\n".join([
                f"AUC(0–0.3): {fixed_norm_prec_auc:.3f} / {norm_norm_prec_auc:.3f}",
                f"@0.10: {norm_prec_at_01_fixed:.3f} / {norm_prec_at_01_norm:.3f}",
            ]),
            transform=ax_e.transAxes,
            fontsize=7.5,
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.9,
                edgecolor="gray",
            ),
        )

        # Axes styling
        ax_e.set_title("(e) Normalized precision", loc="left")
        ax_e.set_xlabel("Normalized center-error threshold")
        ax_e.set_ylabel("Precision")
        ax_e.set_xlim(0.0, 0.30)
        ax_e.set_ylim(0.0, 1.0)
        ax_e.legend(loc="lower right", frameon=False)
        ax_e.grid(True, alpha=0.15)

        # IEEE-style spines
        ax_e.spines["top"].set_visible(False)
        ax_e.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Final layout and save
    # ------------------------------------------------------------------
    out_png = os.path.join(video_output_dir, f"five_panel_id_{selected_track_id}.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "selected_track_id": selected_track_id,
        "fixed_rmse": float(fixed_rmse),
        "normalized_rmse": float(norm_rmse),
        "improvement_percent": float(improvement_percent),
        "num_valid_frames": int(T),
        "representative_frame_index": int(rep.frame_idx),
        "improved": int(np.isfinite(improvement_percent) and improvement_percent > 0.0),
        "empirical_sensitivity_threshold": float(S_crit_emp) if np.isfinite(S_crit_emp) else float("nan"),
        "norm_precision_auc_fixed": float(fixed_norm_prec_auc) if np.isfinite(fixed_norm_prec_auc) else float("nan"),
        "norm_precision_auc_normalized": float(norm_norm_prec_auc) if np.isfinite(norm_norm_prec_auc) else float("nan"),
        "norm_precision_at_01_fixed": float(norm_prec_at_01_fixed) if np.isfinite(norm_prec_at_01_fixed) else float("nan"),
        "norm_precision_at_01_normalized": float(norm_prec_at_01_norm) if np.isfinite(norm_prec_at_01_norm) else float("nan"),
        "norm_precision_num_frames": int(n_norm_prec_frames),
    }

    metrics_path = os.path.join(video_output_dir, f"metrics_id_{selected_track_id}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved 5-panel portrait figure to: {out_png}")
    print(json.dumps(metrics, indent=2))
    return metrics

# ============================================================
# Drawing
# ============================================================

def draw_box_xywh(img, box_xywh, color, label, thickness=2):
    out = img.copy()
    cx, cy, w, h = box_xywh
    x1 = int(round(cx - w / 2))
    y1 = int(round(cy - h / 2))
    x2 = int(round(cx + w / 2))
    y2 = int(round(cy + h / 2))
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(out, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out

def draw_point(img, pt, color, label, radius=4):
    out = img.copy()
    x, y = int(round(pt[0])), int(round(pt[1]))
    cv2.circle(out, (x, y), radius, color, -1)
    cv2.putText(out, label, (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out

# ============================================================
# Tight crop for panel (a)
# ============================================================

def make_tight_crop_panel_a(
    cfg: Config,
    rep_img: np.ndarray,
    rep: TrackingRecord,
    fixed_pt: Optional[np.ndarray],
    norm_pt: Optional[np.ndarray],
) -> np.ndarray:
    h, w = rep_img.shape[:2]

    boxes = []
    points = []

    if rep.tracker_box_xywh is not None:
        boxes.append(rep.tracker_box_xywh)
    if rep.gt_box_xywh is not None:
        boxes.append(rep.gt_box_xywh)

    if rep.tracker_center is not None:
        points.append(rep.tracker_center)
    if rep.gt_center is not None:
        points.append(rep.gt_center)
    if fixed_pt is not None:
        points.append(fixed_pt)
    if norm_pt is not None:
        points.append(norm_pt)

    if len(boxes) == 0 and len(points) == 0:
        return rep_img

    xs = []
    ys = []

    for box in boxes:
        x1, y1, x2, y2 = xywh_to_xyxy(box)
        xs.extend([x1, x2])
        ys.extend([y1, y2])

    for pt in points:
        xs.append(pt[0])
        ys.append(pt[1])

    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)

    crop_w = max(bw + 2 * cfg.crop_padding_px, cfg.crop_min_size_px)
    crop_h = max(bh + 2 * cfg.crop_padding_px, cfg.crop_min_size_px)

    crop_x1 = cx - crop_w / 2.0
    crop_y1 = cy - crop_h / 2.0
    crop_x2 = cx + crop_w / 2.0
    crop_y2 = cy + crop_h / 2.0

    crop_x1, crop_y1, crop_x2, crop_y2 = clip_box_to_image(crop_x1, crop_y1, crop_x2, crop_y2, w, h)

    cropped = rep_img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    if cropped.size == 0:
        return rep_img
    return cropped

def ieee_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

# ============================================================
# Contact sheet helpers
# ============================================================

def draw_panel_a_overlays_for_record(
    rec: TrackingRecord,
    fixed_pt: Optional[np.ndarray] = None,
    norm_pt: Optional[np.ndarray] = None,
) -> np.ndarray:
    img = rec.frame_bgr.copy()

    if rec.tracker_box_xywh is not None:
        img = draw_box_xywh(img, rec.tracker_box_xywh, (255, 0, 0), "Tracker")

    if rec.gt_box_xywh is not None:
        img = draw_box_xywh(img, rec.gt_box_xywh, (0, 255, 0), "GT")

    if rec.gt_center is not None:
        img = draw_point(img, rec.gt_center, (0, 255, 0), "GT ctr")

    if fixed_pt is not None:
        img = draw_point(img, fixed_pt, (0, 0, 255), "Fixed")

    if norm_pt is not None:
        img = draw_point(img, norm_pt, (0, 165, 255), "Norm")

    return img

def save_contact_sheet_for_track(
    cfg: Config,
    records: List[TrackingRecord],
    outputs: ObserverOutputs,
    selected_track_id: int,
    video_output_dir: str,
) -> Optional[str]:
    if not cfg.save_contact_sheets:
        return None

    candidate_record_indices = get_candidate_frame_indices(
        records,
        num_examples=cfg.contact_sheet_num_examples,
    )

    if len(candidate_record_indices) == 0:
        print(f"[INFO] No valid candidate crops for track {selected_track_id}")
        return None

    frame_to_obs_idx = {frame_idx: i for i, frame_idx in enumerate(outputs.valid_frame_indices)}

    tiles = []
    titles = []

    for rec_idx in candidate_record_indices:
        rec = records[rec_idx]

        fixed_pt = None
        norm_pt = None

        if rec.frame_idx in frame_to_obs_idx:
            oi = frame_to_obs_idx[rec.frame_idx]
            if oi < len(outputs.fixed_centers) and outputs.fixed_centers[oi] is not None:
                fixed_pt = outputs.fixed_centers[oi]
            if oi < len(outputs.norm_centers) and outputs.norm_centers[oi] is not None:
                norm_pt = outputs.norm_centers[oi]

        vis = draw_panel_a_overlays_for_record(rec, fixed_pt=fixed_pt, norm_pt=norm_pt)

        if cfg.use_tight_crop_panel_a:
            vis = make_tight_crop_panel_a(cfg, vis, rec, fixed_pt, norm_pt)

        tiles.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        titles.append(f"frame {rec.frame_idx}")

    n_tiles = len(tiles)
    ncols = max(1, cfg.contact_sheet_columns)
    nrows = int(np.ceil(n_tiles / ncols))
    nrows = min(nrows, cfg.contact_sheet_max_rows)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    axes_flat = axes.flatten()
    for ax in axes_flat:
        ax.axis("off")

    for i, (tile, title) in enumerate(zip(tiles, titles)):
        axes_flat[i].imshow(tile)
        axes_flat[i].set_title(title)
        axes_flat[i].axis("off")

    fig.suptitle(f"Candidate panel (a) crops for track {selected_track_id}", fontsize=14)
    fig.tight_layout()

    out_path = os.path.join(video_output_dir, f"contact_sheet_track_{selected_track_id}.pdf")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved contact sheet to: {out_path}")
    return out_path
