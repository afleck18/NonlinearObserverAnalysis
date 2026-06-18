import os
import numpy as np
from typing import Dict, List, Optional, Tuple
import csv
from ultralytics import YOLO

from src.config import Config, TrackingRecord, MultiTrackingRecord, FrameDetection

# ============================================================
# Tracking
# ============================================================

def run_tracking_and_extract_all_tracks(cfg: Config, video_path: str, model_path: str) -> List[MultiTrackingRecord]:
    model = YOLO(model_path)

    results = model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker=cfg.tracker_yaml,
        conf=cfg.conf,
        iou=cfg.iou,
        imgsz=cfg.imgsz,
        device=cfg.device,
        verbose=False,
    )

    multi_records: List[MultiTrackingRecord] = []

    for frame_idx, res in enumerate(results):
        frame = res.orig_img.copy()
        detections: List[FrameDetection] = []

        if res.boxes is not None and len(res.boxes) > 0:
            boxes_xywh = res.boxes.xywh.cpu().numpy()
            boxes_conf = res.boxes.conf.cpu().numpy()

            if res.boxes.id is not None:
                boxes_ids = res.boxes.id.cpu().numpy().astype(int)
            else:
                boxes_ids = np.full(len(boxes_xywh), -1, dtype=int)

            for i in range(len(boxes_xywh)):
                tid = int(boxes_ids[i])
                if tid < 0:
                    continue

                box = boxes_xywh[i].astype(float)
                det = FrameDetection(
                    track_id=tid,
                    box_xywh=box.copy(),
                    center=box[:2].copy(),
                    conf=float(boxes_conf[i]),
                )
                detections.append(det)

        multi_records.append(
            MultiTrackingRecord(
                frame_idx=frame_idx,
                frame_bgr=frame,
                detections=detections,
            )
        )

    return multi_records

# ============================================================
# Track summary / selection
# ============================================================

def build_track_summary(multi_records: List[MultiTrackingRecord]) -> List[Dict]:
    summary: Dict[int, Dict] = {}

    for rec in multi_records:
        for det in rec.detections:
            tid = det.track_id
            if tid not in summary:
                summary[tid] = {
                    "track_id": tid,
                    "num_frames": 0,
                    "first_frame": rec.frame_idx,
                    "last_frame": rec.frame_idx,
                    "conf_list": [],
                    "w_list": [],
                    "h_list": [],
                }

            summary[tid]["num_frames"] += 1
            summary[tid]["last_frame"] = rec.frame_idx
            summary[tid]["conf_list"].append(det.conf)
            summary[tid]["w_list"].append(float(det.box_xywh[2]))
            summary[tid]["h_list"].append(float(det.box_xywh[3]))

    rows: List[Dict] = []
    for tid, row in summary.items():
        rows.append({
            "track_id": tid,
            "num_frames": row["num_frames"],
            "first_frame": row["first_frame"],
            "last_frame": row["last_frame"],
            "mean_conf": float(np.mean(row["conf_list"])) if row["conf_list"] else float("nan"),
            "mean_w": float(np.mean(row["w_list"])) if row["w_list"] else float("nan"),
            "mean_h": float(np.mean(row["h_list"])) if row["h_list"] else float("nan"),
        })

    rows = [r for r in rows if r["num_frames"] >= 1]
    rows.sort(key=lambda x: (-x["num_frames"], -x["mean_conf"]))
    return rows

def save_track_summary_csv(video_output_dir: str, rows: List[Dict]) -> None:
    out_csv = os.path.join(video_output_dir, "track_summary.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["track_id", "num_frames", "first_frame", "last_frame", "mean_conf", "mean_w", "mean_h"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved track summary to: {out_csv}")

def choose_default_track_ids(cfg: Config, summary_rows: List[Dict]) -> List[int]:
    candidates = [r for r in summary_rows if r["num_frames"] >= cfg.min_track_length]
    if len(candidates) == 0:
        candidates = summary_rows

    if len(candidates) == 0:
        raise RuntimeError("No tracks found at all.")

    return [int(r["track_id"]) for r in candidates[:cfg.auto_select_top_k]]

def xywh_to_xyxy(box_xywh: np.ndarray) -> np.ndarray:
    """
    Convert [x, y, w, h] to [x1, y1, x2, y2].
    Assumes x, y are top-left.
    """
    x, y, w, h = [float(v) for v in box_xywh]
    return np.array([x, y, x + w, y + h], dtype=float)

def center_wh_to_xywh(center: Tuple[float, float], w: float, h: float) -> np.ndarray:
    """
    Build [x, y, w, h] from center coordinates and width/height.
    """
    cx, cy = float(center[0]), float(center[1])
    return np.array([cx - w / 2.0, cy - h / 2.0, float(w), float(h)], dtype=float)

def iou_xywh(box1_xywh: np.ndarray, box2_xywh: np.ndarray) -> float:
    """
    IoU for two [x, y, w, h] boxes.
    """
    if box1_xywh is None or box2_xywh is None:
        return float("nan")

    x1_1, y1_1, x2_1, y2_1 = xywh_to_xyxy(box1_xywh)
    x1_2, y1_2, x2_2, y2_2 = xywh_to_xyxy(box2_xywh)

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x2_1 - x1_1) * max(0.0, y2_1 - y1_1)
    area2 = max(0.0, x2_2 - x1_2) * max(0.0, y2_2 - y1_2)

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0

    return float(inter_area / union_area)

def build_observer_box_from_center(
    record,
    observer_center: Optional[Tuple[float, float]],
) -> Optional[np.ndarray]:
    """
    Construct observer box using observer center and tracker/detector width-height.
    Falls back to GT width-height if tracker box is unavailable.
    """
    if observer_center is None:
        return None

    ref_box = None
    if getattr(record, "tracker_box_xywh", None) is not None:
        ref_box = np.asarray(record.tracker_box_xywh, dtype=float)
    elif getattr(record, "gt_box_xywh", None) is not None:
        ref_box = np.asarray(record.gt_box_xywh, dtype=float)

    if ref_box is None:
        return None

    w = float(ref_box[2])
    h = float(ref_box[3])
    return center_wh_to_xywh(observer_center, w, h)

def compute_success_curve(ious: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Success rate at each IoU threshold:
    fraction of valid frames with IoU >= threshold.
    """
    ious = np.asarray(ious, dtype=float)
    valid = np.isfinite(ious)
    if not np.any(valid):
        return np.full_like(thresholds, np.nan, dtype=float)

    ious = ious[valid]
    curve = np.array([np.mean(ious >= thr) for thr in thresholds], dtype=float)
    return curve

def box_center_from_xywh_center_format(box_xywh: np.ndarray) -> np.ndarray:
    """
    Return center (cx, cy) from [cx, cy, w, h].
    """
    box_xywh = np.asarray(box_xywh, dtype=float)
    return box_xywh[:2].copy()

def build_observer_box_from_center(
    rec: TrackingRecord,
    observer_center: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """
    Build an observer box in [cx, cy, w, h] format using the observer center
    and the tracker box size. Falls back to GT box size if tracker box is unavailable.
    """
    if observer_center is None:
        return None

    ref_box = None
    if rec.tracker_box_xywh is not None:
        ref_box = np.asarray(rec.tracker_box_xywh, dtype=float)
    elif rec.gt_box_xywh is not None:
        ref_box = np.asarray(rec.gt_box_xywh, dtype=float)

    if ref_box is None:
        return None

    w = float(ref_box[2])
    h = float(ref_box[3])

    return np.array([
        float(observer_center[0]),
        float(observer_center[1]),
        w,
        h,
    ], dtype=float)

def normalized_center_error(
    pred_box_xywh: np.ndarray,
    gt_box_xywh: np.ndarray,
    normalization: str = "diag",
) -> float:
    """
    Compute normalized center error using boxes in [cx, cy, w, h] format.

    normalization:
        - "diag": divide by GT box diagonal sqrt(w^2 + h^2)
        - "sqrt_area": divide by sqrt(w*h)
    """
    if pred_box_xywh is None or gt_box_xywh is None:
        return float("nan")

    pred_box_xywh = np.asarray(pred_box_xywh, dtype=float)
    gt_box_xywh = np.asarray(gt_box_xywh, dtype=float)

    pred_c = box_center_from_xywh_center_format(pred_box_xywh)
    gt_c = box_center_from_xywh_center_format(gt_box_xywh)

    center_dist = float(np.linalg.norm(pred_c - gt_c))

    gt_w = max(float(gt_box_xywh[2]), 1e-8)
    gt_h = max(float(gt_box_xywh[3]), 1e-8)

    if normalization == "sqrt_area":
        denom = np.sqrt(gt_w * gt_h)
    else:  # default
        denom = np.sqrt(gt_w ** 2 + gt_h ** 2)

    denom = max(float(denom), 1e-8)
    return float(center_dist / denom)

def compute_normalized_precision_curve(
    normalized_errors: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    Normalized precision curve:
    fraction of valid frames with normalized center error <= threshold.
    """
    errs = np.asarray(normalized_errors, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    valid = np.isfinite(errs)
    if not np.any(valid):
        return np.full_like(thresholds, np.nan, dtype=float)

    errs = errs[valid]
    curve = np.array([np.mean(errs <= thr) for thr in thresholds], dtype=float)
    return curve

# ============================================================
# GT matching
# ============================================================

def match_yolo_track_to_gt_target_visdrone(
    multi_records: List[MultiTrackingRecord],
    gt_by_frame: Dict[int, Dict[int, np.ndarray]],
    selected_track_id: int,
) -> Optional[int]:
    gt_scores: Dict[int, List[float]] = {}

    for rec in multi_records:
        yolo_det = None
        for det in rec.detections:
            if det.track_id == selected_track_id:
                yolo_det = det
                break

        if yolo_det is None:
            continue
        if rec.frame_idx not in gt_by_frame:
            continue

        gt_targets = gt_by_frame[rec.frame_idx]
        for gt_tid, gt_box in gt_targets.items():
            score = iou_xywh(yolo_det.box_xywh, gt_box)
            gt_scores.setdefault(gt_tid, []).append(score)

    if len(gt_scores) == 0:
        return None

    best_tid = None
    best_mean = -1.0
    for gt_tid, scores in gt_scores.items():
        mean_score = float(np.mean(scores))
        if mean_score > best_mean:
            best_mean = mean_score
            best_tid = gt_tid

    print(f"[INFO] Matched YOLO track {selected_track_id} to GT target {best_tid} with mean IoU {best_mean:.3f}")
    return best_tid

