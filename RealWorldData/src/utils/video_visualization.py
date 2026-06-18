import os
import cv2
import numpy as np
from typing import Dict, List, Optional

from src.config import Config, MultiTrackingRecord, TrackingRecord
from src.utils.utils import color_for_track_id, stable_norm
from src.utils.plotting import draw_box_xywh
from src.utils.preprocessing import load_simple_gt_xywh_per_frame,load_visdrone_gt_boxes
from src.tracking import match_yolo_track_to_gt_target_visdrone

# ============================================================
# Selected-track extraction
# ============================================================

def extract_selected_track_records(
    cfg: Config,
    multi_records: List[MultiTrackingRecord],
    selected_track_id: int,
    gt_path: str,
    gt_format: str
) -> List[TrackingRecord]:
    gt_simple = None
    gt_visdrone = None
    matched_gt_target_id = None

    if gt_format == "simple_xywh_per_frame":
        gt_simple = load_simple_gt_xywh_per_frame(gt_path)

    elif gt_format == "visdrone_mot":
        gt_visdrone = load_visdrone_gt_boxes(gt_path)
        matched_gt_target_id = match_yolo_track_to_gt_target_visdrone(
            multi_records=multi_records,
            gt_by_frame=gt_visdrone,
            selected_track_id=selected_track_id,
        )
    else:
        raise ValueError(f"Unsupported gt_format: {gt_format}")

    records: List[TrackingRecord] = []
    prev_tracker_center: Optional[np.ndarray] = None
    prev_gt_center: Optional[np.ndarray] = None
    track_started = False

    for rec in multi_records:
        chosen_det = None
        for det in rec.detections:
            if det.track_id == selected_track_id:
                chosen_det = det
                break

        tracker_box = None
        tracker_center = None
        tracker_conf = None
        track_id = None

        if chosen_det is not None:
            tracker_box = chosen_det.box_xywh.copy()
            tracker_center = chosen_det.center.copy()
            tracker_conf = chosen_det.conf
            track_id = chosen_det.track_id
            track_started = True

        gt_box = None
        gt_center = None

        if gt_format == "simple_xywh_per_frame":
            if gt_simple is not None and rec.frame_idx in gt_simple:
                gt_box = gt_simple[rec.frame_idx].copy()
                gt_center = gt_box[:2].copy()

        elif gt_format == "visdrone_mot":
            if (
                gt_visdrone is not None
                and matched_gt_target_id is not None
                and rec.frame_idx in gt_visdrone
                and matched_gt_target_id in gt_visdrone[rec.frame_idx]
            ):
                gt_box = gt_visdrone[rec.frame_idx][matched_gt_target_id].copy()
                gt_center = gt_box[:2].copy()

        sensitivity_proxy = None

        if not track_started and gt_center is None:
            records.append(
                TrackingRecord(
                    frame_idx=rec.frame_idx,
                    frame_bgr=rec.frame_bgr.copy(),
                    tracker_box_xywh=None,
                    tracker_center=None,
                    tracker_conf=None,
                    gt_center=None,
                    gt_box_xywh=None,
                    track_id=None,
                    sensitivity_proxy=None,
                )
            )
            continue

        if (
            tracker_center is not None
            and prev_tracker_center is not None
            and gt_center is not None
            and prev_gt_center is not None
        ):
            numerator = stable_norm(tracker_center - prev_tracker_center)
            denominator = max(stable_norm(gt_center - prev_gt_center) + cfg.eta, 1.0)
            sensitivity_proxy = numerator / denominator

        records.append(
            TrackingRecord(
                frame_idx=rec.frame_idx,
                frame_bgr=rec.frame_bgr.copy(),
                tracker_box_xywh=tracker_box,
                tracker_center=tracker_center,
                tracker_conf=tracker_conf,
                gt_center=gt_center.copy() if gt_center is not None else None,
                gt_box_xywh=gt_box.copy() if gt_box is not None else None,
                track_id=track_id,
                sensitivity_proxy=sensitivity_proxy,
            )
        )

        if tracker_center is not None:
            prev_tracker_center = tracker_center.copy()
        if gt_center is not None:
            prev_gt_center = gt_center.copy()

    return records

# ============================================================
# Combined video visualization
# ============================================================

def build_selected_records_map(
    cfg: Config,
    multi_records: List[MultiTrackingRecord],
    selected_track_ids: List[int],
    gt_path: str,
    gt_format: str
) -> Dict[int, List[TrackingRecord]]:
    return {
        tid: extract_selected_track_records(cfg, multi_records, tid, gt_path, gt_format)
        for tid in selected_track_ids
    }

def visualize_all_selected_tracks_same_video(
    cfg: Config,
    multi_records: List[MultiTrackingRecord],
    selected_records_map: Dict[int, List[TrackingRecord]],
    video_output_dir: str,
) -> None:
    if len(multi_records) == 0:
        return

    h, w = multi_records[0].frame_bgr.shape[:2]
    out_path = os.path.join(video_output_dir, cfg.combined_selected_video_name)

    writer = None
    if cfg.save_selected_video:
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (w, h),
        )

    rec_lookup = {}
    for tid, recs in selected_records_map.items():
        rec_lookup[tid] = {r.frame_idx: r for r in recs}

    paused = False
    i = 0
    frame_count = len(multi_records)

    print("[INFO] Controls: q quit | p pause/resume | n next frame when paused")

    while i < frame_count:
        base = multi_records[i].frame_bgr.copy()

        for tid, frame_map in rec_lookup.items():
            if i not in frame_map:
                continue
            rec = frame_map[i]
            color = color_for_track_id(tid)

            if rec.tracker_box_xywh is not None:
                base = draw_box_xywh(base, rec.tracker_box_xywh, color, f"YOLO ID {tid}")

            if cfg.draw_gt_on_video and rec.gt_box_xywh is not None:
                gt_color = tuple(int(max(0, c - 80)) for c in color)
                base = draw_box_xywh(base, rec.gt_box_xywh, gt_color, f"GT {tid}")

        cv2.putText(
            base,
            f"Frame {i}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            base,
            f"Selected IDs: {list(selected_records_map.keys())}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if writer is not None:
            writer.write(base)

        if cfg.show_video_window:
            cv2.imshow("Selected Tracks Combined", base)
            key = cv2.waitKey(0 if paused else cfg.window_delay_ms) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = not paused
                continue
            elif key == ord("n") and paused:
                i += 1
                continue

        i += 1

    if writer is not None:
        writer.release()
        print(f"Saved combined selected-tracks video to: {out_path}")

    if cfg.show_video_window:
        cv2.destroyAllWindows()
