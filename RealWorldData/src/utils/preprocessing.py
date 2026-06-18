
from typing import Dict
import numpy as np

# ============================================================
# GT loaders
# ============================================================

def load_simple_gt_xywh_per_frame(gt_path: str) -> Dict[int, np.ndarray]:
    gt_by_frame: Dict[int, np.ndarray] = {}
    with open(gt_path, "r") as f:
        for frame_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            line = line.replace("\t", ",").replace(" ", ",")
            parts = [p for p in line.split(",") if p != ""]
            if len(parts) < 4:
                raise ValueError(f"Could not parse GT line: {line}")
            x, y, w, h = map(float, parts[:4])
            cx = x + w / 2.0
            cy = y + h / 2.0
            gt_by_frame[frame_idx] = np.array([cx, cy, w, h], dtype=float)
    return gt_by_frame

def load_visdrone_gt_boxes(gt_path: str) -> Dict[int, Dict[int, np.ndarray]]:
    gt: Dict[int, Dict[int, np.ndarray]] = {}
    with open(gt_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 10:
                print(f"[WARNING] Skipping malformed GT line {line_num}: {line}")
                continue

            try:
                frame_id = int(float(parts[0]))
                target_id = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])

                cx = x + w / 2.0
                cy = y + h / 2.0
                frame_idx = frame_id - 1

                if frame_idx not in gt:
                    gt[frame_idx] = {}
                gt[frame_idx][target_id] = np.array([cx, cy, w, h], dtype=float)

            except Exception as e:
                print(f"[WARNING] Failed parsing GT line {line_num}: {line}")
                print(e)

    return gt
