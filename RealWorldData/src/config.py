from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

@dataclass
class VideoJob:
    name: str
    video_path: str
    model_path: str
    gt_path: str
    gt_format: str
    selected_track_ids: Optional[List[int]] = None
    per_track_representative_frames: Dict[int, int] = field(default_factory=dict)


@dataclass
class Config:
    output_dir: str = "RealWorldData/output"

    # Leave selected track ids empty for automatic selection
    video_jobs: List[VideoJob] = field(default_factory=lambda: [
        VideoJob(
            name="uav",
            video_path="RealWorldData/sample_data/uav/uav0000279_00001_v.mov",
            model_path="RealWorldData/src/models/standard_model.pt",
            gt_path="RealWorldData/sample_data/uav/uav0000279_00001_v.txt",
            gt_format="visdrone_mot",
            selected_track_ids=[177],
            per_track_representative_frames={177:231},
        ),
        VideoJob(
            name="underwater",
            video_path="RealWorldData/sample_data/underwater/Dolphin1.mp4",
            model_path="RealWorldData/src/models/dolphin_model.pt",
            gt_path="RealWorldData/sample_data/underwater/groundtruth_rect.txt",
            gt_format="simple_xywh_per_frame",
            selected_track_ids=[1],
            per_track_representative_frames={},
        )
    ])

    tracker_yaml: str = "botsort.yaml"
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    device: str = "mps"

    auto_select_if_empty: bool = True
    auto_select_top_k: int = 8
    min_track_length: int = 20

    # Best-track paper selection
    min_track_length_for_ranking: int = 40
    top_k_for_table: int = 5

    show_video_window: bool = True
    save_selected_video: bool = True
    combined_selected_video_name: str = "selected_tracks_combined.mp4"
    window_delay_ms: int = 30
    draw_gt_on_video: bool = True

    use_tight_crop_panel_a: bool = True
    crop_padding_px: int = 80
    crop_min_size_px: int = 220

    frame_selection_mode: str = "manual"   # "auto" or "manual"
    representative_frame_index: Optional[int] = None

    save_contact_sheets: bool = False
    contact_sheet_num_examples: int = 8
    contact_sheet_columns: int = 4
    contact_sheet_max_rows: int = 2

    # Observer
    F: np.ndarray = field(default_factory=lambda: np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float))

    H: np.ndarray = field(default_factory=lambda: np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=float))

    K_tilde: np.ndarray = field(default_factory=lambda: np.array([
        [0.7, 0.0],
        [0.0, 0.7],
        [0.2, 0.0],
        [0.0, 0.2],
    ], dtype=float))

    alpha_fixed: float = 0.5
    beta: float = 0.8
    epsilon: float = 5e-2
    eta: float = 1e-6

    rolling_window: int = 25

    # Video-level normalization:
    # "median_sqrt_area" or "median_diag"
    video_normalization_mode: str = "median_sqrt_area"

    max_missing_frames_before_reset: int = 10

    bootstrap_reps: int = 10000
    bootstrap_seed: int = 0


# ============================================================
# Data structures
# ============================================================

@dataclass
class FrameDetection:
    track_id: int
    box_xywh: np.ndarray
    center: np.ndarray
    conf: float


@dataclass
class MultiTrackingRecord:
    frame_idx: int
    frame_bgr: np.ndarray
    detections: List[FrameDetection]


@dataclass
class TrackingRecord:
    frame_idx: int
    frame_bgr: np.ndarray
    tracker_box_xywh: Optional[np.ndarray]
    tracker_center: Optional[np.ndarray]
    tracker_conf: Optional[float]
    gt_center: Optional[np.ndarray]
    gt_box_xywh: Optional[np.ndarray]
    track_id: Optional[int]
    sensitivity_proxy: Optional[float]


@dataclass
class ObserverOutputs:
    fixed_centers: List[Optional[np.ndarray]]
    norm_centers: List[Optional[np.ndarray]]
    fixed_errors: List[float]
    norm_errors: List[float]
    fixed_gamma_proxy: List[float]
    norm_gamma_proxy: List[float]
    sensitivity_proxy: List[float]
    valid_frame_indices: List[int]
