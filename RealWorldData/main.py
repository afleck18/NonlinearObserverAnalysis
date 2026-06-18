from typing import Dict, List

from src.config import Config
from src.observers import run_observers
from src.tracking import (
    choose_default_track_ids,
    run_tracking_and_extract_all_tracks,
    build_track_summary,
    save_track_summary_csv
)
from src.utils.utils import (
    ensure_dir,
    get_video_output_dir,
    print_candidate_frame_windows
)
from src.utils.plotting import (
    build_four_panel_figure,
    build_five_panel_figure,
    save_contact_sheet_for_track
)
from src.utils.video_visualization import (
    build_selected_records_map, 
    visualize_all_selected_tracks_same_video
)

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    cfg = Config()
    ensure_dir(cfg.output_dir)

    all_track_data: List[Dict] = []
    all_pixel_metrics_rows: List[Dict] = []

    for video_job in cfg.video_jobs:
        print(f"\n========== Processing video: {video_job.name} ==========")
        video_output_dir = get_video_output_dir(cfg, video_job.name)

        print("Running tracking and collecting all tracked objects...")
        multi_records = run_tracking_and_extract_all_tracks(cfg, video_job.video_path, video_job.model_path)

        print("Building track summary...")
        summary_rows = build_track_summary(multi_records)
        save_track_summary_csv(video_output_dir, summary_rows)

        print("\nTop available track IDs:")
        for row in summary_rows[:20]:
            print(row)

        selected_track_ids = video_job.selected_track_ids
        if selected_track_ids is None or len(selected_track_ids) == 0:
            if cfg.auto_select_if_empty:
                selected_track_ids = choose_default_track_ids(cfg, summary_rows)
                print(f"\n[INFO] No selected_track_ids provided. Auto-selected: {selected_track_ids}")
            else:
                raise RuntimeError(f"No selected_track_ids provided for video {video_job.name}.")

        print(f"\n[INFO] Selected track IDs for {video_job.name}: {selected_track_ids}")

        print("\nExtracting selected tracks...")
        selected_records_map = build_selected_records_map(
            cfg=cfg,
            multi_records=multi_records,
            selected_track_ids=selected_track_ids,
            gt_path=video_job.gt_path,
            gt_format=video_job.gt_format
        )

        for selected_track_id in selected_track_ids:
            print_candidate_frame_windows(
                selected_records_map[selected_track_id],
                selected_track_id,
                num_examples=cfg.contact_sheet_num_examples,
            )

        print("Visualizing all selected track IDs in the same video...")
        visualize_all_selected_tracks_same_video(
            cfg=cfg,
            multi_records=multi_records,
            selected_records_map=selected_records_map,
            video_output_dir=video_output_dir,
        )

        for selected_track_id in selected_track_ids:
            print(f"\n[INFO] Processing track ID {selected_track_id} in {video_job.name}...")

            records = selected_records_map[selected_track_id]

            print("Running fixed-gain and normalized observers...")
            outputs = run_observers(cfg, records)

            if cfg.save_contact_sheets:
                save_contact_sheet_for_track(
                    cfg=cfg,
                    records=records,
                    outputs=outputs,
                    selected_track_id=selected_track_id,
                    video_output_dir=video_output_dir,
                )

            print("Building five-panel figure...")
            build_four_panel_figure(
                cfg=cfg,
                records=records,
                outputs=outputs,
                selected_track_id=selected_track_id,
                video_output_dir=video_output_dir,
                per_track_representative_frames=video_job.per_track_representative_frames,
            )

            metrics = build_five_panel_figure(
                cfg=cfg,
                records=records,
                outputs=outputs,
                selected_track_id=selected_track_id,
                video_output_dir=video_output_dir,
                per_track_representative_frames=video_job.per_track_representative_frames,
            )
          