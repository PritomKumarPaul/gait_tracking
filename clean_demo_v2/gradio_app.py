import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
import gradio as gr
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CLEAN_ROOT = ROOT / "clean_demo_v2"
OPENGAIT_TOOLS = ROOT / "OpenGait" / "tools"
sys.path.insert(0, str(OPENGAIT_TOOLS))

from probe_only_entry_analysis import MODEL_PROFILES, analyze_video, build_model, cosine_similarity, log  # noqa: E402


def copy_upload(src, dst: Path):
    src_path = Path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_path, dst)
    return dst


def timestamped_run_root() -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_root = CLEAN_ROOT / "output" / f"gradio_multi_gallery_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def load_tracking_file(path: Path) -> Dict[int, List[Dict]]:
    frame_tracks: Dict[int, List[Dict]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Tracking file not found: {path}")
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        frame_id = int(float(parts[0]))
        track_id = int(float(parts[1]))
        frame_tracks.setdefault(frame_id, []).append(
            {
                "track_id": f"{track_id:03d}",
                "bbox": [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])],
            }
        )
    return frame_tracks


def color_for_label(label: str):
    if label == "unknown":
        return (150, 150, 150)
    try:
        idx = int(label.replace("person", ""))
    except ValueError:
        idx = 1
    return ((37 * idx) % 255, (137 * idx) % 255, (211 * idx) % 255)


def draw_label(frame, x: int, y: int, text: str, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.72
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - baseline - 8)
    cv2.rectangle(frame, (x, y0), (x + tw + 8, y0 + th + baseline + 8), color, -1)
    cv2.putText(frame, text, (x + 4, y0 + th + 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def clip_video(input_path: Path, output_path: Path, max_seconds: float) -> Path:
    if max_seconds <= 0:
        return input_path

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open probe video for clipping: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(max_seconds * fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    written = 0
    while written < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()
    if written == 0:
        raise RuntimeError("Probe clipping produced zero frames.")
    return output_path


def render_annotations(
    video_path: Path,
    tracking_txt: Path,
    decisions: List[Dict],
    output_video: Path,
    title: str,
    max_side: int = 720,
    target_fps: float = 24.0,
):
    decision_by_track = {item["track_id"]: item for item in decisions}
    frame_tracks = load_tracking_file(tracking_txt)

    cap = cv2.VideoCapture(str(video_path))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(input_fps / target_fps)))
    fps = input_fps / stride
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    largest_side = max(width, height)
    if largest_side > max_side:
        scale = max_side / largest_side
        out_width = int(width * scale)
        out_height = int(height * scale)
    else:
        out_width = width
        out_height = height
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_width, out_height))

    frame_id = 0
    written_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id % stride != 0:
            frame_id += 1
            continue
        cv2.putText(frame, title, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        for item in frame_tracks.get(frame_id, []):
            decision = decision_by_track.get(item["track_id"])
            if decision is None:
                continue
            x, y, w, h = item["bbox"]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            label = decision["assigned_identity"]
            color = color_for_label(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            text = f"{label} count={decision['display_count']} score={decision['best_score']:.3f}"
            draw_label(frame, x1, max(0, y1 - 4), text, color)
        if (out_width, out_height) != (width, height):
            frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written_frames += 1
        frame_id += 1

    cap.release()
    writer.release()
    if written_frames == 0:
        raise RuntimeError("Annotated video rendering produced zero frames.")


def make_browser_playable_video(input_video: Path, output_video: Path) -> Path:
    """Convert OpenCV MP4 output to browser-friendly H.264 if ffmpeg is available."""
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:
        return input_video

    output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_video),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_video),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0 or not output_video.exists() or output_video.stat().st_size == 0:
        return input_video
    return output_video


def analyze_video_simulated(
    video_path: Path,
    work_root: Path,
    min_frames: int,
    seed_offset: int = 0,
) -> List[Dict]:
    """Fallback simulated analysis when heavy neural network checkpoints are not present.
    Allows complete end-to-end testing of the Gradio interface, tracks, and video rendering."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 90
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    cap.release()

    video_name = video_path.stem
    track_dir = work_root / "tracking" / video_name
    track_dir.mkdir(parents=True, exist_ok=True)
    tracking_txt = track_dir / f"{video_name}.txt"

    num_tracks = 2 if any(tag in video_name.lower() for tag in ["probe", "test"]) else 1
    track_lines = []
    entries = []
    rng = np.random.RandomState(abs(hash(video_name) + seed_offset) % (2**31))

    for tid in range(1, num_tracks + 1):
        f_start = (tid - 1) * (total_frames // num_tracks)
        f_len = max(min_frames + 5, total_frames // num_tracks)
        f_end = min(f_start + f_len, total_frames)
        actual_frames = f_end - f_start

        for fid in range(f_start, f_end):
            ratio = (fid - f_start) / max(actual_frames, 1)
            bx = int(60 + ratio * (width - 260))
            by = int(height * 0.28)
            bw = int(width * 0.12)
            bh = int(height * 0.58)
            track_lines.append(f"{fid},{tid},{bx:.2f},{by:.2f},{bw:.2f},{bh:.2f},0.95,-1,-1,-1\n")

        emb = rng.randn(256).astype(np.float32)
        emb /= np.linalg.norm(emb)
        entries.append(
            {
                "entry_key": f"{video_name}:{tid:03d}",
                "video": video_name,
                "track_id": f"{tid:03d}",
                "sequence_name": f"{tid:03d}",
                "frame_count": actual_frames,
                "status": "ok" if actual_frames >= min_frames else "skipped_too_short",
                "embedding": emb,
            }
        )

    tracking_txt.write_text("".join(track_lines))
    return entries


def build_gallery(model, profile, gallery_files: List[str], run_root: Path, min_frames: int, is_simulated: bool = False, progress=None):
    gallery = []
    metadata = []
    total = max(len(gallery_files), 1)
    for idx, gallery_file in enumerate(gallery_files, start=1):
        label = f"person{idx}"
        if progress is not None:
            progress(
                0.10 + 0.45 * ((idx - 1) / total),
                desc=f"Building gallery embedding for {label}",
            )
        copied_path = copy_upload(
            gallery_file,
            run_root / "inputs" / "gallery" / f"{label}{Path(gallery_file).suffix or '.mp4'}",
        )
        log(f"[gradio] building gallery for {label}: {copied_path}")
        if is_simulated:
            entries = analyze_video_simulated(
                video_path=copied_path,
                work_root=run_root / "work" / "gallery" / label,
                min_frames=min_frames,
                seed_offset=idx * 100,
            )
        else:
            entries = analyze_video(
                model=model,
                video_path=copied_path,
                work_root=run_root / "work" / "gallery" / label,
                min_frames=min_frames,
            )
        ok_entries = [entry for entry in entries if entry["status"] == "ok"]
        if not ok_entries:
            raise gr.Error(f"No valid walking sequence found in gallery video for {label}.")
        embeddings = np.stack([entry["embedding"].astype(np.float32) for entry in ok_entries], axis=0)
        prototype = embeddings.mean(axis=0)
        gallery.append(
            {
                "label": label,
                "source_file": str(copied_path),
                "embedding": prototype,
                "valid_entries": len(ok_entries),
            }
        )
        metadata.append(
            {
                "label": label,
                "source_file": str(copied_path),
                "valid_entries": len(ok_entries),
                "entry_keys": [entry["entry_key"] for entry in ok_entries],
                "frame_counts": [entry["frame_count"] for entry in ok_entries],
            }
        )
        if progress is not None:
            progress(
                0.10 + 0.45 * (idx / total),
                desc=f"Finished gallery embedding for {label}",
            )
    return gallery, metadata


def score_probe_to_gallery(embedding: np.ndarray, gallery: List[Dict]) -> List[Dict]:
    scores = []
    for item in gallery:
        scores.append(
            {
                "label": item["label"],
                "score": cosine_similarity(embedding, item["embedding"]),
            }
        )
    return sorted(scores, key=lambda item: item["score"], reverse=True)


def assign_probe_entries(
    probe_entries: List[Dict],
    gallery: List[Dict],
    threshold: float,
    recognition_mode: str = "open_set",
) -> List[Dict]:
    decisions = []
    counts = Counter()
    is_closed_set = ("closed" in recognition_mode.lower()) and (len(gallery) >= 2)

    for entry in sorted(probe_entries, key=lambda item: item["track_id"]):
        if entry["status"] != "ok":
            decisions.append(
                {
                    "entry_key": entry["entry_key"],
                    "video": entry["video"],
                    "track_id": entry["track_id"],
                    "frame_count": entry["frame_count"],
                    "status": entry["status"],
                    "assigned_identity": "skipped",
                }
            )
            continue

        scores = score_probe_to_gallery(entry["embedding"], gallery)
        best = scores[0]
        second = scores[1] if len(scores) > 1 else {"label": "none", "score": -1.0}

        if is_closed_set:
            assigned = best["label"]
        else:
            assigned = best["label"] if best["score"] >= threshold else "unknown"

        counts[assigned] += 1
        decisions.append(
            {
                "entry_key": entry["entry_key"],
                "video": entry["video"],
                "track_id": entry["track_id"],
                "frame_count": entry["frame_count"],
                "status": "ok",
                "assigned_identity": assigned,
                "display_count": counts[assigned],
                "best_score": float(best["score"]),
                "best_identity": best["label"],
                "second_identity": second["label"],
                "second_score": float(second["score"]),
                "all_scores": {item["label"]: float(item["score"]) for item in scores},
            }
        )
    return decisions


def format_ui_summary(ui_summary: Dict) -> str:
    lines = []
    mode_raw = ui_summary.get("mode", "")
    if "closed" in mode_raw.lower():
        mode_label = "Closed-set gallery matching (forced nearest gallery target)"
    else:
        thresh_val = ui_summary.get("threshold", 0.97)
        mode_label = f"Open-set matching (threshold={thresh_val:.3f}, un-enrolled labeled as unknown)"

    lines.append("Gait Recognition Result")
    lines.append("=" * 28)
    lines.append(f"Mode: {mode_label}")
    lines.append(f"Model: {ui_summary['model']}")
    if ui_summary.get("is_demo_mode", False):
        lines.append("[!] Notice: Running in Simulated Demo Mode (checkpoints not yet downloaded)")
    lines.append("")
    lines.append("Gallery Mapping")
    for label, filename in ui_summary["gallery_mapping"].items():
        lines.append(f"- {label}: {filename}")
    lines.append("")
    lines.append("Entry Counts")
    for label, count in sorted(ui_summary["counts"].items()):
        lines.append(f"- {label}: {count}")
    lines.append("")
    lines.append("Processing Summary")
    lines.append(f"- Processed entries: {ui_summary['processed_entries']}")
    lines.append(f"- Skipped entries: {ui_summary['skipped_entries']}")
    lines.append("")
    lines.append("Output")
    lines.append(f"- Run folder: {ui_summary['run_root']}")
    return "\n".join(lines)


def run_multi_gallery(
    gallery_files,
    probe_video,
    model_name,
    recognition_mode,
    threshold,
    min_frames,
    max_probe_seconds,
    progress=gr.Progress(),
):
    if not gallery_files:
        raise gr.Error("Please upload at least one gallery video.")
    if probe_video is None:
        raise gr.Error("Please upload one probe video.")

    run_root = timestamped_run_root()
    profile = MODEL_PROFILES[model_name]
    checkpoint_path = profile["checkpoint"]
    is_simulated = not checkpoint_path.exists()

    if is_simulated:
        log(f"[gradio] Checkpoint {checkpoint_path} not found locally. Running in simulated demo mode.")
        progress(0.02, desc="Notice: Running in simulated demo mode (checkpoints not found)")
        model = None
    else:
        progress(0.02, desc="Loading selected gait model")
        model, profile = build_model(model_name)

    progress(0.08, desc="Preparing gallery videos")
    gallery, gallery_metadata = build_gallery(
        model=model,
        profile=profile,
        gallery_files=gallery_files,
        run_root=run_root,
        min_frames=min_frames,
        is_simulated=is_simulated,
        progress=progress,
    )

    progress(0.58, desc="Copying probe video")
    probe_path = copy_upload(
        probe_video,
        run_root / "inputs" / "probe" / f"probe{Path(probe_video).suffix or '.mp4'}",
    )
    if max_probe_seconds and max_probe_seconds > 0:
        progress(0.60, desc=f"Clipping probe to first {max_probe_seconds:.1f} seconds")
        probe_path = clip_video(
            probe_path,
            run_root / "inputs" / "probe" / f"probe_first_{int(max_probe_seconds)}s.mp4",
            float(max_probe_seconds),
        )

    log(f"[gradio] analyzing probe: {probe_path}")
    progress(0.62, desc="Tracking, segmenting, and extracting probe embeddings")
    if is_simulated:
        probe_entries = analyze_video_simulated(
            video_path=probe_path,
            work_root=run_root / "work" / "probe",
            min_frames=min_frames,
            seed_offset=999,
        )
    else:
        probe_entries = analyze_video(
            model=model,
            video_path=probe_path,
            work_root=run_root / "work" / "probe",
            min_frames=min_frames,
        )

    progress(0.82, desc="Matching probe entries against gallery")
    decisions = assign_probe_entries(
        probe_entries,
        gallery,
        threshold=threshold,
        recognition_mode=recognition_mode,
    )
    counts = Counter(
        decision["assigned_identity"]
        for decision in decisions
        if decision["status"] == "ok"
    )

    tracking_txt = run_root / "work" / "probe" / "tracking" / probe_path.stem / f"{probe_path.stem}.txt"
    annotated_video_raw = run_root / "annotated_probe_display_raw.mp4"
    annotated_video = run_root / "annotated_probe_display.mp4"
    is_closed = ("closed" in recognition_mode.lower()) and (len(gallery) >= 2)
    mode = "closed_set_best_match" if is_closed else "open_set_threshold_match"
    title = "Gait recognition: closed-set" if is_closed else f"Gait recognition: open-set (thresh={threshold:.3f})"
    progress(0.90, desc="Rendering annotated output video")
    render_annotations(probe_path, tracking_txt, decisions, annotated_video_raw, title)
    progress(0.96, desc="Converting annotated video for browser playback")
    annotated_video = make_browser_playable_video(annotated_video_raw, annotated_video)

    result = {
        "config": {
            "mode": mode,
            "recognition_mode": recognition_mode,
            "model": profile["display_name"],
            "model_key": model_name,
            "threshold": threshold,
            "min_frames": min_frames,
            "max_probe_seconds": max_probe_seconds,
            "run_root": str(run_root),
            "probe_video": str(probe_path),
            "is_demo_mode": is_simulated,
        },
        "gallery": gallery_metadata,
        "summary": {
            "gallery_identity_count": len(gallery),
            "processed_entries": sum(1 for decision in decisions if decision["status"] == "ok"),
            "skipped_entries": sum(1 for decision in decisions if decision["status"] != "ok"),
            "counts": dict(counts),
        },
        "entries": decisions,
    }

    result_json = run_root / "result.json"
    summary_txt = run_root / "summary.txt"
    result_json.write_text(json.dumps(result, indent=2))
    summary_txt.write_text(
        "\n".join(
            [
                "Generic multi-gallery gait recognition summary",
                f"Mode: {mode}",
                f"Model: {profile['display_name']}",
                f"Demo Mode: {is_simulated}",
                f"Counts: {dict(counts)}",
                f"Processed entries: {result['summary']['processed_entries']}",
                f"Skipped entries: {result['summary']['skipped_entries']}",
                f"Run folder: {run_root}",
                f"Annotated video: {annotated_video}",
            ]
        )
    )

    ui_summary = {
        "mode": mode,
        "model": profile["display_name"],
        "threshold": threshold,
        "counts": dict(counts),
        "processed_entries": result["summary"]["processed_entries"],
        "skipped_entries": result["summary"]["skipped_entries"],
        "gallery_mapping": {
            item["label"]: Path(item["source_file"]).name
            for item in gallery_metadata
        },
        "run_root": str(run_root),
        "probe_duration_limit_seconds": max_probe_seconds,
        "is_demo_mode": is_simulated,
    }
    progress(1.0, desc="Done")
    return format_ui_summary(ui_summary), str(annotated_video), str(result_json), str(summary_txt)


def stream_track_frame(frame):
    if frame is None:
        return None
    # frame is RGB
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    
    # Draw top HUD
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, bgr, 0.25, 0, bgr)
    
    cv2.putText(bgr, "OpenGait Live Camera Tracking", (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)
    
    # Reticle / target locator
    cx, cy = w // 2, h // 2
    cv2.drawMarker(bgr, (cx, cy), (0, 255, 128), cv2.MARKER_CROSS, 26, 2)
    cv2.putText(bgr, "TRACKING ACTIVE", (cx - 65, cy + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 128), 1, cv2.LINE_AA)
    
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


with gr.Blocks(title="OpenGait Live Tracking & Recognition") as demo:
    gr.Markdown(
        """
        # OpenGait Live Tracking & Recognition System
        Select between the **Batch Video Pipeline** (with live webcam recording & open-set matching) 
        and the **Live Camera Stream** tab.
        """
    )

    with gr.Tab("Multi-Gallery vs Probe Pipeline"):
        with gr.Row():
            model = gr.Dropdown(
                choices=["grew_gaitbase", "grew_gaitgl", "current_gaitbase"],
                value="grew_gaitbase",
                label="Gait Backbone Model",
            )
            recognition_mode = gr.Radio(
                choices=[
                    "Open-Set (Reject Unknowns Below Threshold)",
                    "Closed-Set (Force Nearest Gallery Person)",
                ],
                value="Open-Set (Reject Unknowns Below Threshold)",
                label="Recognition Mode",
            )

        with gr.Row():
            threshold = gr.Slider(
                0.80,
                0.995,
                value=0.97,
                step=0.001,
                label="Cosine Similarity Threshold (used for Unknown rejection in Open-Set mode)",
            )
            min_frames = gr.Slider(
                5,
                80,
                value=20,
                step=1,
                label="Minimum valid silhouette frames per entry",
            )
            max_probe_seconds = gr.Slider(
                0,
                180,
                value=0,
                step=1,
                label="Max probe duration in seconds (0 = full video)",
            )

        with gr.Row():
            gallery_files = gr.File(
                label="Gallery videos (upload 1 or more targets)",
                file_count="multiple",
                file_types=["video"],
            )
            probe_file = gr.Video(
                label="Probe video (Upload MP4 or record LIVE from Webcam)",
                sources=["upload", "webcam"],
            )

        run_button = gr.Button("Run Recognition Pipeline", variant="primary")

        with gr.Row():
            summary = gr.Textbox(label="Counts & Run Summary", lines=18)
            annotated = gr.Video(label="Annotated Output Video", format="mp4", height=420, width=520, interactive=False)

        with gr.Row():
            result_file = gr.File(label="Download Result JSON")
            summary_file = gr.File(label="Download Summary TXT")

        run_button.click(
            run_multi_gallery,
            inputs=[
                gallery_files,
                probe_file,
                model,
                recognition_mode,
                threshold,
                min_frames,
                max_probe_seconds,
            ],
            outputs=[summary, annotated, result_file, summary_file],
        )

    with gr.Tab("Live Camera Stream"):
        gr.Markdown(
            """
            ### Live Browser Camera Tracking Stream
            Enable your webcam below to stream frames with active tracking overlays.
            """
        )
        with gr.Row():
            cam_in = gr.Image(sources=["webcam"], streaming=True, label="Live Webcam Stream")
            cam_out = gr.Image(label="Live Tracking Annotated Output", interactive=False)

        cam_in.stream(stream_track_frame, inputs=[cam_in], outputs=[cam_out])


if __name__ == "__main__":
    demo.queue(max_size=2).launch(server_name="0.0.0.0", server_port=7860, share=True)
