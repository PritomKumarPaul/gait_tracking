import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent

sys.path.append(str(REPO_ROOT / "demo" / "libs"))
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "opengait"))

try:
    from demo.libs.track import track  # noqa: E402
    from demo.libs.segment import seg  # noqa: E402
    import demo.libs.model.baselineDemo as baseline_demo  # noqa: E402
    from opengait.utils import config_loader  # noqa: E402
    from opengait.modeling import models  # noqa: E402
except Exception:
    track = None
    seg = None
    baseline_demo = None
    config_loader = None
    models = None


def log(message: str) -> None:
    print(message, flush=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return -1.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def get_track_id(inputs) -> str:
    return str(inputs[2][0])


def get_sequence_name(inputs) -> str:
    return str(inputs[3][0])


def get_num_frames(inputs) -> int:
    seq_len = inputs[4]
    if seq_len is None:
        return 0
    return int(seq_len[0][0])


def build_opengait_model(cfgs: Dict, checkpoint_path: Path):
    model_name = cfgs["model_cfg"]["model"]
    model_cls = getattr(models, model_name)
    model = model_cls.__new__(model_cls)
    torch.nn.Module.__init__(model)
    model.cfgs = cfgs
    model.engine_cfg = cfgs["evaluator_cfg"]
    model.iteration = 0
    model.msg_mgr = None
    model.build_network(cfgs["model_cfg"])
    checkpoint = torch.load(str(checkpoint_path), map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(checkpoint["model"], strict=cfgs["evaluator_cfg"]["restore_ckpt_strict"])
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    model._expects_numeric_labels_for_inference = True
    return model


def build_gaitbase_demo_model(cfg_path: Path, checkpoint_path: Path):
    baseline_demo.model_cfgs["gait_model"] = str(checkpoint_path)
    model = baseline_demo.BaselineDemo(config_loader(str(cfg_path)), training=False)
    model.requires_grad_(False)
    model.eval()
    model._expects_numeric_labels_for_inference = False
    return model


MODEL_PROFILES = {
    "grew_gaitbase": {
        "display_name": "GREW GaitBase",
        "loader": "baseline_demo",
        "config": REPO_ROOT / "configs" / "gaitbase" / "gaitbase_da_gait3d.yaml",
        "checkpoint": REPO_ROOT / "demo" / "checkpoints" / "gait_model" / "GREW" / "Baseline" / "GaitBase_DA" / "checkpoints" / "GaitBase_DA-180000.pt",
        "archive_hint": REPO_ROOT / "demo" / "checkpoints" / "gait_model" / "pretrained_grew_gaitbase.zip",
        "dataset_name": "GREW",
    },
    "current_gaitbase": {
        "display_name": "Current Local GaitBase",
        "loader": "baseline_demo",
        "config": REPO_ROOT / "configs" / "gaitbase" / "gaitbase_da_gait3d.yaml",
        "checkpoint": REPO_ROOT / "demo" / "checkpoints" / "gait_model" / "GaitBase_DA-180000.pt",
        "dataset_name": "Gait3D",
    },
    "grew_gaitgl": {
        "display_name": "GREW GaitGL",
        "loader": "gaitgl",
        "config": REPO_ROOT / "configs" / "gaitgl" / "gaitgl_GREW.yaml",
        "checkpoint": REPO_ROOT / "demo" / "checkpoints" / "gait_model" / "GaitGL" / "checkpoints" / "GaitGL-250000.pt",
        "archive_hint": REPO_ROOT / "demo" / "checkpoints" / "gait_model" / "pretrained_grew_gaitgl.zip",
        "dataset_name": "GREW",
    },
}


def build_model(model_name: str):
    profile = MODEL_PROFILES[model_name]
    checkpoint_path = profile["checkpoint"]
    log(f"[model] preparing profile={model_name} ({profile['display_name']})")
    log(f"[model] checkpoint={checkpoint_path}")
    if not checkpoint_path.exists():
        archive_hint = profile.get("archive_hint")
        hint = f" Expected checkpoint: {checkpoint_path}."
        if archive_hint is not None:
            hint += f" If you downloaded the release asset, make sure {archive_hint.name} is extracted under demo/checkpoints/gait_model."
        raise FileNotFoundError(
            f"Selected model '{model_name}' is not ready locally.{hint}"
        )

    if profile["loader"] == "baseline_demo":
        log("[model] using BaselineDemo loader for GaitBase-compatible checkpoint")
        model = build_gaitbase_demo_model(profile["config"], checkpoint_path)
        log(f"[model] ready type={type(model).__name__}")
        return model, profile

    cfgs = config_loader(str(profile["config"]))
    cfgs["data_cfg"]["dataset_name"] = profile["dataset_name"]
    cfgs["data_cfg"]["test_dataset_name"] = profile["dataset_name"]
    cfgs["evaluator_cfg"]["enable_float16"] = False
    cfgs["evaluator_cfg"]["restore_ckpt_strict"] = True
    cfgs["evaluator_cfg"]["transform"] = [{"type": "BaseSilTransform"}]

    log(f"[model] using OpenGait loader for model={cfgs['model_cfg']['model']}")
    model = build_opengait_model(cfgs, checkpoint_path)
    model.requires_grad_(False)
    model.eval()
    log(f"[model] ready type={type(model).__name__}")
    return model, profile


def extract_embedding(model, inputs) -> np.ndarray:
    with torch.no_grad():
        if getattr(model, "_expects_numeric_labels_for_inference", False):
            prepared_inputs = (inputs[0], [0], inputs[2], inputs[3], inputs[4])
        else:
            prepared_inputs = inputs
        prepared = model.inputs_pretreament(prepared_inputs)
        retval = model.forward(prepared)
        if isinstance(retval, tuple):
            retval = retval[0]
        emb = retval["inference_feat"]["embeddings"].detach().cpu().numpy()[0]
    return emb


def analyze_video(
    model,
    video_path: Path,
    work_root: Path,
    min_frames: int,
) -> List[Dict]:
    video_name = video_path.stem
    video_output_dir = work_root / "tracking" / video_name
    sil_root = work_root / "silhouettes"
    video_output_dir.mkdir(parents=True, exist_ok=True)

    log(f"[video:{video_name}] starting")
    log(f"[video:{video_name}] tracking -> {video_output_dir}")
    track_result = track(str(video_path), str(video_output_dir))
    frame_hits = len(track_result)
    track_ids = sorted({int(item[0]) for values in track_result.values() for item in values}) if track_result else []
    log(f"[video:{video_name}] tracking done, frames_with_tracks={frame_hits}, unique_track_ids={track_ids}")

    log(f"[video:{video_name}] segmentation -> {sil_root}")
    sil_inputs = seg(str(video_path), track_result, str(sil_root))
    log(f"[video:{video_name}] segmentation done, candidate_entries={len(sil_inputs)}")

    entries = []
    for inputs in sil_inputs:
        frame_count = get_num_frames(inputs)
        track_id = get_track_id(inputs)
        sequence_name = get_sequence_name(inputs)
        log(f"[video:{video_name}] entry track_id={track_id} sequence={sequence_name} frames={frame_count}")
        if frame_count < min_frames:
            log(f"[video:{video_name}] skipping track_id={track_id} because frames<{min_frames}")
            entries.append(
                {
                    "entry_key": f"{video_name}:{track_id}",
                    "video": video_name,
                    "track_id": track_id,
                    "sequence_name": sequence_name,
                    "frame_count": frame_count,
                    "status": "skipped_too_short",
                }
            )
            continue

        log(f"[video:{video_name}] embedding track_id={track_id}")
        emb = extract_embedding(model, inputs)
        log(f"[video:{video_name}] embedding done track_id={track_id} shape={list(emb.shape)}")
        entries.append(
            {
                "entry_key": f"{video_name}:{track_id}",
                "video": video_name,
                "track_id": track_id,
                "sequence_name": sequence_name,
                "frame_count": frame_count,
                "status": "ok",
                "embedding": emb,
            }
        )
    log(f"[video:{video_name}] finished, kept_entries={sum(1 for entry in entries if entry['status'] == 'ok')}")
    return entries


def assign_identities(entries: List[Dict], threshold: float) -> Dict:
    registry: List[Dict] = []
    resolved_entries: List[Dict] = []

    for entry in entries:
        base = {
            "entry_key": entry["entry_key"],
            "video": entry["video"],
            "track_id": entry["track_id"],
            "sequence_name": entry["sequence_name"],
            "frame_count": entry["frame_count"],
            "status": entry["status"],
        }
        if entry["status"] != "ok":
            resolved_entries.append(base)
            continue

        embedding = entry["embedding"]
        best_match: Optional[Dict] = None
        best_score = -1.0
        for person in registry:
            score = cosine_similarity(embedding, person["prototype"])
            if score > best_score:
                best_score = score
                best_match = person

        if best_match is None or best_score < threshold:
            person_id = f"person_{len(registry) + 1:03d}"
            registry.append(
                {
                    "person_id": person_id,
                    "prototype": embedding.copy(),
                    "entry_keys": [entry["entry_key"]],
                }
            )
            assigned_person_id = person_id
            assigned_score = None
            matched_existing = False
            log(f"[match] new identity {assigned_person_id} for {entry['entry_key']}")
        else:
            assigned_person_id = best_match["person_id"]
            best_match["entry_keys"].append(entry["entry_key"])
            matched_existing = True
            assigned_score = best_score
            count = len(best_match["entry_keys"])
            best_match["prototype"] = (
                best_match["prototype"] * (count - 1) + embedding
            ) / count
            log(
                f"[match] matched {entry['entry_key']} -> {assigned_person_id} "
                f"(cos={assigned_score:.4f}, threshold={threshold:.4f})"
            )

        base.update(
            {
                "assigned_person_id": assigned_person_id,
                "matched_existing_person": matched_existing,
                "best_cosine_similarity": assigned_score,
            }
        )
        resolved_entries.append(base)

    people = []
    for person in registry:
        people.append(
            {
                "person_id": person["person_id"],
                "entry_count": len(person["entry_keys"]),
                "entry_keys": person["entry_keys"],
            }
        )

    return {
        "summary": {
            "processed_entries": sum(1 for entry in resolved_entries if entry["status"] == "ok"),
            "skipped_entries": sum(1 for entry in resolved_entries if entry["status"] != "ok"),
            "unique_people_estimated": len(people),
            "total_entries_estimated": sum(1 for entry in resolved_entries if entry["status"] == "ok"),
            "cosine_threshold": threshold,
        },
        "people": people,
        "entries": resolved_entries,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Probe-only gait entry analysis for a folder of videos."
    )
    parser.add_argument(
        "--video-dir",
        default=str(PROJECT_ROOT / "team2videos"),
        help="Folder containing input videos.",
    )
    parser.add_argument(
        "--video-path",
        default=None,
        help="Optional path to a single input video. Overrides --video-dir.",
    )
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "output" / "probe_only_entry_analysis.json"),
        help="Where to save the summary JSON.",
    )
    parser.add_argument(
        "--work-root",
        default=str(REPO_ROOT / "demo" / "output" / "probe_only"),
        help="Working folder for tracking videos and silhouettes.",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for matching a new entry to an existing person.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=20,
        help="Minimum number of silhouette frames required to keep an entry.",
    )
    parser.add_argument(
        "--model",
        choices=["grew_gaitbase", "current_gaitbase", "grew_gaitgl"],
        default="grew_gaitbase",
        help="Which gait checkpoint/profile to use.",
    )
    args = parser.parse_args()

    output_json = Path(args.output_json).resolve()
    work_root = Path(args.work_root).resolve()
    if args.video_path is not None:
        single_video = Path(args.video_path).resolve()
        if not single_video.exists():
            raise FileNotFoundError(f"Video file does not exist: {single_video}")
        video_dir = single_video.parent
        video_paths = [single_video]
    else:
        video_dir = Path(args.video_dir).resolve()
        if not video_dir.exists():
            raise FileNotFoundError(f"Video folder does not exist: {video_dir}")
        video_paths = sorted(
            [path for path in video_dir.iterdir() if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
        )
        if not video_paths:
            raise FileNotFoundError(f"No videos found in {video_dir}")

    model, profile = build_model(args.model)
    log(f"[run] total_videos={len(video_paths)} model={args.model}")

    all_entries: List[Dict] = []
    for video_path in video_paths:
        log(f"[run] analyzing {video_path.name}")
        all_entries.extend(
            analyze_video(
                model=model,
                video_path=video_path,
                work_root=work_root,
                min_frames=args.min_frames,
            )
        )

    log("[run] assigning identities")
    results = assign_identities(all_entries, args.cosine_threshold)
    results["config"] = {
        "video_dir": str(video_dir),
        "work_root": str(work_root),
        "output_json": str(output_json),
        "model": profile["display_name"],
        "model_key": args.model,
        "checkpoint": str(profile["checkpoint"]),
        "config_path": str(profile["config"]),
        "notes": [
            "Each stable track is treated as one entry candidate.",
            "This is a first-pass probe-only matcher without a fixed gallery.",
            "The gait model still operates on silhouettes generated from RGB video.",
        ],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2))
    log(json.dumps(results["summary"], indent=2))
    log(f"[run] saved analysis JSON to {output_json}")


if __name__ == "__main__":
    main()
