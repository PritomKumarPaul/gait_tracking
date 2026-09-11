import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_ROOT = ROOT / "OpenGait" / "demo" / "checkpoints"


def download_file(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] Already exists: {dst} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")
        return dst
    print(f"[download] Downloading {url} -> {dst}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dst, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                percent = downloaded / total * 100
                print(f"\r  {downloaded / 1024 / 1024:.1f}MB / {total / 1024 / 1024:.1f}MB ({percent:.1f}%)", end="", flush=True)
            else:
                print(f"\r  {downloaded / 1024 / 1024:.1f}MB", end="", flush=True)
    print()
    return dst


def download_gdown(file_id: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] Already exists: {dst} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")
        return dst
    import gdown
    print(f"[gdown] Downloading Google Drive ID {file_id} -> {dst}")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(dst), quiet=False)
    return dst


def extract_zip(zip_path: Path, extract_dir: Path):
    print(f"[extract] Unzipping {zip_path.name} -> {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    print(f"[extract] Done unzipping {zip_path.name}")


def main():
    print("=== Downloading Gait Tracking Model Checkpoints ===")
    
    # 1. ByteTrack
    bytetrack_dst = CHECKPOINTS_ROOT / "bytetrack_model" / "bytetrack_x_mot17.pth.tar"
    download_gdown("1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5", bytetrack_dst)
    
    # 2. PaddleSeg human segmentation
    seg_zip = CHECKPOINTS_ROOT / "seg_model" / "human_pp_humansegv2_mobile_192x192_inference_model_with_softmax.zip"
    seg_dir = CHECKPOINTS_ROOT / "seg_model" / "human_pp_humansegv2_mobile_192x192_inference_model_with_softmax"
    if not (seg_dir / "deploy.yaml").exists():
        download_file(
            "https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax.zip",
            seg_zip,
        )
        extract_zip(seg_zip, CHECKPOINTS_ROOT / "seg_model")
    else:
        print(f"[skip] PaddleSeg deploy.yaml already exists at {seg_dir}")

    # 3. GREW GaitBase
    gaitbase_zip = CHECKPOINTS_ROOT / "gait_model" / "pretrained_grew_gaitbase.zip"
    gaitbase_pt = CHECKPOINTS_ROOT / "gait_model" / "GREW" / "Baseline" / "GaitBase_DA" / "checkpoints" / "GaitBase_DA-180000.pt"
    if not gaitbase_pt.exists():
        download_file(
            "https://github.com/ShiqiYu/OpenGait/releases/download/v2.0/pretrained_grew_gaitbase.zip",
            gaitbase_zip,
        )
        extract_zip(gaitbase_zip, CHECKPOINTS_ROOT / "gait_model")
    else:
        print(f"[skip] GaitBase checkpoint already exists at {gaitbase_pt}")

    print("=== Checkpoint verification ===")
    print("ByteTrack:", bytetrack_dst.exists(), bytetrack_dst)
    print("PaddleSeg:", (seg_dir / "deploy.yaml").exists(), seg_dir)
    print("GaitBase:", gaitbase_pt.exists(), gaitbase_pt)


if __name__ == "__main__":
    main()
