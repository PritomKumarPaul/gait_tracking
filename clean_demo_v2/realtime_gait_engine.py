import os
import sys
import time
import threading
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CLEAN_ROOT = ROOT / "clean_demo_v2"
OPENGAIT_TOOLS = ROOT / "OpenGait" / "tools"
OPENGAIT_LIBS = ROOT / "OpenGait" / "demo" / "libs"
PADDLE_DIR = OPENGAIT_LIBS / "paddle"

sys.path.insert(0, str(OPENGAIT_TOOLS))
sys.path.insert(0, str(OPENGAIT_LIBS))
sys.path.insert(0, str(PADDLE_DIR))

from probe_only_entry_analysis import build_model, extract_embedding, cosine_similarity
from seg_demo import get_seg_predictor
from track import get_tracker_model
from tracker.byte_tracker import BYTETracker
from tracking_utils.predictor import Predictor
from tracking_utils.timer import Timer

SEG_MODEL_CFG = ROOT / "OpenGait" / "demo" / "checkpoints" / "seg_model" / "human_pp_humansegv2_mobile_192x192_inference_model_with_softmax" / "deploy.yaml"
GALLERY_FILE = CLEAN_ROOT / "output" / "live_gallery.npz"


class TrackedPersonState:
    def __init__(self, track_id: int, bbox: Tuple[float, float, float, float], score: float):
        self.track_id = track_id
        self.bbox = bbox  # x1, y1, w, h
        self.score = score
        self.last_seen = time.time()
        self.sil_buffer = deque(maxlen=30)
        self.latest_silhouette: Optional[np.ndarray] = None
        self.latest_crop: Optional[np.ndarray] = None
        self.embedding: Optional[np.ndarray] = None
        self.identity: str = "Unenrolled"
        self.confidence: float = 0.0
        self.status: str = "COLLECTING"
        self.last_inferred_time: float = 0.0


class RealtimeGaitEngine:
    def __init__(self, camera_id: int = 0, threshold: float = 0.85):
        self.camera_id = camera_id
        self.threshold = threshold
        self.running = False
        self.lock = threading.Lock()

        self.raw_frame: Optional[np.ndarray] = None
        self.fps: float = 0.0
        self.frame_count: int = 0
        self.start_time: float = time.time()

        self.tracks: Dict[int, TrackedPersonState] = {}
        self.active_track_ids: List[int] = []

        self.gallery: Dict[str, np.ndarray] = {}
        self.gallery_meta: Dict[str, Dict] = {}
        self.load_gallery_from_disk()

        self.recent_events: deque = deque(maxlen=20)

        print("[INFO] Loading ByteTrack...")
        exp, bmodel = get_tracker_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bytetrack_predictor = Predictor(bmodel, exp, None, None, device, device.type == "cuda")
        self.tracker = BYTETracker(frame_rate=30)
        self.exp = exp
        self.timer = Timer()

        print("[INFO] Loading PaddleSeg...")
        self.seg_predictor = get_seg_predictor(str(SEG_MODEL_CFG))

        print("[INFO] Loading GaitBase...")
        self.gait_model, self.gait_profile = build_model("grew_gaitbase")
        print("[INFO] Models initialized.")

        self.capture_thread: Optional[threading.Thread] = None
        self.tracking_thread: Optional[threading.Thread] = None
        self.seg_thread: Optional[threading.Thread] = None
        self.gait_thread: Optional[threading.Thread] = None

    def load_gallery_from_disk(self):
        if GALLERY_FILE.exists():
            try:
                data = np.load(str(GALLERY_FILE), allow_pickle=True)
                labels = data["labels"]
                embeddings = data["embeddings"]
                for lbl, emb in zip(labels, embeddings):
                    self.gallery[str(lbl)] = emb.astype(np.float32)
                    self.gallery_meta[str(lbl)] = {"enrolled_at": time.strftime("%H:%M:%S")}
                print(f"[INFO] Loaded {len(self.gallery)} identities from cache.")
            except Exception as e:
                print(f"[WARN] Failed loading gallery cache: {e}")

    def save_gallery_to_disk(self):
        try:
            GALLERY_FILE.parent.mkdir(parents=True, exist_ok=True)
            if not self.gallery:
                if GALLERY_FILE.exists():
                    GALLERY_FILE.unlink()
                return
            labels = list(self.gallery.keys())
            embeddings = np.array([self.gallery[k] for k in labels], dtype=np.float32)
            np.savez(str(GALLERY_FILE), labels=labels, embeddings=embeddings)
        except Exception as e:
            print(f"[WARN] Failed saving gallery cache: {e}")

    def enroll_person(self, track_id: int, person_name: str) -> Tuple[bool, str]:
        with self.lock:
            if track_id not in self.tracks:
                return False, f"Track #{track_id} is not active."
            track = self.tracks[track_id]
            if len(track.sil_buffer) < 10:
                return False, f"Need at least 10 frames (currently {len(track.sil_buffer)})."

            if track.embedding is None:
                seq = np.array(list(track.sil_buffer), dtype=np.uint8)
                inp = ([[seq]], [person_name], ["live"], ["000"], np.array([[len(seq)]]))
                track.embedding = extract_embedding(self.gait_model, inp)

            self.gallery[person_name] = track.embedding.copy()
            self.gallery_meta[person_name] = {
                "enrolled_at": time.strftime("%H:%M:%S"),
                "track_id": track_id,
                "frames": len(track.sil_buffer)
            }
            track.identity = person_name
            track.confidence = 1.0
            track.status = "MATCH"
            self.save_gallery_to_disk()
            msg = f"Enrolled {person_name} (Track #{track_id})"
            self.recent_events.appendleft({
                "time": time.strftime("%H:%M:%S"),
                "event": "ENROLL",
                "message": msg,
                "track_id": track_id,
                "identity": person_name,
                "score": 1.0
            })
            print(f"[INFO] {msg}")
            return True, msg

    def clear_gallery(self) -> str:
        with self.lock:
            self.gallery.clear()
            self.gallery_meta.clear()
            self.save_gallery_to_disk()
            for t in self.tracks.values():
                t.identity = "Unenrolled"
                t.confidence = 0.0
                t.status = "COLLECTING"
            msg = "Gallery cleared."
            self.recent_events.appendleft({
                "time": time.strftime("%H:%M:%S"),
                "event": "CLEAR",
                "message": msg,
                "track_id": "-",
                "identity": "-",
                "score": 0.0
            })
            return msg

    def delete_identity(self, name: str) -> str:
        with self.lock:
            if name in self.gallery:
                del self.gallery[name]
                if name in self.gallery_meta:
                    del self.gallery_meta[name]
                self.save_gallery_to_disk()
                for t in self.tracks.values():
                    if t.identity == name:
                        t.identity = "Unenrolled"
                        t.status = "COLLECTING"
                        t.confidence = 0.0
                return f"Deleted {name}."
            return f"{name} not found."

    def start(self):
        if self.running:
            return
        self.running = True

        if sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {self.camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.tracking_thread = threading.Thread(target=self._tracking_worker, daemon=True)
        self.seg_thread = threading.Thread(target=self._segmentation_worker, daemon=True)
        self.gait_thread = threading.Thread(target=self._gait_worker, daemon=True)

        self.capture_thread.start()
        self.tracking_thread.start()
        self.seg_thread.start()
        self.gait_thread.start()
        print("[INFO] Camera capture and tracking started.")

    def stop(self):
        self.running = False
        for t in [self.capture_thread, self.tracking_thread, self.seg_thread, self.gait_thread]:
            if t and t.is_alive():
                t.join(timeout=1.0)
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        print("[INFO] Pipeline stopped.")

    def _capture_worker(self):
        fps_counter = 0
        fps_start = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            with self.lock:
                self.raw_frame = frame
                self.frame_count += 1
                fps_counter += 1
                if time.time() - fps_start >= 1.0:
                    self.fps = fps_counter / (time.time() - fps_start)
                    fps_counter = 0
                    fps_start = time.time()

            time.sleep(0.01)

    def _tracking_worker(self):
        while self.running:
            frame_to_track = None
            with self.lock:
                if self.raw_frame is not None:
                    frame_to_track = self.raw_frame.copy()

            if frame_to_track is None:
                time.sleep(0.02)
                continue

            try:
                outputs, img_info = self.bytetrack_predictor.inference(frame_to_track, self.timer)
                current_active_ids = []
                now = time.time()

                if outputs[0] is not None:
                    targets = self.tracker.update(outputs[0], [img_info["height"], img_info["width"]], self.exp.test_size)
                    with self.lock:
                        for t in targets:
                            tlwh = t.tlwh
                            if tlwh[2] * tlwh[3] < 150 or (tlwh[2] / max(tlwh[3], 1.0)) > 2.0:
                                continue
                            tid = int(t.track_id)
                            current_active_ids.append(tid)
                            bbox = (float(tlwh[0]), float(tlwh[1]), float(tlwh[2]), float(tlwh[3]))
                            score = float(t.score)

                            if tid not in self.tracks:
                                self.tracks[tid] = TrackedPersonState(tid, bbox, score)
                            else:
                                self.tracks[tid].bbox = bbox
                                self.tracks[tid].score = score
                                self.tracks[tid].last_seen = now

                        dead_ids = [tid for tid, state in self.tracks.items() if now - state.last_seen > 5.0]
                        for tid in dead_ids:
                            del self.tracks[tid]

                        self.active_track_ids = current_active_ids
            except Exception as e:
                time.sleep(0.05)

            time.sleep(0.02)

    def _segmentation_worker(self):
        while self.running:
            target_track: Optional[TrackedPersonState] = None
            curr_frame: Optional[np.ndarray] = None

            with self.lock:
                if self.raw_frame is not None and self.active_track_ids:
                    curr_frame = self.raw_frame.copy()
                    for tid in self.active_track_ids:
                        if tid in self.tracks:
                            target_track = self.tracks[tid]
                            break

            if target_track is None or curr_frame is None:
                time.sleep(0.04)
                continue

            try:
                x, y, w_box, h_box = [int(v) for v in target_track.bbox]
                h_img, w_img = curr_frame.shape[:2]

                x1_new = max(0, int(x - 0.1 * w_box))
                x2_new = min(w_img, int(x + w_box + 0.1 * w_box))
                y1_new = max(0, int(y - 0.1 * h_box))
                y2_new = min(h_img, int(y + h_box + 0.1 * h_box))

                if (x2_new - x1_new) > 10 and (y2_new - y1_new) > 10:
                    crop = curr_frame[y1_new:y2_new, x1_new:x2_new]
                    ch, cw = crop.shape[:2]
                    side = max(ch, cw)
                    square = np.full((side, side, 3), 255, dtype=np.uint8)
                    ox = (side - cw) // 2
                    oy = (side - ch) // 2
                    square[oy:oy + ch, ox:ox + cw] = crop

                    sq192 = cv2.resize(square, (192, 192))
                    bg = 255 * np.ones((192, 192, 3), dtype=np.uint8)
                    out_img, out_mask = self.seg_predictor.run(sq192, bg)
                    if out_mask.ndim == 3:
                        out_mask = out_mask.squeeze(-1)
                    mask = np.where(out_mask < 80, 0, 255).astype(np.uint8)

                    y_sum = mask.sum(axis=1)
                    y_nonzero = np.where(y_sum > 0)[0]
                    if len(y_nonzero) > 5:
                        y_top = y_nonzero[0]
                        y_btm = y_nonzero[-1]
                        sil_cropped = mask[y_top:y_btm + 1, :]
                        sh, sw = sil_cropped.shape[:2]
                        ratio = sw / max(sh, 1)
                        target_w = max(1, int(64 * ratio))
                        sil_resized = cv2.resize(sil_cropped, (target_w, 64), interpolation=cv2.INTER_NEAREST)

                        canvas = np.zeros((64, 64), dtype=np.uint8)
                        if target_w >= 64:
                            canvas = sil_resized[:, :64]
                        else:
                            start_x = (64 - target_w) // 2
                            canvas[:, start_x:start_x + target_w] = sil_resized
                    else:
                        canvas = cv2.resize(mask, (64, 64))

                    with self.lock:
                        target_track.sil_buffer.append(canvas)
                        target_track.latest_silhouette = mask
                        target_track.latest_crop = crop

            except Exception as e:
                time.sleep(0.05)

            time.sleep(0.04)

    def _gait_worker(self):
        while self.running:
            tracks_to_infer: List[Tuple[int, np.ndarray]] = []
            now = time.time()

            with self.lock:
                for tid, track in self.tracks.items():
                    if len(track.sil_buffer) >= 12 and (now - track.last_inferred_time) > 0.5:
                        seq = np.array(list(track.sil_buffer), dtype=np.uint8)
                        tracks_to_infer.append((tid, seq))

            for tid, seq in tracks_to_infer:
                try:
                    inp = ([[seq]], [str(tid)], ["live"], ["000"], np.array([[len(seq)]]))
                    emb = extract_embedding(self.gait_model, inp)

                    with self.lock:
                        if tid in self.tracks:
                            track = self.tracks[tid]
                            track.embedding = emb
                            track.last_inferred_time = time.time()

                            if not self.gallery:
                                track.identity = "Unenrolled"
                                track.confidence = 0.0
                                track.status = "COLLECTING"
                            else:
                                scores = [
                                    (name, cosine_similarity(emb, gal_emb))
                                    for name, gal_emb in self.gallery.items()
                                ]
                                scores.sort(key=lambda x: x[1], reverse=True)
                                best_name, best_score = scores[0]

                                if best_score >= self.threshold:
                                    prev_status = track.status
                                    track.identity = best_name
                                    track.confidence = best_score
                                    track.status = "MATCH"
                                    if prev_status != "MATCH":
                                        self.recent_events.appendleft({
                                            "time": time.strftime("%H:%M:%S"),
                                            "event": "MATCH",
                                            "message": f"Match: {best_name} ({best_score:.1%})",
                                            "track_id": tid,
                                            "identity": best_name,
                                            "score": float(best_score)
                                        })
                                else:
                                    track.identity = "Unknown"
                                    track.confidence = best_score
                                    track.status = "UNKNOWN"
                except Exception as e:
                    pass

            time.sleep(0.1)

    def get_latest_annotated_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.raw_frame is None:
                return None
            frame = self.raw_frame.copy()
            tracks_snapshot = list(self.tracks.values())
            gallery_count = len(self.gallery)
            fps_val = self.fps
            active_ids = list(self.active_track_ids)

        h_img, w_img = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_img, 65), (20, 24, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        cv2.putText(frame, "OpenGait Live Tracking", (16, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.70, (0, 220, 255), 2, cv2.LINE_AA)
        status_sub = f"GaitBase DA + ByteTrack + PaddleSeg"
        cv2.putText(frame, status_sub, (16, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 190, 200), 1, cv2.LINE_AA)

        badge_fps = f"{fps_val:.1f} FPS"
        badge_tracks = f"{len(active_ids)} tracked"
        badge_gal = f"{gallery_count} enrolled"

        cv2.putText(frame, badge_fps, (w_img - 110, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 128), 2, cv2.LINE_AA)
        cv2.putText(frame, badge_tracks, (w_img - 210, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, badge_gal, (w_img - 210, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 200, 255), 1, cv2.LINE_AA)

        active_silhouette_to_pip: Optional[np.ndarray] = None
        pip_track_id: Optional[int] = None

        for track in tracks_snapshot:
            if track.track_id not in active_ids:
                continue

            x, y, w_box, h_box = [int(v) for v in track.bbox]
            x = max(0, min(x, w_img - 1))
            y = max(70, min(y, h_img - 1))
            w_box = max(10, min(w_box, w_img - x))
            h_box = max(10, min(h_box, h_img - y))

            if track.status == "MATCH":
                color = (0, 255, 100)
                tag_label = f"{track.identity} ({track.confidence:.1%})"
            elif track.status == "UNKNOWN":
                color = (50, 80, 255)
                tag_label = f"Unknown ({track.confidence:.1%})"
            else:
                color = (255, 190, 0)
                tag_label = "Analyzing..."

            line_len = min(22, int(w_box * 0.25), int(h_box * 0.25))
            thick = 2
            cv2.line(frame, (x, y), (x + line_len, y), color, thick)
            cv2.line(frame, (x, y), (x, y + line_len), color, thick)
            cv2.line(frame, (x + w_box, y), (x + w_box - line_len, y), color, thick)
            cv2.line(frame, (x + w_box, y), (x + w_box, y + line_len), color, thick)
            cv2.line(frame, (x, y + h_box), (x + line_len, y + h_box), color, thick)
            cv2.line(frame, (x, y + h_box), (x, y + h_box - line_len), color, thick)
            cv2.line(frame, (x + w_box, y + h_box), (x + w_box - line_len, y + h_box), color, thick)
            cv2.line(frame, (x + w_box, y + h_box), (x + w_box, y + line_len), color, thick)
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 1)

            main_badge = f"ID #{track.track_id} | {tag_label}"
            (tw, th), _ = cv2.getTextSize(main_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            badge_y1 = max(70, y - th - 12)
            cv2.rectangle(frame, (x, badge_y1), (x + tw + 10, badge_y1 + th + 8), (15, 18, 24), -1)
            cv2.rectangle(frame, (x, badge_y1), (x + tw + 10, badge_y1 + th + 8), color, 1)
            cv2.putText(frame, main_badge, (x + 5, badge_y1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)

            buf_len = len(track.sil_buffer)
            prog = min(1.0, buf_len / 30.0)
            bar_y = badge_y1 - 6
            if bar_y > 70:
                bar_w = int(tw * prog)
                cv2.rectangle(frame, (x, bar_y), (x + tw + 10, bar_y + 4), (40, 40, 40), -1)
                cv2.rectangle(frame, (x, bar_y), (x + bar_w, bar_y + 4), (0, 220, 255), -1)

            if track.latest_silhouette is not None and active_silhouette_to_pip is None:
                active_silhouette_to_pip = track.latest_silhouette
                pip_track_id = track.track_id

        if active_silhouette_to_pip is not None:
            pip_size = 120
            pip_bgr = cv2.cvtColor(active_silhouette_to_pip, cv2.COLOR_GRAY2BGR)
            pip_resized = cv2.resize(pip_bgr, (pip_size, pip_size))

            px = w_img - pip_size - 16
            py = h_img - pip_size - 16

            cv2.rectangle(frame, (px - 2, py - 18), (px + pip_size + 2, py + pip_size + 2), (15, 18, 24), -1)
            cv2.rectangle(frame, (px - 2, py - 18), (px + pip_size + 2, py + pip_size + 2), (0, 255, 128), 1)
            frame[py:py + pip_size, px:px + pip_size] = pip_resized

            badge_text = f"Silhouette #{pip_track_id}"
            cv2.putText(frame, badge_text, (px, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 128), 1, cv2.LINE_AA)

        return frame

    def generate_mjpeg(self):
        while self.running:
            annotated = self.get_latest_annotated_frame()
            if annotated is None:
                time.sleep(0.03)
                continue

            ret, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                time.sleep(0.03)
                continue

            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.03)
