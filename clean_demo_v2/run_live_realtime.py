import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "clean_demo_v2"))

from realtime_gait_engine import RealtimeGaitEngine


def main():
    parser = argparse.ArgumentParser(description="OpenGait Live Camera Tracking")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index")
    parser.add_argument("--threshold", type=float, default=0.85, help="Match threshold")
    args = parser.parse_args()

    print("OpenGait Live Camera Tracking")
    print("Keys: [E] Enroll active track | [C] Clear gallery | [Q] Quit")

    engine = RealtimeGaitEngine(camera_id=args.camera_id, threshold=args.threshold)
    engine.start()

    window_name = "OpenGait Live"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    try:
        while True:
            annotated = engine.get_latest_annotated_frame()
            if annotated is None:
                time.sleep(0.02)
                continue

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                res = engine.clear_gallery()
                print(f"[INFO] {res}")
            elif key == ord('e'):
                if engine.active_track_ids:
                    target_tid = engine.active_track_ids[0]
                    person_name = f"Subject_{len(engine.gallery) + 1}"
                    success, msg = engine.enroll_person(target_tid, person_name)
                    print(f"[INFO] {msg}")
                else:
                    print("[WARN] No active track to enroll.")

    finally:
        engine.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
