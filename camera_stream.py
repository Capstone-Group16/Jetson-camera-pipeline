import math
import time

import cv2
import numpy as np

from CV_pipeline import CVPipeline


# ------------ CONFIG ------------
VIDEO_PATHS = [
    "excavator_session/20250828132109/D01_20250828132109.mp4",
    "excavator_session/20250828132109/D02_20250828132109.mp4",
    "excavator_session/20250828132109/D03_20250828132109.mp4",
    "excavator_session/20250828132109/D04_20250828132109.mp4",
]

# Optional human-readable camera names (same length as VIDEO_PATHS),
# or set to None to use "Cam 0", "Cam 1", ...
CAMERA_NAMES = ["Front", "Left", "Rear", "Right"]
# CAMERA_NAMES = None

START_TIME_SEC = 60.0          # where to start in each video
DISPLAY_SIZE = (640, 360)      # per-camera view in the grid
# -------------------------------


def init_source(path, start_time_sec):
    """
    Open a VideoCapture, seek to START_TIME_SEC, and return a source dict:
      {cap, fps, curr_idx, start_idx, last_frame}
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    start_idx = int(start_time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    ret, frame = cap.read()
    if not ret:
        # fallback: black frame
        frame = np.zeros((DISPLAY_SIZE[1], DISPLAY_SIZE[0], 3), dtype=np.uint8)
    else:
        frame = cv2.resize(frame, DISPLAY_SIZE)

    return {
        "cap": cap,
        "fps": fps,
        "curr_idx": start_idx,
        "start_idx": start_idx,
        "last_frame": frame,
    }


def make_grid(frames, cols=2):
    """
    Safe grid builder: handles None frames or mismatched sizes.
    """
    valid = next((f for f in frames if f is not None), None)
    if valid is None:
        raise RuntimeError("No valid frames to build grid.")

    h, w = valid.shape[:2]

    safe_frames = []
    for f in frames:
        if f is None:
            safe_frames.append(np.zeros((h, w, 3), dtype=np.uint8))
        else:
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h))
            safe_frames.append(f)

    n = len(safe_frames)
    rows = math.ceil(n / cols)

    grid_rows = []
    idx = 0
    for _ in range(rows):
        row_frames = []
        for _ in range(cols):
            if idx < n:
                row_frames.append(safe_frames[idx])
            else:
                row_frames.append(np.zeros((h, w, 3), dtype=np.uint8))
            idx += 1
        grid_rows.append(cv2.hconcat(row_frames))

    return cv2.vconcat(grid_rows)


def main():
    sources = [init_source(p, START_TIME_SEC) for p in VIDEO_PATHS]
    num_cams = len(sources)

    # camera labels
    if CAMERA_NAMES is None or len(CAMERA_NAMES) != num_cams:
        cam_labels = [f"Cam {i}" for i in range(num_cams)]
    else:
        cam_labels = CAMERA_NAMES

    pipeline = CVPipeline(num_cams=len(sources), scale=0.25, conf=0.35)

    print(f"[SIL] Loaded {num_cams} video streams.")
    print("[SIL] Interleaving: each stream advances ~1s of video per turn.")
    print("Press 'q' in the window to quit.")

    current_cam = 0

    # per-camera FPS tracking
    cam_frame_counts = [0] * num_cams
    cam_fps = [0.0] * num_cams
    cam_last_time = [time.time()] * num_cams

    while True:
        t_now = time.time()

        # ---- process ONE camera this loop ----
        src = sources[current_cam]
        cap = src["cap"]
        fps = src["fps"]

        step = max(1, int(round(fps)))       # ~1 second worth of frames in video-time
        src["curr_idx"] += step
        cap.set(cv2.CAP_PROP_POS_FRAMES, src["curr_idx"])

        ret, frame = cap.read()
        if not ret:
            # loop back to start
            src["curr_idx"] = src["start_idx"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, src["start_idx"])
            ret, frame = cap.read()

        if not ret:
            frame = np.zeros((DISPLAY_SIZE[1], DISPLAY_SIZE[0], 3), dtype=np.uint8)
        else:
            frame = pipeline.process_frame(frame, camera_id=current_cam, t_now=t_now)
            frame = cv2.resize(frame, DISPLAY_SIZE)

        src["last_frame"] = frame

        # update per-camera FPS for this camera
        cam_frame_counts[current_cam] += 1
        dt_cam = t_now - cam_last_time[current_cam]
        if dt_cam >= 1.0:
            cam_fps[current_cam] = cam_frame_counts[current_cam] / dt_cam
            cam_frame_counts[current_cam] = 0
            cam_last_time[current_cam] = t_now

        # ---- build per-pane overlays (label + fps in bottom-left) ----
        frames_for_grid = []
        for cam_id, s in enumerate(sources):
            f = s["last_frame"]
            if f is None:
                h, w = DISPLAY_SIZE[1], DISPLAY_SIZE[0]
                f = np.zeros((h, w, 3), dtype=np.uint8)

            f = f.copy()
            h, w = f.shape[:2]

            label = cam_labels[cam_id]
            text = f"{label}: {cam_fps[cam_id]:.1f} FPS"

            cv2.putText(
                f,
                text,
                (10, h - 10),  # bottom-left
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            frames_for_grid.append(f)

        grid = make_grid(frames_for_grid, cols=2)

        cv2.imshow("SIL Multi-Cam (Interleaved 1 fps each)", grid)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # round-robin
        current_cam = (current_cam + 1) % num_cams

    for s in sources:
        s["cap"].release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
