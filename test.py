import time
import math

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def iou(boxA, boxB):
    """Compute Intersection-over-Union between two boxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter_area / float(boxA_area + boxB_area - inter_area + 1e-6)


def update_tracks(
    detections,
    tracks,
    obj_type,
    camera_id,
    t_now,
    next_id,
    iou_thresh=0.3,
    max_miss_time=2.0,
):
    """
    Update track list with new detections of a given type ("person" or "face")
    for a specific camera.
    """
    # Mark existing tracks of this type & camera as unmatched for this frame
    for tr in tracks:
        if tr["type"] == obj_type and tr["camera_id"] == camera_id:
            tr["matched"] = False

    # Associate each detection with an existing track (greedy IoU)
    for det in detections:
        best_tr = None
        best_iou = 0.0

        for tr in tracks:
            if tr["type"] != obj_type or tr["camera_id"] != camera_id:
                continue
            i = iou(det, tr["bbox"])
            if i > best_iou:
                best_iou = i
                best_tr = tr

        if best_tr is not None and best_iou >= iou_thresh:
            # Update existing track
            best_tr["bbox"] = det
            best_tr["last_seen"] = t_now
            best_tr["matched"] = True
        else:
            # Create a new track
            tracks.append(
                {
                    "id": next_id,
                    "type": obj_type,
                    "camera_id": camera_id,
                    "bbox": det,
                    "first_seen": t_now,
                    "last_seen": t_now,
                    "matched": True,
                }
            )
            next_id += 1

    # Remove stale tracks of this type & camera
    new_tracks = []
    for tr in tracks:
        if tr["type"] != obj_type or tr["camera_id"] != camera_id:
            new_tracks.append(tr)
            continue

        if t_now - tr["last_seen"] <= max_miss_time:
            new_tracks.append(tr)
        # else: drop it

    return new_tracks, next_id


def get_frame(camera_id, shared_capture):
    """
    Abstraction for grabbing a frame from a specific camera.

    For now, all logical cameras use the same physical webcam.
    On Jetson, replace this with a list of cv2.VideoCapture objects
    and index by camera_id.
    """
    ret, frame = shared_capture.read()
    return ret, frame


def make_grid(frames, grid_cols=2):
    """
    Build a grid image from a list of frames (BGR).
    frames: list of np.ndarray or None
    grid_cols: number of columns in the grid
    """
    # Replace Nones with black frames matching the first valid frame
    valid_frame = next((f for f in frames if f is not None), None)
    if valid_frame is None:
        return None

    h, w = valid_frame.shape[:2]
    processed = []

    for f in frames:
        if f is None:
            processed.append(np.zeros((h, w, 3), dtype=np.uint8))
        else:
            # Ensure same size
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h))
            processed.append(f)

    num = len(processed)
    cols = min(grid_cols, num)
    rows = math.ceil(num / cols)

    grid_rows = []
    idx = 0
    for r in range(rows):
        row_frames = []
        for c in range(cols):
            if idx < num:
                row_frames.append(processed[idx])
            else:
                row_frames.append(np.zeros((h, w, 3), dtype=np.uint8))
            idx += 1
        grid_rows.append(np.hstack(row_frames))

    grid = np.vstack(grid_rows)
    return grid


def main():
    # ------------------------
    # Config
    # ------------------------
    NUM_CAMERAS = 4
    PER_CAMERA_FPS = 1.0  # target 1 FPS per camera
    GLOBAL_TARGET_FPS = NUM_CAMERAS * PER_CAMERA_FPS
    GLOBAL_INTERVAL = 1.0 / GLOBAL_TARGET_FPS  # seconds between loop iterations

    # ------------------------
    # Device selection
    # ------------------------
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
        use_half = True
    else:
        device = "cpu"
        print("Using CPU")
        use_half = False

    # ------------------------
    # YOLO model (persons)
    # ------------------------
    model = YOLO("yolo11n.pt")
    model.to(device)
    PERSON_CLASS_ID = 0  # COCO index for 'person'

    # ------------------------
    # Face detector (Haar)
    # ------------------------
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade for faces")

    # ------------------------
    # Shared physical camera (simulating multiple logical cameras)
    # ------------------------
    shared_capture = cv2.VideoCapture(0)
    if not shared_capture.isOpened():
        raise RuntimeError("Could not open camera")

    shared_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    shared_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' to quit")

    last_time = time.time()
    frames = 0
    fps = 0.0
    infer_times = []

    # Downscale factor for both YOLO and face detection
    scale = 0.25  # 640x480 -> 160x120

    # Tracking state
    tracks = []
    next_track_id = 0

    current_cam = 0  # round-robin camera index

    # Store last annotated frame for each logical camera
    last_frames = [None for _ in range(NUM_CAMERAS)]

    while True:
        loop_start = time.time()
        frame_time = loop_start
        camera_id = current_cam

        # ---- Grab frame for this logical camera ----
        ret, frame = get_frame(camera_id, shared_capture)
        if not ret:
            print("No frame grabbed, exiting...")
            break

        frames += 1

        # ------------------------
        # Downscale frame for fast detection
        # ------------------------
        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR,
        )
        small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # ------------------------
        # YOLO person detection
        # ------------------------
        t0 = time.time()
        results = model.predict(
            small_rgb,
            device=device,
            imgsz=480,
            conf=0.35,
            verbose=False,
            half=use_half,
        )[0]
        t1 = time.time()
        infer_times.append(t1 - t0)

        person_dets = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id != PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(float)

            # Rescale back to original frame coords
            x1 /= scale
            y1 /= scale
            x2 /= scale
            y2 /= scale

            person_dets.append([int(x1), int(y1), int(x2), int(y2)])

        # ------------------------
        # Face detection (Haar) on downscaled gray
        # ------------------------
        gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.equalizeHist(gray_small)

        faces_small = face_cascade.detectMultiScale(
            gray_small,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
        )

        face_dets = []
        for (fx, fy, fw, fh) in faces_small:
            x1 = int(fx / scale)
            y1 = int(fy / scale)
            x2 = int((fx + fw) / scale)
            y2 = int((fy + fh) / scale)
            face_dets.append([x1, y1, x2, y2])

        # ------------------------
        # Update tracks for persons and faces for THIS camera
        # ------------------------
        tracks, next_track_id = update_tracks(
            person_dets, tracks, "person", camera_id, frame_time, next_track_id
        )
        tracks, next_track_id = update_tracks(
            face_dets, tracks, "face", camera_id, frame_time, next_track_id
        )

        # ------------------------
        # Draw tracks (only those belonging to this camera)
        # ------------------------
        for tr in tracks:
            if tr["camera_id"] != camera_id:
                continue
            if not tr.get("matched", False):
                continue

            x1, y1, x2, y2 = tr["bbox"]
            duration = frame_time - tr["first_seen"]
            duration_str = f"{duration:.1f}s"

            if tr["type"] == "person":
                color = (0, 255, 0)
                label = "person"
            else:
                color = (255, 0, 0)
                label = "face"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{label} {duration_str}"
            cv2.putText(
                frame,
                text,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        # Tag this frame with camera ID
        cv2.putText(
            frame,
            f"Cam {camera_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Store the annotated frame for this camera
        last_frames[camera_id] = frame.copy()

        # ------------------------
        # Global FPS overlay (based on scheduler loop)
        # ------------------------
        now = time.time()
        dt = now - last_time
        if dt >= 1.0:
            fps = frames / dt
            frames = 0
            last_time = now

        # Build 2x2 grid (or generic grid) from last_frames
        grid = make_grid(last_frames, grid_cols=2)
        if grid is None:
            grid = frame  # fallback

        cv2.putText(
            grid,
            f"Global loop FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Interleaved Cameras (Grid View)", grid)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # ---- Round-robin: move to next camera for next loop ----
        current_cam = (current_cam + 1) % NUM_CAMERAS

        # ---- Throttle loop to hit target per-camera FPS ----
        elapsed = time.time() - loop_start
        sleep_time = GLOBAL_INTERVAL - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    shared_capture.release()
    cv2.destroyAllWindows()

    if infer_times:
        avg_infer = sum(infer_times) / len(infer_times)
        print(f"Average YOLO inference time: {avg_infer*1000:.2f} ms per frame")


if __name__ == "__main__":
    main()
