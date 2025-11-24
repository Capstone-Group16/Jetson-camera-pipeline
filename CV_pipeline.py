# cv_pipeline.py

import cv2
import torch
from ultralytics import YOLO


def iou(boxA, boxB) -> float:
    """Intersection-over-Union between two boxes (x1, y1, x2, y2)."""
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


class CVPipeline:
    """
    Core CV logic:
      - YOLO person detection
      - Haar face detection
      - tracking with global IDs and simple cross-camera linking
    """

    def __init__(
        self,
        num_cams: int,
        scale: float = 0.25,
        conf: float = 0.35,
        max_miss_time: float = 2.0,
        cross_cam_window: float = 1.0,
    ):
        # device
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"[CVPipeline] Using GPU: {torch.cuda.get_device_name(0)}")
            self.use_half = True
        else:
            self.device = "cpu"
            print("[CVPipeline] Using CPU")
            self.use_half = False

        # models
        self.model = YOLO("yolo11n.pt").to(self.device)
        self.person_class_id = 0  # COCO 'person'

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade")

        # tracking state
        self.tracks = []         # list of dicts (global across cameras)
        self.next_track_id = 0

        self.scale = scale
        self.conf = conf
        self.max_miss_time = max_miss_time
        self.cross_cam_window = cross_cam_window
        self.num_cams = num_cams

    # ---------- helpers for tracking ----------

    def _neighbors(self, cam_id: int):
        """Left/right neighbor cameras in a ring."""
        left = (cam_id - 1) % self.num_cams
        right = (cam_id + 1) % self.num_cams
        return [left, right]

    def _update_tracks_for_type(
        self,
        detections,
        obj_type: str,
        camera_id: int,
        t_now: float,
        iou_thresh: float = 0.3,
    ):
        """
        Update tracks for a given object type ('person'/'face') and camera.
        Handles:
          - within-camera IoU matching
          - new track creation
          - simple cross-camera linking to left/right neighbors
          - stale track removal & logging
        """
        # 1) mark existing tracks of this type & camera as unmatched
        for tr in self.tracks:
            if tr["type"] == obj_type and tr["camera_id"] == camera_id:
                tr["matched"] = False
                tr["just_created"] = False  # reset flag

        newly_created = []

        # 2) IoU match or create new tracks (within same camera)
        for det in detections:
            best_tr = None
            best_iou = 0.0

            for tr in self.tracks:
                if tr["type"] != obj_type or tr["camera_id"] != camera_id:
                    continue
                i = iou(det, tr["bbox"])
                if i > best_iou:
                    best_iou = i
                    best_tr = tr

            if best_tr is not None and best_iou >= iou_thresh:
                # update existing track
                best_tr["bbox"] = det
                best_tr["last_seen"] = t_now
                best_tr["matched"] = True
                best_tr["cams_seen"].add(camera_id)
            else:
                # create new track
                track_id = self.next_track_id
                self.next_track_id += 1

                new_tr = {
                    "id": track_id,
                    "type": obj_type,
                    "camera_id": camera_id,
                    "bbox": det,
                    "first_seen": t_now,
                    "last_seen": t_now,
                    "matched": True,
                    "just_created": True,
                    "cams_seen": {camera_id},
                }
                self.tracks.append(new_tr)
                newly_created.append(new_tr)

                print(
                    f"[TRACK START] id={track_id} type={obj_type} "
                    f"cam={camera_id} t={t_now:.2f}"
                )

        # 3) Simple cross-camera linking: new tracks ↔ recent tracks in neighbor cams
        for new_tr in newly_created:
            # neighbors of the camera where this track just appeared
            neighbors = self._neighbors(new_tr["camera_id"])

            # candidate tracks: same obj_type, different camera, seen recently, not matched now
            candidate = None
            best_dt = float("inf")

            for tr in self.tracks:
                if tr is new_tr:
                    continue
                if tr["type"] != obj_type:
                    continue
                if tr["camera_id"] not in neighbors:
                    continue
                if tr.get("matched", False):
                    # already updated in this frame; skip
                    continue

                dt = abs(t_now - tr["last_seen"])
                if dt <= self.cross_cam_window and dt < best_dt:
                    best_dt = dt
                    candidate = tr

            if candidate is not None:
                # Merge new_tr into candidate's ID (treat as same object)
                old_id = candidate["id"]
                new_id = new_tr["id"]

                candidate["last_seen"] = max(candidate["last_seen"], new_tr["last_seen"])
                candidate["camera_id"] = new_tr["camera_id"]
                candidate["bbox"] = new_tr["bbox"]
                candidate["matched"] = True
                candidate["cams_seen"].update(new_tr["cams_seen"])

                # Remove the new_tr entry; candidate remains canonical
                self.tracks = [tr for tr in self.tracks if tr is not new_tr]

                print(
                    f"[TRACK MERGE] new_id={new_id} -> old_id={old_id} "
                    f"type={obj_type} cams={sorted(candidate['cams_seen'])} "
                    f"t={t_now:.2f}"
                )

        # 4) Remove stale tracks & log their lifetimes
        still_alive = []
        for tr in self.tracks:
            if tr["type"] != obj_type:
                still_alive.append(tr)
                continue

            if t_now - tr["last_seen"] <= self.max_miss_time:
                still_alive.append(tr)
            else:
                duration = tr["last_seen"] - tr["first_seen"]
                cam_list = sorted(tr["cams_seen"])
                print(
                    f"[TRACK END] id={tr['id']} type={obj_type} "
                    f"cams={cam_list} duration={duration:.2f}s"
                )

        self.tracks = still_alive

    # ---------- main public API ----------

    def process_frame(self, frame, camera_id: int, t_now: float):
        """
        Run detection + tracking + drawing on a single frame.
        Returns annotated frame.
        """
        s = self.scale

        # downscale for speed
        small = cv2.resize(frame, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # YOLO person detection
        res = self.model.predict(
            small_rgb,
            device=self.device,
            imgsz=480,
            conf=self.conf,
            verbose=False,
            half=self.use_half,
        )[0]

        person_dets = []
        for box in res.boxes:
            if int(box.cls[0]) != self.person_class_id:
                continue
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(float)
            x1 /= s
            y1 /= s
            x2 /= s
            y2 /= s
            person_dets.append([int(x1), int(y1), int(x2), int(y2)])

        # Haar faces on small gray
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.equalizeHist(gray_small)
        faces_small = self.face_cascade.detectMultiScale(
            gray_small,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
        )

        face_dets = []
        for (fx, fy, fw, fh) in faces_small:
            x1 = int(fx / s)
            y1 = int(fy / s)
            x2 = int((fx + fw) / s)
            y2 = int((fy + fh) / s)
            face_dets.append([x1, y1, x2, y2])

        # update tracks for persons + faces
        self._update_tracks_for_type(person_dets, "person", camera_id, t_now)
        self._update_tracks_for_type(face_dets, "face", camera_id, t_now)

        # draw tracks that belong to this camera
        for tr in self.tracks:
            if tr["camera_id"] != camera_id:
                continue
            if not tr.get("matched", False):
                continue

            x1, y1, x2, y2 = tr["bbox"]
            duration = t_now - tr["first_seen"]
            dur_str = f"{duration:.1f}s"

            if tr["type"] == "person":
                color = (0, 255, 0)
                label = "person"
            else:
                color = (255, 0, 0)
                label = "face"

            # include track ID so you can see cross-camera continuity
            text = f"{label}#{tr['id']} {dur_str}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
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

        return frame
