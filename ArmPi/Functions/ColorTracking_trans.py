from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import time
import math

import cv2
import numpy as np
import sys
sys.path.append('/home/pi/ArmPi/')

import cv2
import Camera

from LABConfig import color_range
from ArmIK.Transform import getMaskROI, getROI, getCenter, convertCoordinate
from CameraCalibration.CalibrationConfig import square_length

@dataclass
class TrackingPerceptionOutputs:
    """
    What the tracking perception hands off to the motion thread.
    Mirrors the key globals in your ColorTracking script.
    """
    # Detection identity
    detect_color: str = "None"

    # Instantaneous world estimate (used for continuous follow)
    world_x: float = 0.0
    world_y: float = 0.0

    # Stable, averaged pose used for pickup
    world_X: float = 0.0
    world_Y: float = 0.0
    rotation_angle: float = 0.0

    # State flags for the motion thread
    track: bool = False
    start_pick_up: bool = False

    # ROI state (optional for debugging/inspection)
    roi: Tuple[int, int, int, int] = ()
    get_roi: bool = False
    rect: Optional[Tuple[Any, Any, Any]] = None


class ColorTrackingPerception:
    """
    Perception-only refactor of the ColorTracking run(img) pipeline into methods.

    This class does NOT move the arm. It only:
      - processes frames
      - produces track updates (world_x/world_y + track flag)
      - produces pickup trigger after stability (world_X/world_Y + rotation_angle + start_pick_up)

    You inject the robot SDK helpers from ArmIK.Transform:
      - getMaskROI, getROI, getCenter, convertCoordinate
      and calibration parameter square_length, plus LABConfig color_range.
    """

    def __init__(
        self,
        *,
        size: Tuple[int, int] = (640, 480),
        range_rgb: Optional[Dict[str, Tuple[int, int, int]]] = None,
        color_range: Dict[str, Tuple[np.ndarray, np.ndarray]],
        target_colors: Tuple[str, ...] = ("red",),
        # thresholds from your tracking code
        min_valid_area: float = 2500.0,
        contour_valid_area: float = 300.0,
        stable_dist_thresh: float = 0.3,
        stable_time_sec: float = 1.5,
        # SDK helpers
        getMaskROI=None,
        getROI=None,
        getCenter=None,
        convertCoordinate=None,
        square_length: Optional[float] = None,
    ):
        self.size = size
        self.color_range = color_range
        self.target_colors = target_colors

        self.min_valid_area = float(min_valid_area)
        self.contour_valid_area = float(contour_valid_area)
        self.stable_dist_thresh = float(stable_dist_thresh)
        self.stable_time_sec = float(stable_time_sec)

        self.getMaskROI = getMaskROI
        self.getROI = getROI
        self.getCenter = getCenter
        self.convertCoordinate = convertCoordinate
        self.square_length = square_length

        self.range_rgb = range_rgb or {
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
        }

        # ---- Internal state (mirrors tracking script globals) ----
        self.__isRunning = True

        self.roi: Tuple[int, int, int, int] = ()
        self.get_roi: bool = False

        self.detect_color: str = "None"
        self.rect = None

        self.world_x: float = 0.0
        self.world_y: float = 0.0
        self.world_X: float = 0.0
        self.world_Y: float = 0.0
        self.rotation_angle: float = 0.0

        self.track: bool = False
        self.start_pick_up: bool = False

        # stability bookkeeping
        self.last_x: float = 0.0
        self.last_y: float = 0.0
        self.center_list: List[float] = []
        self.count: int = 0

        self.start_count_t1: bool = True
        self.t1: float = 0.0

        # gating (your code only accumulates stability when action_finish is True)
        self.action_finish: bool = True

    # ---------------------------
    # Lifecycle / flags
    # ---------------------------
    def set_running(self, is_running: bool) -> None:
        self.__isRunning = bool(is_running)

    def set_action_finish(self, action_finish: bool) -> None:
        """
        External control can set this to match the motion thread state.
        In the original, stability counting happens only when action_finish is True.
        """
        self.action_finish = bool(action_finish)

    def reset_state(self) -> None:
        """Reset perception-related state (similar to reset() but perception-only)."""
        self.get_roi = False
        self.roi = ()
        self.detect_color = "None"
        self.rect = None

        self.world_x = self.world_y = 0.0
        self.world_X = self.world_Y = 0.0
        self.rotation_angle = 0.0

        self.track = False
        self.start_pick_up = False

        self.last_x = self.last_y = 0.0
        self.center_list = []
        self.count = 0
        self.start_count_t1 = True
        self.t1 = 0.0

    # ---------------------------
    # High-level tasks as methods
    # ---------------------------
    def draw_crosshair(self, img: np.ndarray) -> None:
        """Task: frame overlay for debugging."""
        h, w = img.shape[:2]
        cv2.line(img, (0, h // 2), (w, h // 2), (0, 0, 200), 1)
        cv2.line(img, (w // 2, 0), (w // 2, h), (0, 0, 200), 1)

    def running_check(self) -> bool:
        """Task: runtime guard."""
        return self.__isRunning

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Task: resize + Gaussian blur."""
        frame_resize = cv2.resize(img, self.size, interpolation=cv2.INTER_NEAREST)
        frame_gb = cv2.GaussianBlur(frame_resize, (11, 11), 11)
        return frame_gb

    def apply_roi_focus(self, frame: np.ndarray) -> np.ndarray:
        """
        Task: ROI focus.
        NOTE: This matches your tracking code condition:
              if get_roi and start_pick_up: mask ROI
        """
        if self.get_roi and self.start_pick_up:
            self.get_roi = False
            if self.getMaskROI is None:
                raise RuntimeError("getMaskROI not provided (SDK helper).")
            frame = self.getMaskROI(frame, self.roi, self.size)
        return frame

    def convert_to_lab(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Task: BGR -> LAB."""
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)

    def segment_target_color(
        self, frame_lab: np.ndarray
    ) -> Tuple[Optional[str], Optional[np.ndarray], float]:
        """
        Task: colour segmentation + morphology + contours for the allowed target colours.
        Tracking version effectively chooses one (the last processed) target colour and returns its largest contour.
        """
        best_contour = None
        best_area = 0.0
        chosen_color = None

        # Only segment when not in pickup phase (same as your code)
        if self.start_pick_up:
            return None, None, 0.0

        for color_name in self.color_range:
            if color_name not in self.target_colors:
                continue

            chosen_color = color_name
            lower, upper = self.color_range[color_name]
            mask = cv2.inRange(frame_lab, lower, upper)

            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))

            contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
            max_contour, area = self.get_area_max_contour(contours)

            # For single-colour tracking this is enough; keep the largest if multiple target colours provided
            if max_contour is not None and area > best_area:
                best_contour = max_contour
                best_area = area
                chosen_color = color_name

        return chosen_color, best_contour, best_area

    def get_area_max_contour(self, contours: List[np.ndarray]) -> Tuple[Optional[np.ndarray], float]:
        """Task helper: find max area contour with noise filter."""
        contour_area_max = 0.0
        area_max_contour = None

        for c in contours:
            area_temp = abs(cv2.contourArea(c))
            if area_temp > contour_area_max:
                contour_area_max = area_temp
                if area_temp > self.contour_valid_area:
                    area_max_contour = c

        return area_max_contour, contour_area_max

    def estimate_pose_and_roi(self, contour: np.ndarray) -> Tuple[Tuple, np.ndarray]:
        """
        Task: pose estimation + ROI extraction.
        Returns (rect, box) and updates ROI state.
        """
        if self.getROI is None:
            raise RuntimeError("getROI not provided (SDK helper).")

        rect = cv2.minAreaRect(contour)
        box = np.int0(cv2.boxPoints(rect))

        self.roi = self.getROI(box)
        self.get_roi = True
        self.rect = rect

        return rect, box

    def compute_center_and_world(self, rect: Tuple) -> Tuple[float, float, int, int]:
        """
        Task: get center in pixels then convert to world coords.
        Returns (world_x, world_y, img_centerx, img_centery)
        """
        if self.getCenter is None or self.convertCoordinate is None:
            raise RuntimeError("getCenter/convertCoordinate not provided (SDK helper).")
        if self.square_length is None:
            raise RuntimeError("square_length not provided (calibration).")

        img_centerx, img_centery = self.getCenter(rect, self.roi, self.size, self.square_length)
        world_x, world_y = self.convertCoordinate(img_centerx, img_centery, self.size)
        return float(world_x), float(world_y), int(img_centerx), int(img_centery)

    def annotate_detection(
        self,
        img: np.ndarray,
        *,
        box: np.ndarray,
        detect_color: str,
        world_x: float,
        world_y: float,
    ) -> None:
        """Task: draw contour box + world coordinate label."""
        color_bgr = self.range_rgb.get(detect_color, self.range_rgb["black"])
        cv2.drawContours(img, [box], -1, color_bgr, 2)
        x_text = min(int(box[0, 0]), int(box[2, 0]))
        y_text = int(box[2, 1]) - 10
        cv2.putText(
            img,
            f"({world_x:.1f},{world_y:.1f})",
            (x_text, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_bgr,
            1,
        )

    def update_tracking_handoff(self) -> None:
        """Task: set tracking flag for motion thread."""
        self.track = True

    def stability_gate_and_pick_trigger(self, *, rect_angle: float) -> None:
        """
        Task: stability filter + pickup trigger.
        Matches the logic in your tracking script:
          - compute distance
          - if action_finish and stable (distance < 0.3) for > 1.5s:
              rotation_angle = rect[2]
              world_X/Y = mean accumulated
              start_pick_up = True
        """
        distance = math.sqrt((self.world_x - self.last_x) ** 2 + (self.world_y - self.last_y) ** 2)
        self.last_x, self.last_y = self.world_x, self.world_y

        if not self.action_finish:
            return

        if distance < self.stable_dist_thresh:
            self.center_list.extend((self.world_x, self.world_y))
            self.count += 1

            if self.start_count_t1:
                self.start_count_t1 = False
                self.t1 = time.time()

            if time.time() - self.t1 > self.stable_time_sec:
                self.rotation_angle = float(rect_angle)
                self.start_count_t1 = True

                pts = np.array(self.center_list).reshape(self.count, 2)
                self.world_X, self.world_Y = np.mean(pts, axis=0).tolist()

                self.count = 0
                self.center_list = []
                self.start_pick_up = True
        else:
            self.t1 = time.time()
            self.start_count_t1 = True
            self.count = 0
            self.center_list = []

    # ---------------------------
    # Full pipeline (like run)
    # ---------------------------
    def run(self, img: np.ndarray) -> Tuple[np.ndarray, TrackingPerceptionOutputs]:
        """
        Process one frame and return:
          - annotated image
          - outputs for motion thread

        This mirrors your tracking run(img) function at a high level.
        """
        out_img = img
        img_copy = img.copy()

        # 1) overlay
        self.draw_crosshair(out_img)

        # 2) runtime guard
        if not self.running_check():
            return out_img, self.outputs()

        # 3) preprocess
        frame = self.preprocess(img_copy)

        # 4) ROI focus (matches tracking script condition)
        frame = self.apply_roi_focus(frame)

        # 5) LAB conversion
        frame_lab = self.convert_to_lab(frame)

        # 6) segmentation for target colour(s)
        color_name, contour, area = self.segment_target_color(frame_lab)
        if color_name is None:
            # nothing selected
            self.track = False
            return out_img, self.outputs()

        self.detect_color = color_name

        # 7) area threshold
        if contour is None or area <= self.min_valid_area:
            self.track = False
            return out_img, self.outputs()

        # 8) pose + ROI
        rect, box = self.estimate_pose_and_roi(contour)

        # 9) centre + world coords (instant)
        self.world_x, self.world_y, _, _ = self.compute_center_and_world(rect)

        # 10) annotate (debug)
        self.annotate_detection(out_img, box=box, detect_color=self.detect_color, world_x=self.world_x, world_y=self.world_y)

        # 11) continuous tracking handoff
        self.update_tracking_handoff()

        # 12) stability gate → pickup trigger
        self.stability_gate_and_pick_trigger(rect_angle=float(rect[2]))

        return out_img, self.outputs()

    def outputs(self) -> TrackingPerceptionOutputs:
        """Pack current state as outputs."""
        return TrackingPerceptionOutputs(
            detect_color=self.detect_color,
            world_x=float(self.world_x),
            world_y=float(self.world_y),
            world_X=float(self.world_X),
            world_Y=float(self.world_Y),
            rotation_angle=float(self.rotation_angle),
            track=bool(self.track),
            start_pick_up=bool(self.start_pick_up),
            roi=self.roi,
            get_roi=bool(self.get_roi),
            rect=self.rect,
        )

"""
Perception-only demo for the *tracking* pipeline:
- Opens the ArmPi camera
- Runs ColorTrackingPerception on each frame
- Draws the detected block outline and labels its WORLD (x,y) on the video
- Shows extra text:
    track flag (live following)
    start_pick_up flag (stable enough to pick)
"""




# --------- Paste/import your class here ----------
# If you saved the class in a file, e.g. tracking_perception.py:
# from tracking_perception import ColorTrackingPerception
#
# For convenience, assume the class ColorTrackingPerception is already available
# in the same script (from the previous message).


def main():
    # Instantiate the perception class.
    # Use target_colors=('red',) to match your tracking script behaviour.
    perception = ColorTrackingPerception(
        size=(640, 480),
        range_rgb={
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
        },
        color_range=color_range,
        target_colors=("red",),          # change to ("red","green","blue") if you want
        min_valid_area=2500.0,
        contour_valid_area=300.0,
        stable_dist_thresh=0.3,
        stable_time_sec=1.5,
        getMaskROI=getMaskROI,
        getROI=getROI,
        getCenter=getCenter,
        convertCoordinate=convertCoordinate,
        square_length=square_length,
    )

    # We are not running the motion thread here, so action_finish is always True.
    perception.set_action_finish(True)
    perception.set_running(True)

    cam = Camera.Camera()
    cam.camera_open()

    try:
        while True:
            img = cam.frame
            if img is None:
                continue

            frame = img.copy()
            annotated, out = perception.run(frame)

            # Extra overlay at the top-left (easy to read)
            if out.track:
                cv2.putText(
                    annotated,
                    f"{out.detect_color}  x={out.world_x:.1f}  y={out.world_y:.1f}  "
                    f"pick={int(out.start_pick_up)}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    annotated,
                    "No target detected",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Tracking Perception Demo", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    finally:
        cam.camera_close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if sys.version_info.major == 2:
        print("Please run this program with python3!")
        sys.exit(0)
    main()