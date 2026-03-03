#!/usr/bin/python3
# coding=utf8
import sys
sys.path.append('/home/pi/ArmPi/')

import cv2
import time
import math
import numpy as np

import Camera
from LABConfig import *   # match original behaviour exactly
from ArmIK.Transform import *
from CameraCalibration.CalibrationConfig import *


class ColorTrackingPerception:
    """
    Perception-only version of the original ColorTracking run(img).
    Logic matches the original code as closely as possible.
    """

    def __init__(self, target_colors=('red',), size=(640, 480)):
        self.size = size
        self.__target_color = target_colors

        # original globals/state
        self.roi = ()
        self.rect = None
        self.get_roi = False

        self.detect_color = 'None'
        self.track = False
        self.start_pick_up = False
        self.action_finish = True

        self.rotation_angle = 0
        self.world_X, self.world_Y = 0, 0
        self.world_x, self.world_y = 0, 0

        self.center_list = []
        self.count = 0
        self.start_count_t1 = True
        self.t1 = 0

        self.last_x, self.last_y = 0, 0

        self.__isRunning = True  # perception demo always running

        self.range_rgb = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
        }

    # ---------- High-level blocks as methods ----------
    def draw_crosshair(self, img):
        h, w = img.shape[:2]
        cv2.line(img, (0, int(h / 2)), (w, int(h / 2)), (0, 0, 200), 1)
        cv2.line(img, (int(w / 2), 0), (int(w / 2), h), (0, 0, 200), 1)

    def preprocess(self, img):
        frame_resize = cv2.resize(img, self.size, interpolation=cv2.INTER_NEAREST)
        frame_gb = cv2.GaussianBlur(frame_resize, (11, 11), 11)
        return frame_gb

    def apply_roi_focus(self, frame_gb):
        # match original tracking condition
        if self.get_roi and self.start_pick_up:
            self.get_roi = False
            frame_gb = getMaskROI(frame_gb, self.roi, self.size)
        return frame_gb

    def to_lab(self, frame_gb):
        return cv2.cvtColor(frame_gb, cv2.COLOR_BGR2LAB)

    def segment_and_find_contour(self, frame_lab):
        # match original
        area_max = 0
        areaMaxContour = 0

        if not self.start_pick_up:
            for i in color_range:
                if i in self.__target_color:
                    self.detect_color = i
                    frame_mask = cv2.inRange(frame_lab,
                                             color_range[self.detect_color][0],
                                             color_range[self.detect_color][1])
                    opened = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))
                    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))
                    contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
                    areaMaxContour, area_max = getAreaMaxContour(contours)

        return areaMaxContour, area_max

    def estimate_pose_and_world(self, contour):
        self.rect = cv2.minAreaRect(contour)
        box = np.int0(cv2.boxPoints(self.rect))

        self.roi = getROI(box)
        self.get_roi = True

        img_centerx, img_centery = getCenter(self.rect, self.roi, self.size, square_length)
        self.world_x, self.world_y = convertCoordinate(img_centerx, img_centery, self.size)

        return box

    def annotate(self, img, box):
        cv2.drawContours(img, [box], -1, self.range_rgb[self.detect_color], 2)
        cv2.putText(
            img,
            '(' + str(self.world_x) + ',' + str(self.world_y) + ')',
            (min(box[0, 0], box[2, 0]), box[2, 1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.range_rgb[self.detect_color],
            1
        )

    def stability_gate(self):
        distance = math.sqrt(pow(self.world_x - self.last_x, 2) + pow(self.world_y - self.last_y, 2))
        self.last_x, self.last_y = self.world_x, self.world_y

        self.track = True  # match original: set whenever detection exists

        if self.action_finish:
            if distance < 0.3:
                self.center_list.extend((self.world_x, self.world_y))
                self.count += 1

                if self.start_count_t1:
                    self.start_count_t1 = False
                    self.t1 = time.time()

                if time.time() - self.t1 > 1.5:
                    self.rotation_angle = self.rect[2]
                    self.start_count_t1 = True
                    self.world_X, self.world_Y = np.mean(np.array(self.center_list).reshape(self.count, 2), axis=0)
                    self.count = 0
                    self.center_list = []
                    self.start_pick_up = True
            else:
                self.t1 = time.time()
                self.start_count_t1 = True
                self.count = 0
                self.center_list = []

    # ---------- Main perception call ----------
    def run(self, img):
        self.draw_crosshair(img)

        if not self.__isRunning:
            return img

        frame_gb = self.preprocess(img.copy())
        frame_gb = self.apply_roi_focus(frame_gb)
        frame_lab = self.to_lab(frame_gb)

        contour, area = self.segment_and_find_contour(frame_lab)

        if area > 2500:
            box = self.estimate_pose_and_world(contour)
            self.annotate(img, box)
            self.stability_gate()
        else:
            self.track = False

        return img


def main():
    perception = ColorTrackingPerception(target_colors=('red',), size=(640, 480))

    cam = Camera.Camera()
    cam.camera_open()

    try:
        while True:
            img = cam.frame
            if img is None:
                continue

            frame = img.copy()
            out = perception.run(frame)

            # Top-left status
            if perception.track:
                cv2.putText(
                    out,
                    f"{perception.detect_color} x={perception.world_x:.1f} y={perception.world_y:.1f} pick={int(perception.start_pick_up)}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    out,
                    "No target detected",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("ColorTracking Perception Only", out)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        cam.camera_close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if sys.version_info.major == 2:
        print("Please run this program with python3!")
        sys.exit(0)
    main()