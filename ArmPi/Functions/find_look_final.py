#!/usr/bin/python3
# coding=utf8
import sys
sys.path.append('/home/pi/ArmPi/')

import cv2
import time
import Camera

# ---- import your perception class (the one you built) ----
# from tracking_perception import ColorTrackingPerception
# (Or paste the class in the same file)

from LABConfig import *
from ArmIK.Transform import *
from CameraCalibration.CalibrationConfig import *

# ---- motion class from above ----
# (Paste ColorTrackingMotion class here, or import it)
# from motion_class import ColorTrackingMotion


def main():
    perception = ColorTrackingPerception(target_colors=('red',), size=(640, 480))
    motion = ColorTrackingMotion(servo1=500)

    motion.home()

    cam = Camera.Camera()
    cam.camera_open()

    try:
        while True:
            img = cam.frame
            if img is None:
                continue

            frame = img.copy()
            out = perception.run(frame)

            # 1) follow live updates (optional)
            if perception.track and (not perception.first_move) and (not perception.start_pick_up):
                motion.follow_live(perception.detect_color, perception.world_x, perception.world_y)

            # 2) if stable trigger happens
            if perception.start_pick_up and perception.detect_color != 'None':

                # first approach stage (only once)
                if perception.first_move:
                    ok = motion.first_approach(perception.detect_color, perception.world_X, perception.world_Y)
                    if not ok:
                        # unreachable; you can decide what to do here
                        perception.start_pick_up = False
                        continue
                    perception.first_move = False
                    perception.start_pick_up = False
                    continue

                # pick + place
                motion.pick(perception.world_X, perception.world_Y, perception.rotation_angle)
                motion.place(perception.detect_color)
                motion.home()

                # IMPORTANT: reset perception state so it can detect again
                perception.first_move = True
                perception.start_pick_up = False
                perception.track = False
                perception.get_roi = False

            # display
            cv2.imshow("PickAndPlace", out)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    finally:
        cam.camera_close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()