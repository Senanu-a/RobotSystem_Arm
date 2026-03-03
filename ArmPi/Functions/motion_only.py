import time
import HiwonderSDK.Board as Board
from ArmIK.ArmMoveIK import ArmIK
from ArmIK.Transform import getAngle

class ColorTrackingMotion:
    def __init__(self, servo1=500):
        self.AK = ArmIK()
        self.servo1 = servo1
        self.unreachable = False

        self.drop_bins = {
            'red':   (-14.5, 11.5, 1.5),
            'green': (-14.5, 5.5,  1.5),
            'blue':  (-14.5, -0.5, 1.5),
        }

    # ---- feedback ----
    def set_buzzer(self, t=0.1):
        Board.setBuzzer(0)
        Board.setBuzzer(1)
        time.sleep(t)
        Board.setBuzzer(0)

    def set_rgb(self, color):
        if color == "red":
            Board.RGB.setPixelColor(0, Board.PixelColor(255, 0, 0))
            Board.RGB.setPixelColor(1, Board.PixelColor(255, 0, 0))
        elif color == "green":
            Board.RGB.setPixelColor(0, Board.PixelColor(0, 255, 0))
            Board.RGB.setPixelColor(1, Board.PixelColor(0, 255, 0))
        elif color == "blue":
            Board.RGB.setPixelColor(0, Board.PixelColor(0, 0, 255))
            Board.RGB.setPixelColor(1, Board.PixelColor(0, 0, 255))
        else:
            Board.RGB.setPixelColor(0, Board.PixelColor(0, 0, 0))
            Board.RGB.setPixelColor(1, Board.PixelColor(0, 0, 0))
        Board.RGB.show()

    # ---- poses ----
    def home(self):
        Board.setBusServoPulse(1, self.servo1 - 50, 300)
        Board.setBusServoPulse(2, 500, 500)
        self.AK.setPitchRangeMoving((0, 10, 10), -30, -30, -90, 1500)
        time.sleep(1.5)

    def first_approach(self, color, world_X, world_Y):
        """Move above the object once (your first_move stage)."""
        self.set_rgb(color)
        self.set_buzzer(0.1)
        result = self.AK.setPitchRangeMoving((world_X, world_Y - 2, 5), -90, -90, 0)
        if result is False:
            self.unreachable = True
            return False
        self.unreachable = False
        time.sleep(result[2] / 1000.0)
        return True

    def follow_live(self, color, world_x, world_y):
        """Small fast corrections towards the live position (tracking)."""
        self.set_rgb(color)
        self.AK.setPitchRangeMoving((world_x, world_y - 2, 5), -90, -90, 0, 20)
        time.sleep(0.02)

    def pick(self, world_X, world_Y, rotation_angle):
        """Open, rotate, descend, close, lift."""
        Board.setBusServoPulse(1, self.servo1 - 280, 500)  # open
        servo2_angle = getAngle(world_X, world_Y, rotation_angle)
        Board.setBusServoPulse(2, servo2_angle, 500)
        time.sleep(0.8)

        self.AK.setPitchRangeMoving((world_X, world_Y, 2), -90, -90, 0, 1000)
        time.sleep(2.0)

        Board.setBusServoPulse(1, self.servo1, 500)  # close
        time.sleep(1.0)

        Board.setBusServoPulse(2, 500, 500)
        self.AK.setPitchRangeMoving((world_X, world_Y, 12), -90, -90, 0, 1000)
        time.sleep(1.0)

    def place(self, color):
        """Move to bin, descend, open, lift."""
        x, y, z = self.drop_bins[color]

        result = self.AK.setPitchRangeMoving((x, y, 12), -90, -90, 0)
        time.sleep(result[2] / 1000.0)

        servo2_angle = getAngle(x, y, -90)
        Board.setBusServoPulse(2, servo2_angle, 500)
        time.sleep(0.5)

        self.AK.setPitchRangeMoving((x, y, z + 3), -90, -90, 0, 500)
        time.sleep(0.5)

        self.AK.setPitchRangeMoving((x, y, z), -90, -90, 0, 1000)
        time.sleep(0.8)

        Board.setBusServoPulse(1, self.servo1 - 200, 500)  # release
        time.sleep(0.8)

        self.AK.setPitchRangeMoving((x, y, 12), -90, -90, 0, 800)
        time.sleep(0.8)