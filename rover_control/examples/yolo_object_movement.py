# External Libraries
from pathlib import Path
import time

import cv2

# Local Files to Import
from rover_control import conversions
from rover_control.pid import PID
from rover_control.rover import Rover

# Local Libraries to Import
from YOLO_agent.yolo_pose_estimator import YOLOPoseEstimator


class YoloFollower(Rover):
    TARGET_OBJECT = "bottle"  # the COCO object name we drive toward

    def __init__(self, stop_tolerance_m=0.25):
        super().__init__()

        # How far away from the object (in meters) the rover should stop
        self.stop_tolerance_m = stop_tolerance_m

        # Camera Stream Address
        self.img_addr = "http://192.168.50.123:80/capture"

        # The locally downloaded YOLOv8n weights that ship with the YOLO_agent folder
        model_path = Path(__file__).resolve().parents[2] / "YOLO_agent" / "models" / "yolov8n.pt"

        # Create our YOLO 3D pose estimator (imgsz=960 finds smaller/farther objects)
        self.estimator = YOLOPoseEstimator(
            model_path=model_path,
            imgsz=960,
            verbose=True,  # set False if you only want data
        )

        # PID on the forward distance to the object (+Z is forward), output is forward speed in m/s
        self.distance_pid = PID(kp=0.75, ki=0.10, kd=0.15, output_limit=self.MAX_VELOCITY)

        # PID on the sideways offset of the object (+X is right), output is a turning speed in m/s
        self.heading_pid = PID(kp=0.6, ki=0.0, kd=0.05, output_limit=self.MAX_VELOCITY / 2.0)

        # Remember when we last ran the controller so the PIDs get a real time step
        self._last_pid_time = time.perf_counter()

    def wheel_speeds_for(self, position):
        # Measure the time since the last control step for the PID math
        now = time.perf_counter()
        dt = now - self._last_pid_time
        self._last_pid_time = now

        # The error is how much farther than the stop tolerance the object is (+Z: forward)
        distance_error = position[2] - self.stop_tolerance_m

        # Once we're inside the tolerance, stop and clear the controllers
        if distance_error <= 0:
            self.distance_pid.reset()
            self.heading_pid.reset()
            return [0.0, 0.0]

        # Forward speed comes from the distance PID, turning comes from the sideways offset PID
        forward = self.distance_pid.update(distance_error, dt)
        turn = self.heading_pid.update(position[0], dt)

        # Mix forward and turn for a differential drive (object to the right -> left wheel speeds
        # up)
        speed = [forward + turn, forward - turn]

        # Correctly convert for our pseudo-twist message
        max_pos = conversions.convert_linear_vel_to_angular_vel(
            self.MAX_VELOCITY, self.wheel_diameter / 2.0
        )
        min_pos = conversions.convert_linear_vel_to_angular_vel(
            self.MIN_VELOCITY, self.wheel_diameter / 2.0
        )
        speed[0] = conversions.convert_linear_vel_to_angular_vel(
            speed[0], self.wheel_diameter / 2.0
        )
        speed[1] = conversions.convert_linear_vel_to_angular_vel(
            speed[1], self.wheel_diameter / 2.0
        )

        # Clamp the speed so it doesn't go insane
        speed = self.clamp(speed, min_pos, max_pos)
        return speed

    def update(self):
        print(
            f"Driving to YOLO object '{self.TARGET_OBJECT}', stopping {self.stop_tolerance_m} m "
            f"away - press Q in the window to quit"
        )

        # Running Constantly
        while True:
            # Grab one JPEG still from the ESP32 camera, then find the object and its 3D pose
            try:
                frame = self.estimator.get_frame_from_http(self.img_addr)
            except Exception as error:  # noqa: BLE001 -- one bad frame should not stop the rover mid-drive
                print(f"Camera error: {error}")
                continue
            annotated, detections = self.estimator.process(frame)

            # Keep only our target object, and aim at the nearest one if several show up
            targets = [d for d in detections if d["name"] == self.TARGET_OBJECT]
            if targets:
                nearest = min(targets, key=lambda d: d["position"][2])
                print("Pose : ", nearest["position"])
                wheel_speeds = self.wheel_speeds_for(nearest["position"])
            else:
                # Object not in sight, so hold still
                wheel_speeds = [0.0, 0.0]

            # Optional display if verbose=True
            if annotated is not None:
                cv2.imshow("YOLO Follower", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            self.write_real_velocities(wheel_speeds)
            # Sleep only the leftover time so commands go out at the firmware's rate
            self.sleep_to_command_rate()

        # Make sure the rover doesn't keep rolling after we quit
        self.stop()
        cv2.destroyAllWindows()


# Create our rover class and start following the object
if __name__ == "__main__":
    rover = YoloFollower()
    rover.update()
