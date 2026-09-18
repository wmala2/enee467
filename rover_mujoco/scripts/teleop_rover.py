"""Drive a rover directly in MuJoCo with a 10 Hz velocity-command loop.

The Rover class turns a forward speed and a turn speed into two wheel speeds.
MuJoCo handles the forces, ground contact, and resulting motion.

    Scene XML -> model (world description) and data (current state)
    Rover -> differential-drive math and wheel motor commands
    Arrow keys -> forward speed in m/s and turn speed in rad/s
    Main loop -> commands at 10 Hz, physics at 500 Hz, drawing at 50 Hz
"""

from pathlib import Path
import time

import mujoco
from scripts.teleop_viewer import TeleopViewer

COMMAND_RATE_HZ = 10.0  # accept a new command at most once every 100 ms
FRAME_RATE_HZ = 50.0  # show motion between commands
SCENE_PATH = Path(__file__).resolve().parents[1] / "assets/robots/rover/rover_scene.xml"


class Rover:
    """Control the rover already placed in the world's XML file.

    Positive forward speed drives toward the front caster; positive turn speed
    turns left (counterclockwise when viewed from above).
    """

    # Dimensions of the rolling collision cylinders in rover.xml, in meters.
    WHEEL_RADIUS_M = 0.03435
    WHEEL_SEPARATION_M = 0.1625589  # distance between the two cylinder centers

    def __init__(self, model, data):
        self.data = data
        # Look up motors by name so their order in the XML does not matter.
        self.left_motor = model.actuator("left_wheel_motor").id
        self.right_motor = model.actuator("right_wheel_motor").id

        # Provide both motors a symmetric +/-10 rad/s limit
        self.max_wheel_speed = min(
            model.actuator_ctrlrange[self.left_motor, 1],
            model.actuator_ctrlrange[self.right_motor, 1],
        )

    def set_velocity(self, forward_mps, turn_rad_s):
        """Set forward speed (m/s) and turn speed (rad/s) until the next command."""
        # In a turn, the outer wheel travels faster than the inner wheel.
        # Each wheel is half the wheel separation away from the rover's midpoint.
        half_separation = self.WHEEL_SEPARATION_M / 2.0
        left_mps = forward_mps - turn_rad_s * half_separation
        right_mps = forward_mps + turn_rad_s * half_separation

        # A wheel's rim speed is radius * angular speed: v = r * omega.
        left_rad_s = left_mps / self.WHEEL_RADIUS_M
        right_rad_s = right_mps / self.WHEEL_RADIUS_M

        # If either wheel is too fast, slow BOTH by the same factor to preserve the turn.
        scale = max(
            1.0, abs(left_rad_s) / self.max_wheel_speed, abs(right_rad_s) / self.max_wheel_speed
        )
        # The CAD model's left axle points the other way, so only its sign is flipped.
        self.data.ctrl[self.left_motor] = -left_rad_s / scale
        self.data.ctrl[self.right_motor] = right_rad_s / scale

    def stop(self):
        """Command both wheels to stop."""
        self.set_velocity(0.0, 0.0)


def main():
    # Load a world containing the rover and a floor, then connect its motors.
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))  # ty: ignore[unresolved-attribute]
    data = mujoco.MjData(model)  # ty: ignore[unresolved-attribute]
    rover = Rover(model, data)
    mujoco.mj_forward(model, data)  # ty: ignore[unresolved-attribute]

    # Ten 2 ms physics steps per frame; five frames per command at the defaults.
    steps_per_frame = max(1, round(1.0 / (FRAME_RATE_HZ * model.opt.timestep)))
    frame_period = steps_per_frame * model.opt.timestep
    command_period = 1.0 / COMMAND_RATE_HZ
    print("Hold arrows to drive; +/- changes speed; Space stops; Esc exits.")

    # The viewer handles the window and keyboard; the rover math stays above.
    with TeleopViewer(model, data) as viewer:
        next_command = time.perf_counter()
        try:
            while viewer.is_running():
                frame_start = time.perf_counter()
                viewer.poll_events()

                # Accept a new command only after 100 ms; hold it between updates.
                if frame_start >= next_command:
                    forward_mps, turn_rad_s = viewer.keyboard_velocity()
                    rover.set_velocity(forward_mps, turn_rad_s)
                    next_command = frame_start + command_period

                # Physics and drawing continue between commands.
                mujoco.mj_step(model, data, nstep=steps_per_frame)  # ty: ignore[unresolved-attribute]
                viewer.draw()
                remaining = frame_period - (time.perf_counter() - frame_start)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            rover.stop()


if __name__ == "__main__":
    main()
