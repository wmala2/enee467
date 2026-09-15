"""Drives the rover live from your keyboard: WASD/arrow keys to move, +/- to change speed.

Instead of reading a sensor and deciding what to do (like the ArUco examples), this rover just
turns whichever keys are currently held down into wheel speeds, every tick, forever.
"""

# External Libraries
from pynput import keyboard as kb

# Local Files to Import
from rover_control import conversions
from rover_control.rover import Rover


class ManualRover(Rover):
    """A rover whose wheel speeds come directly from whatever keys you're holding down."""

    def __init__(self):
        super().__init__()
        # Track which keyboard keys are currently held down
        self.pressed_keys = set()

    def _on_press(self, key):
        # pynput calls this the instant any key goes down; just remember it's held.
        self.pressed_keys.add(key)

    def _on_release(self, key):
        # And this the instant a key comes back up; Esc additionally stops the whole listener.
        self.pressed_keys.discard(key)
        if key == kb.Key.esc:
            return False  # stops the listener

    def compute_wheel_speeds(self):
        """Reads whichever keys are currently held and turns them into [left, right] speeds."""
        speed = [0, 0]

        # Allow user to drive the rover around
        if kb.KeyCode.from_char("w") in self.pressed_keys or kb.Key.up in self.pressed_keys:
            # Drive forward at some desired speed
            speed[0] = self.velocity_bias
            speed[1] = self.velocity_bias
        if kb.KeyCode.from_char("s") in self.pressed_keys or kb.Key.down in self.pressed_keys:
            # Drive backward at some desired speed
            speed[0] = -1 * self.velocity_bias
            speed[1] = -1 * self.velocity_bias
        if kb.KeyCode.from_char("a") in self.pressed_keys or kb.Key.left in self.pressed_keys:
            # Drive left at some desired speed
            speed[0] = -1 * self.velocity_bias
            speed[1] = self.velocity_bias
        if kb.KeyCode.from_char("d") in self.pressed_keys or kb.Key.right in self.pressed_keys:
            # Drive right at some desired speed
            speed[0] = self.velocity_bias
            speed[1] = -1 * self.velocity_bias

        # Allow user to control the desired speed
        if kb.KeyCode.from_char("=") in self.pressed_keys:
            self.velocity_bias = min(self.velocity_bias + 0.05, self.MAX_VELOCITY)
        if kb.KeyCode.from_char("-") in self.pressed_keys:
            self.velocity_bias = max(self.velocity_bias - 0.05, self.MIN_VELOCITY)

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

        speed = self.clamp(speed, min_pos, max_pos)
        return speed

    def update(self):
        """The main loop: read the keyboard, send wheel speeds, repeat until Esc."""
        print("Rover control - WASD to drive, = / - to adjust speed, ESC to quit")

        # pynput's Listener runs in the background and calls _on_press/_on_release for us; we
        # just keep sending whatever compute_wheel_speeds() says while it's still running.
        with kb.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            while listener.running:
                wheel_speeds = self.compute_wheel_speeds()
                self.write_real_velocities(wheel_speeds)
                self.sleep_to_command_rate()

        # Make sure the rover doesn't keep rolling after we quit
        self.stop()


# Create our rover class and start driving
if __name__ == "__main__":
    rover = ManualRover()
    rover.update()
