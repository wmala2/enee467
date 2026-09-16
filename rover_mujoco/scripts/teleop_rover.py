import time

import envs  # noqa: F401  (imported for its side effect: registers Rover-v0)
import gymnasium as gym
import mujoco.viewer
import numpy as np

# Physical limits, in rad/s (see docs/onshape-to-robot-mjcf.md for where these come from)
MOTOR_RATED_RPM = 463  # real motor's rated speed - the hard ceiling for a commanded wheel speed
MOTOR_MAX_RAD_S = MOTOR_RATED_RPM * 2 * np.pi / 60  # ~48.5 rad/s
MAX_SPEED = MOTOR_MAX_RAD_S * 0.5  # stay well under the rated (no-load) speed
MIN_SPEED = 0.5  # rough floor so the wheels roll instead of stalling; not yet measured
SPEED_STEP = 0.5  # rad/s adjusted per '+'/'-' press

ANGULAR_RATIO = 3.0 / 5.0  # turning in place is gentler than driving straight

# The real rover's firmware listens for commands at about this rate (see rover_control/
# rover.py's COMMAND_RATE_HZ) - matching it here means driving in sim behaves like the
# real thing: each tick, we send one wheel-speed command and let it run for one tick.
COMMAND_RATE_HZ = 10.0
COMMAND_PERIOD_S = 1.0 / COMMAND_RATE_HZ

# Arrow keys, not WASD - MuJoCo's viewer already binds every letter key to its own shortcuts
# (e.g. "W" toggles Wireframe), so WASD would fight with it. These are GLFW's key codes.
GLFW_KEY_RIGHT, GLFW_KEY_LEFT, GLFW_KEY_DOWN, GLFW_KEY_UP = 262, 263, 264, 265
GLFW_KEY_SPACE = 32
GLFW_KEY_EQUAL, GLFW_KEY_MINUS = 61, 45  # '+' is shift+'=', but GLFW reports the physical '='

speed = 5.0  # top speed (rad/s), adjustable at runtime with +/-

last_up = last_down = last_left = last_right = -1.0  # timestamp of each key's last press


def on_key(keycode):
    """MuJoCo's viewer calls this once every time a key is pressed.

    It doesn't drive the rover directly - it just records *that* a key was pressed, by
    stamping the current time into that key's `last_*` variable. `main()`'s loop is what
    actually reads those timestamps and turns them into wheel speeds.
    """
    # `global` is required to assign to these - without it, `speed = ...` below would create
    # a new local variable instead of updating the module-level one everything else reads.
    global speed, last_up, last_down, last_left, last_right
    now = time.perf_counter()
    if keycode == GLFW_KEY_UP:
        last_up = now
    elif keycode == GLFW_KEY_DOWN:
        last_down = now
    elif keycode == GLFW_KEY_LEFT:
        last_left = now
    elif keycode == GLFW_KEY_RIGHT:
        last_right = now
    elif keycode == GLFW_KEY_SPACE:
        # Stop: forget every direction at once, same as if no key had ever been pressed.
        last_up = last_down = last_left = last_right = -1.0
    elif keycode == GLFW_KEY_EQUAL:
        speed = min(speed + SPEED_STEP, MAX_SPEED)  # clamp so +/- can't exceed the motor limits
        print(f"speed: {speed:.1f} rad/s")
    elif keycode == GLFW_KEY_MINUS:
        speed = max(speed - SPEED_STEP, MIN_SPEED)
        print(f"speed: {speed:.1f} rad/s")


def held(last_press_time, now):
    """Was this key pressed recently enough to still count as "held down"?

    GLFW keeps calling on_key() over and over for as long as a key stays down (keyboard
    auto-repeat), which keeps refreshing that key's timestamp. So "held" just means "the
    timestamp is fresher than one command period" - if you let go, the timestamp stops
    updating and quickly falls outside that window.
    """
    return now - last_press_time <= COMMAND_PERIOD_S


def main():
    env = gym.make("Rover-v0")
    env.reset()

    print("Arrow keys to drive, +/- to adjust speed, Space to stop, Esc to exit.")

    # Launch native OpenGL GUI viewer, routing keyboard input through on_key
    with mujoco.viewer.launch_passive(
        env.unwrapped.model,  # ty: ignore[unresolved-attribute]
        env.unwrapped.data,  # ty: ignore[unresolved-attribute]
        key_callback=on_key,
    ) as viewer:
        while viewer.is_running():
            tick_start = time.perf_counter()

            # A rover drives forward/backward by spinning both wheels the same speed, and
            # turns by slowing one side down while speeding the other up. So work out how
            # fast to go straight and how fast to turn, then mix those into a left/right
            # wheel speed (left is negated to match this rover's wiring - see
            # rover_control/rover.py's ENCODER_SIGNS for the same convention on hardware).
            linear = speed * (held(last_up, tick_start) - held(last_down, tick_start))
            turn = held(last_left, tick_start) - held(last_right, tick_start)
            angular = speed * ANGULAR_RATIO * turn
            left_speed = -linear + angular
            right_speed = linear + angular

            # Send that command and let it run for one loop period, same as the real rover
            env.step(np.array([left_speed, right_speed], dtype=np.float32))
            viewer.sync()

            # Maintain real-time wall clock rate based on the command period
            elapsed = time.perf_counter() - tick_start
            if elapsed < COMMAND_PERIOD_S:
                time.sleep(COMMAND_PERIOD_S - elapsed)

    env.close()


if __name__ == "__main__":
    main()
