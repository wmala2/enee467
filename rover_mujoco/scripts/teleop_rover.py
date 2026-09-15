import time

import envs  # noqa: F401  (imported for its side effect: registers Rover-v0)
import gymnasium as gym
import mujoco.viewer
import numpy as np

# --- Physical limits (see docs/onshape-to-robot-mjcf.md's sibling tutorials for context) ---
# Rated speed of the real rover's JGA25-371 gearmotors, assumed to be the wheel-shaft output
# (no extra external gearing between motor and wheel). This is the hard ceiling: commanding
# more than this is asking for something the real hardware can't do.
MOTOR_RATED_RPM = 463
MOTOR_MAX_RAD_S = MOTOR_RATED_RPM * 2 * np.pi / 60  # ~48.5 rad/s

# Usable max kept well under the rated (no-load) speed - loaded speed runs lower in practice.
MAX_SPEED = MOTOR_MAX_RAD_S * 0.5
# Rough placeholder for the minimum wheel speed needed to overcome static friction and
# actually roll instead of just stalling against the floor. Not measured on real hardware yet
# (that's the BAM/friction-modeling tutorial's job) - treat this as a starting estimate.
MIN_SPEED = 0.5
SPEED_STEP = 0.5  # rad/s adjusted per '+'/'-' press

# Keep the turn-in-place speed gentler than straight-line driving.
ANGULAR_RATIO = 3.0 / 5.0

# The real rover's firmware listens for commands at about this rate (see rover_control/
# rover.py's COMMAND_RATE_HZ) - matching it here means driving in sim behaves like the
# real thing: each tick, we send one wheel-speed command and let it run for one tick.
COMMAND_RATE_HZ = 10.0
COMMAND_PERIOD_S = 1.0 / COMMAND_RATE_HZ

# Arrow keys, not WASD: MuJoCo's viewer binds every letter of the alphabet to a built-in
# render/visualization toggle (e.g. "W" is Wireframe, "S" is Shadow) and processes those
# on every keypress regardless of our own key_callback below, so WASD would flip one of
# those every time you drive. GLFW arrow-key and +/- codes (standard, not in either table).
GLFW_KEY_RIGHT, GLFW_KEY_LEFT, GLFW_KEY_DOWN, GLFW_KEY_UP = 262, 263, 264, 265
GLFW_KEY_SPACE = 32
GLFW_KEY_EQUAL, GLFW_KEY_MINUS = 61, 45  # '+' is shift+'=' on most layouts; GLFW reports
# the unshifted physical key, so '=' is what arrives

speed = 5.0  # top speed (rad/s), adjustable at runtime with +/-

# Timestamp of the most recent press for each direction key. A key counts as "held" if
# it was pressed within the last command period - GLFW re-fires the press while a key
# stays down, so holding a key keeps refreshing its timestamp every tick.
last_up = last_down = last_left = last_right = -1.0


def on_key(keycode):
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
    elif keycode == GLFW_KEY_SPACE:  # Space bar stops the rover
        last_up = last_down = last_left = last_right = -1.0
    elif keycode == GLFW_KEY_EQUAL:
        speed = min(speed + SPEED_STEP, MAX_SPEED)
        print(f"speed: {speed:.1f} rad/s")
    elif keycode == GLFW_KEY_MINUS:
        speed = max(speed - SPEED_STEP, MIN_SPEED)
        print(f"speed: {speed:.1f} rad/s")


def held(last_press_time, now):
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
