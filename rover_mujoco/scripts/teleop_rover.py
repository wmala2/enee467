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

# Usable max kept well under the rated (no-load) speed — loaded speed runs lower in practice,
# and this leaves headroom for the ramping below to actually smooth things out.
MAX_SPEED = MOTOR_MAX_RAD_S * 0.5
# Rough placeholder for the minimum wheel speed needed to overcome static friction and
# actually roll instead of just stalling against the floor. Not measured on real hardware yet
# (that's the BAM/friction-modeling tutorial's job) — treat this as a starting estimate.
MIN_SPEED = 0.5
SPEED_STEP = 0.5  # rad/s adjusted per '+'/'-' press

# Caps how fast the commanded speed can change per second. Without this, a direction key or a
# '+' press would snap straight to the target speed — which is exactly what caused the wheelie
# popups mentioned earlier. This value is tuned by feel in sim, not derived from a measured
# motor acceleration curve (also future BAM work).
MAX_ACCEL = 15.0  # rad/s^2

# Keep the turn-in-place speed gentler than straight-line driving, same ratio the old fixed
# LINEAR_SPEED=5.0 / ANGULAR_SPEED=3.0 constants used, but now scaling with the adjustable speed.
ANGULAR_RATIO = 3.0 / 5.0

# Arrow keys, not WASD: MuJoCo's viewer binds every letter of the alphabet to a built-in
# render/visualization toggle (e.g. "W" is Wireframe, "S" is Shadow) and processes those
# on every keypress regardless of our own key_callback below, so WASD would flip one of
# those every time you drive. GLFW arrow-key and +/- codes (standard, not in either table).
GLFW_KEY_RIGHT, GLFW_KEY_LEFT, GLFW_KEY_DOWN, GLFW_KEY_UP = 262, 263, 264, 265
GLFW_KEY_SPACE = 32
GLFW_KEY_EQUAL, GLFW_KEY_MINUS = 61, 45  # '+' is shift+'=' on most layouts; GLFW reports
# the unshifted physical key, so '=' is what arrives

# Target drive direction (-1/0/1 per axis), set by the arrow keys/space
drive_state = {"linear_dir": 0, "angular_dir": 0}
# Adjustable top speed (rad/s), set by '+'/'-', clamped to what the real motors can do
speed_state = {"speed": 5.0}
# Actual, ramped ctrl values the render loop drives toward the target above (see MAX_ACCEL)
applied_state = {"linear": 0.0, "angular": 0.0}


def on_key(keycode):
    if keycode == GLFW_KEY_UP:
        drive_state["linear_dir"] = 1
    elif keycode == GLFW_KEY_DOWN:
        drive_state["linear_dir"] = -1
    elif keycode == GLFW_KEY_LEFT:
        drive_state["angular_dir"] = 1
    elif keycode == GLFW_KEY_RIGHT:
        drive_state["angular_dir"] = -1
    elif keycode == GLFW_KEY_SPACE:  # Space bar stops the rover
        drive_state["linear_dir"] = 0
        drive_state["angular_dir"] = 0
    elif keycode == GLFW_KEY_EQUAL:
        speed_state["speed"] = min(speed_state["speed"] + SPEED_STEP, MAX_SPEED)
        print(f"speed: {speed_state['speed']:.1f} rad/s")
    elif keycode == GLFW_KEY_MINUS:
        speed_state["speed"] = max(speed_state["speed"] - SPEED_STEP, MIN_SPEED)
        print(f"speed: {speed_state['speed']:.1f} rad/s")


def ramp_toward(current, target, max_delta):
    if target > current:
        return min(current + max_delta, target)
    return max(current - max_delta, target)


def main():
    env = gym.make("Rover-v0")
    env.reset()

    # Access underlying MuJoCo model and data structures
    model = env.unwrapped.model  # ty: ignore[unresolved-attribute]
    data = env.unwrapped.data  # ty: ignore[unresolved-attribute]

    print("Arrow keys to drive, +/- to adjust speed, Space to stop, Esc to exit.")

    # Launch native OpenGL GUI viewer, routing keyboard input through on_key
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Ramp the actual applied speed toward the target rather than snapping to it,
            # so changing direction or speed doesn't jolt the chassis into a wheelie
            max_delta = MAX_ACCEL * model.opt.timestep
            speed = speed_state["speed"]
            target_linear = drive_state["linear_dir"] * speed
            target_angular = drive_state["angular_dir"] * speed * ANGULAR_RATIO
            applied_state["linear"] = ramp_toward(applied_state["linear"], target_linear, max_delta)
            applied_state["angular"] = ramp_toward(
                applied_state["angular"], target_angular, max_delta
            )

            # Mix linear/angular drive state into left/right wheel velocity targets
            linear, angular = applied_state["linear"], applied_state["angular"]
            action = np.array([-linear + angular, linear + angular], dtype=np.float32)
            env.step(action)

            # Synchronize 3D graphics state with physics data
            viewer.sync()

            # Maintain real-time wall clock rate based on model timestep
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    env.close()


if __name__ == "__main__":
    main()
