# Sim Manual Control

This example opens the simulated rover in the MuJoCo 3D viewer and lets you drive it around with the arrow keys, so you can confirm your simulation setup works and see how differential-drive steering behaves before writing any control code of your own.

## Prerequisites

- A `uv sync` of the workspace (from the repository root) — this pulls in `mujoco`, `mujoco.viewer`, `gymnasium`, and `numpy`, the only imports the script uses.
- A display: `mujoco.viewer.launch_passive` opens a native OpenGL window, so this won't run over a headless/SSH session without a virtual display.
- Nothing else — no trained policy, no generated track, no GPU. The rover model (`assets/robots/rover/rover_scene.xml`) ships in the repo and is loaded through the `Rover-v0` Gymnasium environment registered by `envs/__init__.py`.

## Run it

```bash
uv run rover_mujoco/scripts/teleop_rover.py
```

A MuJoCo viewer window opens with the rover sitting on the floor; arrow keys drive it, `+`/`-` change the top speed, Space stops it, and Esc closes the window.

![The simulated rover sitting on the floor of the default scene](images/sim_manual_control.png)

## How it works

The script keeps three small dictionaries of mutable state (`drive_state`, `speed_state`, `applied_state`) instead of a class, since everything here runs in one process on one thread — the key callback just writes into them and the render loop reads them back out.

```python
# Target drive direction (-1/0/1 per axis), set by the arrow keys/space
drive_state = {"linear_dir": 0, "angular_dir": 0}
# Adjustable top speed (rad/s), set by '+'/'-', clamped to what the real motors can do
speed_state = {"speed": 5.0}
# Actual, ramped ctrl values the render loop drives toward the target above (see MAX_ACCEL)
applied_state = {"linear": 0.0, "angular": 0.0}
```

Why arrow keys and not WASD? MuJoCo's own viewer already binds every letter to a built-in rendering toggle (`W` for wireframe, `S` for shadows, and so on) and runs those bindings on every keypress no matter what your own callback does — so driving with WASD would silently flip rendering settings while you drive. The GLFW key codes below are just the numeric codes for the arrow keys, space, and `+`/`-`, since MuJoCo's callback hands you raw keycodes rather than names.

```python
GLFW_KEY_RIGHT, GLFW_KEY_LEFT, GLFW_KEY_DOWN, GLFW_KEY_UP = 262, 263, 264, 265
GLFW_KEY_SPACE = 32
GLFW_KEY_EQUAL, GLFW_KEY_MINUS = 61, 45  # '+' is shift+'=' on most layouts; GLFW reports
# the unshifted physical key, so '=' is what arrives
```

`on_key` is the callback MuJoCo calls on every keypress. It doesn't move the rover directly — it only updates the *target* direction and speed, which is what lets the render loop below smoothly ramp toward it instead of snapping.

```python
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
```

Real motors can't jump from stopped to full speed instantly — and neither can MuJoCo's physics without looking wrong. Commanding a big torque change in a single simulation step used to pop the rover into a wheelie. `ramp_toward` is the fix: it nudges a value toward its target by at most `max_delta` per call, so speed and direction changes glide instead of snap.

```python
def ramp_toward(current, target, max_delta):
    if target > current:
        return min(current + max_delta, target)
    return max(current - max_delta, target)
```

`main` builds the Gymnasium environment, grabs MuJoCo's underlying `model`/`data` objects directly (useful for reading the physics timestep), and opens the viewer with `on_key` wired in as the keyboard handler.

```python
def main():
    env = gym.make("Rover-v0")
    env.reset()

    # Access underlying MuJoCo model and data structures
    model = env.unwrapped.model  # ty: ignore[unresolved-attribute]
    data = env.unwrapped.data  # ty: ignore[unresolved-attribute]

    print("Arrow keys to drive, +/- to adjust speed, Space to stop, Esc to exit.")

    # Launch native OpenGL GUI viewer, routing keyboard input through on_key
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
```

Each pass through the loop ramps the applied speed toward the target, then mixes the linear/angular drive state into left/right wheel commands. The leading `-` on the left wheel isn't textbook differential-drive math — it's because this rover's CAD wheel meshes are mirrored, so the two wheel joints spin in opposite senses for the same physical rolling direction.

```python
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
```

Finally, the loop syncs the 3D viewer to the new physics state and sleeps just long enough to keep the simulation running at real-time speed, rather than as fast as the CPU can go.

```python
            # Synchronize 3D graphics state with physics data
            viewer.sync()

            # Maintain real-time wall clock rate based on model timestep
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
```

## See also

- [rover_mujoco/](../../rover_mujoco/) — the simulation environment this example depends on
- [Back to README](../../README.md)
