"""Window and keyboard support for teleop_rover.py.

This helper owns the display, not the rover or its physics loop. GLFW supplies
real key-release events so a held arrow never depends on keyboard-repeat timing.
"""

import glfw
import mujoco


class TeleopViewer:
    """Show the world and read velocity requests from the keyboard."""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.pressed_keys = set()
        self.speed_mps = 0.15
        self.turn_speed_rad_s = 1.0
        self.context = None

    def __enter__(self):
        if not glfw.init():
            raise RuntimeError("GLFW could not open a display")
        try:
            self.window = glfw.create_window(1000, 750, "Rover teleop", None, None)
            if not self.window:
                raise RuntimeError("GLFW could not create the rover window")
            glfw.make_context_current(self.window)
            glfw.swap_interval(0)  # the example's loop sets the frame rate
            glfw.set_key_callback(self.window, self.on_key)
            glfw.set_window_focus_callback(
                self.window, lambda window, focused: self.pressed_keys.clear()
            )
            self.camera = mujoco.MjvCamera()  # ty: ignore[unresolved-attribute]
            self.camera.lookat[:] = [0, 0, 0]
            self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.5, 90, -50
            self.scene = mujoco.MjvScene(self.model, maxgeom=1000)  # ty: ignore[unresolved-attribute]
            self.context = mujoco.MjrContext(self.model, 150)  # ty: ignore[unresolved-attribute]
            self.options = mujoco.MjvOption()  # ty: ignore[unresolved-attribute]
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Release graphics resources while the OpenGL context still exists."""
        if self.context is not None:
            self.context.free()
            self.context = None
        glfw.terminate()

    def is_running(self):
        return not glfw.window_should_close(self.window)

    def poll_events(self):
        """Collect keyboard and window events without advancing physics."""
        glfw.poll_events()

    def on_key(self, window, key, scancode, action, modifiers):
        """Track held keys; repeated presses do not interrupt a driving command."""
        if action == glfw.RELEASE:
            self.pressed_keys.discard(key)
        elif action == glfw.PRESS:
            self.pressed_keys.add(key)
            if key == glfw.KEY_SPACE:
                self.pressed_keys.clear()
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key in (glfw.KEY_EQUAL, glfw.KEY_MINUS):
                change = 0.025 if key == glfw.KEY_EQUAL else -0.025
                self.speed_mps = min(0.25, max(0.025, self.speed_mps + change))
                print(f"Forward/reverse speed: {self.speed_mps:.3f} m/s")
            elif key in (glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_LEFT, glfw.KEY_RIGHT):
                forward_mps, turn_rad_s = self.keyboard_velocity()
                print(
                    f"Requested command: forward={forward_mps:+.3f} m/s, "
                    f"turn={turn_rad_s:+.3f} rad/s"
                )

    def keyboard_velocity(self):
        """Return requested forward speed (m/s) and left-turn speed (rad/s)."""
        forward = (glfw.KEY_UP in self.pressed_keys) - (glfw.KEY_DOWN in self.pressed_keys)
        turn = (glfw.KEY_LEFT in self.pressed_keys) - (glfw.KEY_RIGHT in self.pressed_keys)
        return self.speed_mps * forward, self.turn_speed_rad_s * turn

    def draw(self):
        """Draw the current state without changing wheel targets or advancing physics."""
        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)  # ty: ignore[unresolved-attribute]
        mujoco.mjv_updateScene(  # ty: ignore[unresolved-attribute]
            self.model,
            self.data,
            self.options,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,  # ty: ignore[unresolved-attribute]
            self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)  # ty: ignore[unresolved-attribute]
        glfw.swap_buffers(self.window)
