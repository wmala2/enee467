"""Seeded, provisional DR ranges and BAM wheel dynamics for the PPO line follower."""

from collections import deque
import math

import mujoco
import numpy as np

from envs import motor

# These engineering priors are deliberately recorded as ranges, not claimed bench fits.
RANGES = {
    "mass_scale": [0.9, 1.1],
    "contact_friction": [0.6, 1.2],
    "voltage": [10.8, 12.6],
    "kt_scale": [0.9, 1.1],
    "resistance_scale": [0.9, 1.1],
    "kp_scale": [0.8, 1.2],
    "friction_scale": [0.5, 1.5],
    "camera_x_m": [-0.003, 0.003],
    "camera_y_m": [-0.003, 0.003],
    "camera_z_m": [-0.002, 0.002],
    "camera_tilt_deg": [42.0, 48.0],
    "camera_fovy_deg": [57.0, 63.0],
    "brightness": [0.85, 1.15],
    "pixel_noise_std": [0.0, 2.0],
    "encoder_noise_std": [0.0, 0.1],
    "command_delay_steps": [0, 1],
}
NOMINAL = dict.fromkeys(RANGES, 1.0)
NOMINAL.update(
    voltage=12.0,
    camera_x_m=0.0,
    camera_y_m=0.0,
    camera_z_m=0.0,
    camera_tilt_deg=45.0,
    camera_fovy_deg=60.0,
    pixel_noise_std=0.0,
    encoder_noise_std=0.0,
    command_delay_steps=0,
)
WHEEL_PARAMETERS = {"kt_scale", "resistance_scale", "kp_scale", "friction_scale"}
# Use the CAD radius consistently when translating the firmware's linear command envelope.
WHEEL_RADIUS_M = 0.03435
MIN_WHEEL_RAD_S = 0.075 / WHEEL_RADIUS_M
MAX_WHEEL_RAD_S = 0.25 / WHEEL_RADIUS_M


def validate_ranges(overrides=None):
    """Resolve a JSON range override and reject physically invalid or misspelled parameters."""
    ranges = {key: list(value) for key, value in RANGES.items()}
    for key, bounds in (overrides or {}).items():
        if key not in ranges:
            raise ValueError(f"Unknown DR parameter: {key}")
        ranges[key] = bounds
    for key, bounds in ranges.items():
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError(f"Invalid DR bounds for {key}: {bounds}")
        if key.startswith("camera_") and key.endswith("_m"):
            continue
        if key in ("command_delay_steps", "pixel_noise_std", "encoder_noise_std"):
            if bounds[0] < 0:
                raise ValueError(f"{key} must be nonnegative")
        elif bounds[0] <= 0:
            raise ValueError(f"{key} must be positive")
    if any(int(value) != value for value in ranges["command_delay_steps"]):
        raise ValueError("command_delay_steps requires integer bounds")
    if ranges["camera_fovy_deg"][1] >= 180:
        raise ValueError("camera_fovy_deg must be below 180")
    return ranges


class LineDynamics:
    """Apply reset-level DR and update the motor model at every physics substep."""

    def __init__(self, model, mode, ranges=None):
        self.model = model
        self.mode = mode
        self.ranges = validate_ranges(ranges)
        self.dofs = np.array([model.joint(name).dofadr[0] for name in ("left_axle", "right_axle")])
        self.actuators = [
            model.actuator(name).id for name in ("left_wheel_motor", "right_wheel_motor")
        ]
        self.camera = model.camera("top_cam")
        self.mass = model.body_mass.copy()
        self.inertia = model.body_inertia.copy()
        self.camera_pos = self.camera.pos.copy()
        self.frictions = [motor.make_friction_model(), motor.make_friction_model()]

    def reset(self, rng, data):
        # Draw independent motor properties but share chassis, camera, and supply parameters.
        self.rng = rng
        self.parameters = {}
        for key, (low, high) in self.ranges.items():
            size = 2 if key in WHEEL_PARAMETERS else None
            if self.mode == "dr":
                value = (
                    rng.integers(low, high + 1)
                    if key == "command_delay_steps"
                    else rng.uniform(low, high, size=size)
                )
            else:
                value = np.full(2, NOMINAL[key]) if size else NOMINAL[key]
            self.parameters[key] = np.asarray(value).tolist()
        p = self.parameters
        # Restore absolute baselines before scaling so successive resets cannot accumulate drift.
        self.model.body_mass[:] = self.mass * p["mass_scale"]
        self.model.body_inertia[:] = self.inertia * p["mass_scale"]
        # Set both sides of every contact because MuJoCo mixes equal-priority friction by maximum.
        self.model.geom_friction[:, 0] = p["contact_friction"]
        self.camera.pos[:] = self.camera_pos + [p[f"camera_{axis}_m"] for axis in "xyz"]
        angle = math.radians(p["camera_tilt_deg"]) / 2
        self.camera.quat[:] = [0, 0, math.sin(angle), math.cos(angle)]
        self.camera.fovy[:] = p["camera_fovy_deg"]
        mujoco.mj_setConst(self.model, data)  # ty: ignore[unresolved-attribute]
        for index, friction in enumerate(self.frictions):
            scale = p["friction_scale"][index] * motor.STALL_TORQUE_OUTPUT_NM
            friction.friction_base.value = 0.1 * scale
            friction.friction_stribeck.value = 0.1 * scale
            friction.friction_viscous.value = 0.05 * scale
        self.electrical = {
            "vin": p["voltage"],
            "kt": motor.KT * np.array(p["kt_scale"]),
            "resistance": motor.R * np.array(p["resistance_scale"]),
            "kp": motor.VELOCITY_KP * np.array(p["kp_scale"]),
        }
        self.commands = deque(np.zeros(2) for _ in range(int(p["command_delay_steps"])))

    def advance(self, data, wheel_targets, substeps, *, settling=False):
        # The firmware envelope is applied to physical wheel targets before the CAD sign change.
        targets = motor.apply_actuator_envelope(wheel_targets, MAX_WHEEL_RAD_S, MIN_WHEEL_RAD_S)
        if not settling:
            self.commands.append(targets.copy())
            targets = self.commands.popleft()
        targets = targets * [-1, 1]
        for _ in range(substeps):
            speeds = data.qvel[self.dofs]
            drive, damping = motor.motor_drive_and_damping(targets, speeds, **self.electrical)
            for index, friction in enumerate(self.frictions):
                motor.apply_friction(
                    friction, self.model, self.dofs[index], speeds[index], damping[index]
                )
            data.ctrl[self.actuators] = drive
            mujoco.mj_step(self.model, data)  # ty: ignore[unresolved-attribute]

    def sensors(self, frame, speeds):
        # Corrupt pixels before feature extraction so line visibility responds to image quality.
        p = self.parameters
        if p["brightness"] != 1 or p["pixel_noise_std"]:
            pixels = frame.astype(float) * p["brightness"]
            pixels += self.rng.normal(0, p["pixel_noise_std"], frame.shape)
            frame = np.clip(pixels, 0, 255).astype(np.uint8)
        if p["encoder_noise_std"]:
            speeds = speeds + self.rng.normal(0, p["encoder_noise_std"], 2)
        return frame, speeds
