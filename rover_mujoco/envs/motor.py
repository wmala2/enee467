"""DC motor model for the rover's JGA25-371 drive wheels, built on BAM's friction model
(https://github.com/Rhoban/bam) plus a hand-derived electrical model from the motor's
datasheet. See docs/rl-line-follower.md for the full derivation, why BAM's own Actuator/
MujocoController classes aren't used directly (they assume a position-controlled servo;
these are continuously-driven wheels under velocity control), and the caveats on the numbers
below (rough datasheet specs, not bench-measured, and BAM's friction parameters are its
un-fit defaults — real identification needs recorded trajectories we don't have yet).

Datasheet: https://www.openimpulse.com/blog/products-page/25d-gearmotors/jga25-371-dc-gearmotor-encoder-463-rpm-12-v-2/
"""

from bam.actuator import DCMotorActuator
from bam.model import Model
import numpy as np

# --- JGA25-371 datasheet, 12V nominal ---
VIN = 12.0  # V, nominal supply
NO_LOAD_RPM_OUTPUT = 463.0  # free-run speed at the wheel (output) shaft
NO_LOAD_CURRENT = 0.046  # A
STALL_CURRENT = 1.0  # A
STALL_TORQUE_OUTPUT_NM = 1.1 * 0.0980665  # 1.1 kgf*cm -> N*m, at the wheel (output) shaft
GEAR_RATIO = 9.28  # for reference; not used below, see note

# kt/R are derived and used entirely at the *output* (wheel) shaft, treating motor+gearbox
# as one black box rather than modeling the gearbox's torque/speed conversion explicitly.
# This sidesteps having to also convert BAM's friction terms across the gearbox (Coulomb
# friction scales by the gear ratio referred output-to-motor, viscous/damping terms scale
# by its *square* — extra complexity not worth it for numbers this rough to begin with) and
# matches how the datasheet itself specifies stall torque/current: at the shaft you can
# actually measure from outside, i.e. the wheel.
KT = STALL_TORQUE_OUTPUT_NM / STALL_CURRENT  # N*m/A, output-shaft-referred
R = VIN / STALL_CURRENT  # ohm, derived from the stall condition (back-EMF = 0 there)

# Caveat, worth knowing before trusting this model too far: plugging KT/R back into the
# free-run speed equation (zero torque, so omega = VIN/KT) predicts ~1062 RPM at the output
# shaft, but the datasheet states 463 RPM directly, a factor of 2.3. Cheap gearmotor
# datasheets are commonly rounded/approximate rather than
# bench-measured, so a ~2x mismatch between two independently-quoted numbers is plausible,
# not obviously a bug here — but it means this model should not be trusted at the level of
# "matches the datasheet's free-run speed," only "roughly the right order of magnitude."
# Closing that gap for real is exactly what BAM's identification pipeline (fitting from
# recorded trajectories) exists for.

# Simple internal velocity-tracking control law standing in for the rover's real motor
# driver loop (voltage proportional to speed error, saturated at the supply rail). This is
# what turns our action space (target wheel velocity) into a voltage command; BAM's own
# VoltageControlledActuator instead assumes a *position* target, which doesn't fit a wheel.
VELOCITY_KP = 6.0  # V per (rad/s) of speed error


def make_friction_model():
    """BAM Stribeck friction model for one wheel.

    BAM's own default Parameter values (~0.05-0.1 Nm) are calibrated for the much larger
    reference actuators it ships fitted models for (Dynamixel MX-series etc.) — applied
    as-is, they'd exceed this motor's entire ~0.108 Nm stall torque and the wheel could
    never move at all. Scaled down here to a small fraction of our own computed stall
    torque so the numbers are at least dimensionally sane; this is still a guess, not a
    real fit (that needs recorded trajectories from the actual motor, per BAM's
    identification pipeline), so treat it as a starting point to replace once that data
    exists, not a calibrated model."""
    model = Model(stribeck=True)
    model.set_actuator(DCMotorActuator(testbench_class=None, vin=VIN, kp=0.0))
    model.kt.value = KT
    model.R.value = R
    model.friction_base.value = 0.1 * STALL_TORQUE_OUTPUT_NM
    model.friction_stribeck.value = 0.1 * STALL_TORQUE_OUTPUT_NM
    model.friction_viscous.value = 0.05 * STALL_TORQUE_OUTPUT_NM
    return model


def motor_torque(target_omega, measured_omega):
    """DC motor torque [Nm, at the wheel] for a commanded wheel speed, using the same
    tau = kt*V/R - kt^2*omega/R equation BAM's VoltageControlledActuator uses internally,
    with our own velocity (not position) control law feeding it."""
    error = np.asarray(target_omega) - np.asarray(measured_omega)
    voltage = np.clip(VELOCITY_KP * error, -VIN, VIN)
    torque = KT * voltage / R - (KT**2) * np.asarray(measured_omega) / R
    return torque


def apply_friction(friction_model, model, dof_adr, dtheta):
    """Write this step's Stribeck frictionloss/damping onto the given dof (see
    bam.model.Model.compute_frictions — motor_torque/external_torque are unused for a
    non-load-dependent model, so 0.0 is a legitimate simplification, not a placeholder)."""
    frictionloss, damping = friction_model.compute_frictions(0.0, 0.0, dtheta)
    model.dof_frictionloss[dof_adr] = frictionloss
    model.dof_damping[dof_adr] = damping


# --- The real rover's actuator envelope, from rover_control/rover.py -----------------------
# MAX_VELOCITY 0.25 m/s and MIN_VELOCITY 0.075 m/s at this wheel radius. The dead zone is not
# a safety margin: below it the wheels cannot overcome friction and simply do not turn, so a
# command inside it produces no motion at all. Any env whose policy is meant to run on the
# physical rover should train inside this envelope rather than let rl_rover.py clamp at
# inference -- its own docstring warns that causes "jerky behavior right at the clamp
# boundaries".
WHEEL_RADIUS = 0.0335
MAX_WHEEL_SPEED = 0.25 / WHEEL_RADIUS  # ~7.46 rad/s
MIN_WHEEL_SPEED = 0.075 / WHEEL_RADIUS  # ~2.24 rad/s


def apply_actuator_envelope(omega, max_omega=None, min_omega=None):
    """Clip to the motors' top speed and zero anything inside their dead zone.

    Takes and returns wheel speeds in rad/s. Pass explicit bounds to apply per-episode
    domain randomization over unit-to-unit motor variation."""
    import numpy as np

    top = MAX_WHEEL_SPEED if max_omega is None else max_omega
    floor = MIN_WHEEL_SPEED if min_omega is None else min_omega
    omega = np.clip(omega, -top, top)
    return np.where(np.abs(omega) < floor, 0.0, omega)
