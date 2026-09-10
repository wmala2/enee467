"""DC motor model for the rover's JGA25-371 drive wheels: BAM's friction model
(https://github.com/Rhoban/bam) plus an electrical model derived from the datasheet.

Every parameter here comes from a datasheet rather than a bench, and BAM's friction values
are its un-fit defaults, so treat this as the right shape rather than the right numbers.
tests/test_motor_model.py pins what is actually verified.

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
GEAR_RATIO = 9.28  # reference only; kt/R below are output-shaft referred

# kt/R are referred to the output (wheel) shaft, treating motor+gearbox as one black box.
# That matches how the datasheet quotes stall torque and current, and avoids converting
# BAM's friction terms across the gear ratio.
KT = STALL_TORQUE_OUTPUT_NM / STALL_CURRENT  # N*m/A, output-shaft-referred
R = VIN / STALL_CURRENT  # ohm, derived from the stall condition (back-EMF = 0 there)

# KT/R put the free-run speed (omega = VIN/KT) at ~1062 RPM against the datasheet's stated
# 463, a factor of 2.3. Two independently quoted datasheet numbers disagreeing is plausible
# for a cheap gearmotor, so trust the order of magnitude, not the value.

# Stands in for the rover's motor driver: voltage proportional to speed error, saturated at
# the rail. BAM's own VoltageControlledActuator assumes a position target, which a
# continuously-driven wheel does not have.
VELOCITY_KP = 6.0  # V per (rad/s) of speed error


def make_friction_model():
    """BAM Stribeck friction model for one wheel.

    BAM's defaults (~0.05-0.1 Nm) are fitted to much larger actuators and would exceed this
    motor's ~0.108 Nm stall torque, leaving the wheel unable to move, so they are scaled to a
    fraction of our own stall torque. Dimensionally sane, still a guess, not a fit."""
    model = Model(stribeck=True)
    model.set_actuator(DCMotorActuator(testbench_class=None, vin=VIN, kp=0.0))
    model.kt.value = KT
    model.R.value = R
    model.friction_base.value = 0.1 * STALL_TORQUE_OUTPUT_NM
    model.friction_stribeck.value = 0.1 * STALL_TORQUE_OUTPUT_NM
    model.friction_viscous.value = 0.05 * STALL_TORQUE_OUTPUT_NM
    return model


def motor_torque(target_omega, measured_omega):
    """DC motor torque [Nm, at the wheel] for a commanded wheel speed.

    Uses tau = kt*V/R - kt^2*omega/R, the same relation as BAM's VoltageControlledActuator,
    fed by our velocity control law instead of its position one."""
    error = np.asarray(target_omega) - np.asarray(measured_omega)
    voltage = np.clip(VELOCITY_KP * error, -VIN, VIN)
    torque = KT * voltage / R - (KT**2) * np.asarray(measured_omega) / R
    return torque


def motor_drive_and_damping(
    target_omega, measured_omega, *, vin=VIN, kt=KT, resistance=R, kp=VELOCITY_KP
):
    """motor_torque split into a constant drive and a damping coefficient, so the
    speed-dependent half can go to `dof_damping` where the integrator solves it implicitly.

    Applied explicitly instead, this loop settles backwards at a free wheel's 2.0e-5 kg m^2;
    ground contact hides that by dragging the chassis, so it surfaces only when a wheel
    leaves the ground. Returns (drive [Nm], damping [Nm/(rad/s)])."""
    target = np.asarray(target_omega, dtype=float)
    measured = np.asarray(measured_omega, dtype=float)
    # Episode parameters may be independent per wheel while retaining the original defaults.
    raw_voltage = kp * (target - measured)
    saturated = np.abs(raw_voltage) >= vin
    drive = np.where(
        saturated, kt * np.sign(raw_voltage) * vin / resistance, kt * kp * target / resistance
    )
    damping = np.where(saturated, kt**2 / resistance, kt * kp / resistance + kt**2 / resistance)
    return drive, damping


def apply_friction(friction_model, model, dof_adr, dtheta, extra_damping=0.0):
    """Write this step's Stribeck frictionloss and damping onto the given dof.

    `extra_damping` carries the motor's speed-dependent torque from motor_drive_and_damping.
    Both land in the same `dof_damping` slot, so they must be summed; writing either alone
    discards the other."""
    frictionloss, damping = friction_model.compute_frictions(0.0, 0.0, dtheta)
    model.dof_frictionloss[dof_adr] = frictionloss
    model.dof_damping[dof_adr] = damping + extra_damping


# The real rover's envelope, from rover_control/rover.py. MIN is a dead zone, not a safety
# margin: below it the wheels cannot overcome friction at all, so policies meant for hardware
# should train inside this range rather than be clamped at inference.
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
