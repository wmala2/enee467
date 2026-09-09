"""Cross-check envs/motor.py against BAM's own implementation and against MuJoCo.

The rover's motor model is the one number in this project taken largely on faith: its
parameters come from a cheap gearmotor datasheet rather than a bench, and `envs/motor.py`
says so. Nothing here can fix that, and none of these tests claim the model matches the real
JGA25-371. What they do check is the two things that are checkable without hardware.

First, that our hand-written torque law is the equation we claim it is. `motor_torque` says
in its docstring that it uses the same relation as BAM's voltage-controlled actuator; that is
a claim about someone else's code, so it should be tested against that code rather than
believed.

Second, that MuJoCo actually integrates the model we hand it. The torque law and the friction
model can both be right while the coupling into `dof_frictionloss` / `dof_damping` is wrong,
and the symptom would be a policy trained against dynamics that no equation in this repo
describes. This mirrors the three-way validation in the Microduck project (real bench against
MuJoCo+BAM against BAM's own simulator); the bench leg needs recorded trajectories we do not
have, so these cover the other two.
"""

import math

from bam.actuator import VoltageControlledActuator
from bam.model import Model
from envs import motor
import mujoco
import numpy as np
import pytest


def _bam_reference_actuator():
    """BAM's own voltage-controlled actuator, carrying our motor's kt and R."""
    model = Model(stribeck=True)
    actuator = VoltageControlledActuator(testbench_class=None, vin=motor.VIN, kp=0.0)
    model.set_actuator(actuator)
    model.kt.value = motor.KT
    model.R.value = motor.R
    return actuator


def test_torque_law_matches_bam_reference():
    """Our torque equals BAM's for the same voltage and speed."""
    actuator = _bam_reference_actuator()
    for speed in (-8.0, -2.24, 0.0, 2.24, 7.46, 12.0):
        for target in (-10.0, -2.0, 0.0, 2.0, 7.46, 20.0):
            ours = float(motor.motor_torque(target, speed))
            volts = float(np.clip(motor.VELOCITY_KP * (target - speed), -motor.VIN, motor.VIN))
            theirs = float(actuator.compute_torque(volts, True, 0.0, speed))
            assert ours == pytest.approx(theirs, rel=1e-12, abs=1e-15), (
                f"target={target} speed={speed}: ours {ours} vs BAM {theirs}"
            )


def test_back_emf_opposes_motion_and_saturates_at_the_rail():
    """Two properties the equation must have, independent of the parameter values."""
    # Commanding a speed already exceeded produces braking torque, not drive.
    assert float(motor.motor_torque(2.0, 8.0)) < 0
    # Beyond the point where kp * error exceeds the supply, torque stops growing with error.
    far = float(motor.motor_torque(1e3, 0.0))
    farther = float(motor.motor_torque(1e6, 0.0))
    assert far == pytest.approx(farther, rel=1e-12)
    assert far == pytest.approx(motor.KT * motor.VIN / motor.R, rel=1e-12)


def _settle(timestep, target=5.0, seconds=8.0):
    """Spin one free wheel under the real control path and return its settled speed."""
    xml = f"""
    <mujoco>
      <option timestep="{timestep}"/>
      <worldbody>
        <body name="wheel">
          <joint name="axle" type="hinge" axis="0 0 1"/>
          <geom type="cylinder" size="0.0335 0.01" mass="0.05"/>
        </body>
      </worldbody>
      <actuator><motor name="drive" joint="axle" gear="1"/></actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "axle")]
    friction = motor.make_friction_model()
    for _ in range(int(seconds / timestep)):
        speed = float(data.qvel[dof])
        motor.apply_friction(friction, model, dof, speed)
        data.ctrl[0] = float(motor.motor_torque(target, speed))
        mujoco.mj_step(model, data)
    return float(data.qvel[dof])


def _predicted_speed(target=5.0):
    """The speed the model itself implies, with no reference to MuJoCo.

    At steady state the wheel stops accelerating, so drive torque equals the friction the
    model asks MuJoCo to apply. Solving that scalar balance by bisection gives the answer the
    simulator ought to reproduce.
    """
    friction = motor.make_friction_model()

    def imbalance(speed):
        frictionloss, damping = friction.compute_frictions(0.0, 0.0, speed)
        drive = float(motor.motor_torque(target, speed))
        return drive - (float(frictionloss) * np.sign(speed) + float(damping) * speed)

    low, high = 1e-9, target
    assert imbalance(low) > 0 > imbalance(high), "no sign change to bracket; the model changed"
    for _ in range(200):
        mid = 0.5 * (low + high)
        low, high = (mid, high) if imbalance(mid) > 0 else (low, mid)
    return 0.5 * (low + high)


def test_mujoco_reproduces_the_model_when_the_timestep_resolves_it():
    """With the dynamics resolved, stepping MuJoCo lands exactly on the analytic fixed point.

    This is the part that proves the coupling into dof_frictionloss and dof_damping is right.
    The torque law and the friction model can each be correct while the wiring between them
    and the simulator is not, and the only way to separate those is to remove integration
    error from the comparison. At 0.2 ms the two agree to well under a percent.
    """
    assert _settle(0.0002) == pytest.approx(_predicted_speed(), rel=0.005)


def test_production_timestep_bias_stays_within_its_documented_band():
    """At the 2 ms timestep the scenes actually use, the wheel settles a few percent slow.

    Our control law is evaluated in Python from the previous step's velocity, so it is
    integrated explicitly no matter which integrator MuJoCo is configured with; implicitfast
    cannot see it. With this motor's electrical time constant that leaves a small systematic
    bias rather than instability, because the friction terms damp it. The bias is real and
    one-sided, so it is pinned here: the simulated rover runs slightly slower than its own
    motor model says it should, and if that gap grows the sim2real story changes.
    """
    settled = _settle(0.002)
    predicted = _predicted_speed()
    bias = (settled - predicted) / predicted
    assert -0.08 < bias < 0.0, f"settled {settled:.4f} vs predicted {predicted:.4f} ({bias:+.2%})"


def test_loaded_wheel_in_the_real_scene_is_stable():
    """The production env must not oscillate, which is what the bias above could become.

    A free wheel carrying only its own 2e-5 kg m^2 is close enough to the explicit stability
    limit at 2 ms to matter; in the real scene ground contact couples the chassis mass and the
    friction terms damp the loop, and the wheel settles flat. This test is the guard on that:
    if a future change lightens the wheel, shortens the contact, or raises VELOCITY_KP, the
    speed trace starts alternating and this catches it before a policy trains against it.
    """
    import envs as _envs  # noqa: F401  (registers the environments)
    import gymnasium as gym

    environment = gym.make(
        "LineFollowerReal-v0", track="rover_line_oval_real.xml", domain_randomize=False
    )
    unwrapped = environment.unwrapped
    try:
        environment.reset(seed=0)
        speeds = []
        for _ in range(60):
            _, _, terminated, truncated, _ = environment.step(np.zeros(2, dtype=np.float32))
            speeds.append(float(unwrapped.data.qvel[unwrapped._wheel_dof_adr][1]))
            if terminated or truncated:
                break
        # Guard against a vacuous pass: a two-sample trace has a tiny std for free.
        assert len(speeds) >= 20, f"episode ended after {len(speeds)} steps; nothing was measured"
        trace = np.asarray(speeds[len(speeds) // 2 :])
        assert trace.std() < 0.05, f"wheel speed is not settling: std {trace.std():.4f}"
        steps = np.diff(trace)
        reversals = int(np.sum(np.sign(steps[:-1]) * np.sign(steps[1:]) < 0))
        assert reversals <= len(steps) // 3, f"speed alternates every step: {reversals} reversals"
    finally:
        environment.close()


def test_free_run_speed_matches_the_stall_derived_parameters():
    """Pin the datasheet inconsistency that motor.py documents, so it cannot drift silently.

    kt and R are both derived from the stall condition, and feeding them back through the
    no-load case (zero torque, so w = V / kt) predicts a free-run speed roughly twice the
    463 RPM the datasheet states. motor.py argues that is datasheet rounding rather than a
    bug. Either way the number bounds how far the model can be trusted, so it is pinned.
    """
    free_run_rad_s = motor.VIN / motor.KT
    free_run_rpm = free_run_rad_s * 60.0 / (2.0 * math.pi)
    assert free_run_rpm == pytest.approx(1062.3, rel=0.01)
    assert free_run_rpm / motor.NO_LOAD_RPM_OUTPUT == pytest.approx(2.29, rel=0.02)
