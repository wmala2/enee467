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
        drive, electrical_damping = motor.motor_drive_and_damping(target, speed)
        motor.apply_friction(friction, model, dof, speed, extra_damping=float(electrical_damping))
        data.ctrl[0] = float(drive)
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


def test_split_form_is_exactly_the_torque_law():
    """drive - damping * speed must reproduce motor_torque, in both regions.

    The split exists only to move the speed-dependent half of the torque into the integrator;
    it must not change the physics. Saturation is the interesting case, since there the
    voltage stops tracking the error and only back-EMF remains speed-dependent.
    """
    for target in (-10.0, 0.0, 2.0, 5.0, 7.46, 20.0):
        for speed in (-8.0, 0.0, 2.24, 5.0, 12.0):
            drive, damping = motor.motor_drive_and_damping(target, speed)
            assert float(drive - damping * speed) == pytest.approx(
                float(motor.motor_torque(target, speed)), abs=1e-15
            ), f"target={target} speed={speed}"


def test_production_timestep_reproduces_the_model():
    """At the 2 ms the scenes use, the settled speed matches the model's own fixed point.

    Applying the whole torque as ctrl left this about 5.6% slow, because the speed-dependent
    part was evaluated from the previous step's velocity and the resulting time constant is
    shorter than the step. Handing that part to dof_damping lets the integrator take it
    implicitly, which removes the bias rather than shrinking it. If this regresses, the
    simulated rover is quietly driving slower than its own motor model says.
    """
    settled = _settle(0.002)
    predicted = _predicted_speed()
    assert settled == pytest.approx(predicted, rel=0.005), (
        f"settled {settled:.4f} vs predicted {predicted:.4f} "
        f"({(settled - predicted) / predicted:+.2%})"
    )


def test_split_form_is_stable_across_the_inertias_the_rover_spans():
    """The implicit split must hold from a bare wheel up to a wheel dragging the chassis.

    A free wheel carries about 2.0e-5 kg m^2 and in contact it also drags the rover, which
    raises the effective inertia by orders of magnitude. The old explicit form settled
    backwards at -1.79 rad/s at the free-wheel figure and only became correct above roughly
    2.8e-5; it looked fine in the loaded scene purely because contact hid the light end. That
    is the case this covers, since a wheel leaving the ground reaches it for real.

    The sweep starts at 1.1e-5 because below that both forms diverge: that is the timestep
    against the electrical time constant, which moving the term into the integrator does not
    claim to fix. Within the covered range every case must land on the model's own fixed
    point and settle rather than alternate.
    """
    predicted = _predicted_speed()
    for mass in (0.02, 0.035, 0.05, 0.5, 5.0):
        xml = f"""
        <mujoco>
          <option timestep="0.002"/>
          <worldbody>
            <body name="wheel">
              <joint name="axle" type="hinge" axis="0 0 1"/>
              <geom type="cylinder" size="0.0335 0.01" mass="{mass}"/>
            </body>
          </worldbody>
          <actuator><motor name="drive" joint="axle" gear="1"/></actuator>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "axle")]
        friction = motor.make_friction_model()
        trace = []
        for _ in range(6000):
            speed = float(data.qvel[dof])
            drive, electrical = motor.motor_drive_and_damping(5.0, speed)
            motor.apply_friction(friction, model, dof, speed, extra_damping=float(electrical))
            data.ctrl[0] = float(drive)
            mujoco.mj_step(model, data)
            trace.append(float(data.qvel[dof]))
        tail = np.asarray(trace[-500:])
        assert tail.mean() == pytest.approx(predicted, rel=0.01), (
            f"mass {mass}: settled {tail.mean():.4f} vs predicted {predicted:.4f}"
        )
        assert tail.std() < 0.01, f"mass {mass}: not settling, std {tail.std():.5f}"


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
