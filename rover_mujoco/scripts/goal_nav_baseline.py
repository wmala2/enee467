"""Classical baseline for GoalNav-v0: pure pursuit to the goal pose, with reactive
camera-based obstacle avoidance. No learning anywhere.

The point is to establish the number a policy has to beat before spending hours training one.
It is given exactly the same information as the RL policy -- the odometry-derived goal vector
and the same 64x64 camera frame -- and is scored on the same episodes with the same metrics.

    uv run python scripts/goal_nav_baseline.py --episodes 200
    uv run python scripts/goal_nav_baseline.py --episodes 200 --no-avoid   # ablation
"""

import argparse
import math

import envs  # noqa: F401  (imported for its side effect: registers GoalNav-v0)
from envs import arena
from envs.tasks.goal_nav_env import GOAL_SCALE
from envs.tasks.goal_nav_env import GOAL_TOLERANCE_M
import gymnasium as gym
import numpy as np

# Pure pursuit gains. Deliberately simple: if a 30-line controller beats PPO, that is the
# finding, and tuning it further would only make the comparison harder to argue with.
CRUISE = 0.85 * arena.MAX_WHEEL_SPEED  # rad/s, a little under the ceiling for steering headroom
TURN_GAIN = 3.0  # rad/s of differential per radian of bearing error
ALIGN_GAIN = 2.5  # same, for the final in-place heading alignment
FACE_FIRST_RAD = math.radians(50.0)  # beyond this bearing error, turn in place before driving

# Aim *inside* the tolerance. The controller only knows where odometry says the goal is, and
# that estimate drifts ~0.10 m even with domain randomization off, so stopping the instant
# odometry reads "arrived" leaves the rover short of the true goal -- measured 0.40 m mean
# final distance against a 0.25 m tolerance, i.e. 13% arrivals on the easiest setting. Driving
# to half the tolerance absorbs the drift. This is the same correction the real rover needs.
STOP_FRACTION = 0.5

# Nothing below the motors' dead zone produces motion, so a command in that band is the same
# as commanding zero -- which then reads as "stuck". Anything nonzero gets pushed out to the
# floor instead.
DEAD_ZONE = arena.MIN_WHEEL_SPEED


def escape_dead_zone(wheels):
    """Push any nonzero command up to the motors' minimum, preserving sign."""
    out = np.array(wheels, dtype=np.float32)
    small = (np.abs(out) > 1e-6) & (np.abs(out) < DEAD_ZONE)
    out[small] = np.sign(out[small]) * DEAD_ZONE
    return out


# Reactive avoidance. The camera is monocular so there is no range: all we can do is notice
# that a lot of not-floor is in view and steer away from whichever side it sits on.
#
# Only the bottom band of the frame is usable. At the 90 deg forward mount, with the arena
# unwalled, most of the image is horizon/background -- an *empty* arena flags 43% of the full
# frame as "not floor", so a whole-frame test fires on every step and the rover just spins.
# The ground occupies the lower part of the view, which is also the only part where something
# being there means it is in the way.
# Bottom quarter only. A wider band also picks up obstacles far across the arena, which are
# not in the way -- and then the test flickers on and off around its threshold while the goal
# seeking pulls the other way, so the rover oscillates on the spot and never goes anywhere
# (observed: true distance to goal pinned at 2.37 m for 160 steps). The bottom quarter is the
# near ground, where something being visible actually means it is about to be hit.
GROUND_BAND = 0.25
# Measured over 25 seeds at the 60 deg mount: an empty arena flags 0.000 of this band, an
# obstacle 0.4 m ahead flags 0.323. Anywhere between separates them; 0.08 leaves headroom for
# the camera domain randomization to darken or blur the frame without losing the obstacle.
OBSTACLE_PIXEL_FRACTION = 0.08
AVOID_TURN = 0.9 * arena.MAX_WHEEL_SPEED
# Once committed to a side, keep turning that way for a few steps. Reactive avoiders oscillate
# without this: the obstacle leaves view, goal-seeking turns back into it, it reappears.
AVOID_COMMIT_STEPS = 8


def obstacle_bearing(image):
    """(fraction_of_frame, horizontal_centroid_in_[-1,1]) for pixels that are not floor.

    The floor is the modal brightness in a mostly-empty frame, so anything far from the frame
    median is a candidate. Same centroid trick scripts/line_follower.py uses on the line, which
    is the obvious classical move with this camera."""
    gray = image[:, :, 0].astype(np.float32)
    band = gray[int(gray.shape[0] * (1.0 - GROUND_BAND)) :, :]
    mask = np.abs(band - np.median(band)) > 25.0
    fraction = float(mask.mean())
    if fraction < 1e-6:
        return 0.0, 0.0
    cols = np.where(mask.any(axis=0))[0]
    centroid = cols.mean()
    return fraction, float((centroid - band.shape[1] / 2) / (band.shape[1] / 2))


class Controller:
    """Pure pursuit plus reactive avoidance. Stateful only to hold the avoidance commitment."""

    def __init__(self, avoid=True):
        self.avoid = avoid
        self._commit_steps = 0
        self._commit_turn = 0.0

    def __call__(self, obs):
        return control(obs, avoid=self.avoid, state=self)


def control(obs, avoid=True, state=None):
    """One control decision: (left, right) wheel velocities in rad/s."""
    forward, left, cos_dh, sin_dh = obs["goal"]
    forward *= GOAL_SCALE
    left *= GOAL_SCALE
    distance = math.hypot(forward, left)

    if distance <= STOP_FRACTION * GOAL_TOLERANCE_M:
        # Arrived in position as far as odometry can tell: rotate in place onto the commanded
        # heading and hold there until the env calls it.
        turn = ALIGN_GAIN * math.atan2(sin_dh, cos_dh)
        return escape_dead_zone([turn, turn])

    if avoid:
        fraction, centroid = obstacle_bearing(obs["image"])
        committed = state is not None and state._commit_steps > 0
        if committed:
            state._commit_steps -= 1
            crawl = 0.45 * CRUISE
            turn = state._commit_turn
            return np.array([crawl + turn, -crawl + turn], dtype=np.float32)
        if fraction > OBSTACLE_PIXEL_FRACTION:
            if state is not None:
                state._commit_steps = AVOID_COMMIT_STEPS
                state._commit_turn = AVOID_TURN if centroid > 0 else -AVOID_TURN
            # Steer away from whichever side the obstacle mass sits on. Equal-sign commands
            # spin this rover in place (rover.xml mirrors the right wheel).
            # Obstacle mass to the right (centroid > 0) means turn left, i.e. positive.
            # Arc around it rather than spinning on the spot: a stationary pirouette makes no
            # progress, and the env's stuck detector is right to treat that as failure.
            turn = AVOID_TURN if centroid > 0 else -AVOID_TURN
            crawl = 0.45 * CRUISE
            return np.array([crawl + turn, -crawl + turn], dtype=np.float32)

    bearing = math.atan2(left, forward)
    if abs(bearing) > FACE_FIRST_RAD:
        turn = ALIGN_GAIN * bearing
        return escape_dead_zone([turn, turn])

    # Differential drive: equal and opposite is straight ahead, the offset steers.
    turn = TURN_GAIN * bearing
    return escape_dead_zone([CRUISE + turn, -CRUISE + turn])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--no-avoid", action="store_true", help="ablate the camera avoidance")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = gym.make("GoalNav-v0", include_image=True, curriculum=False).unwrapped
    controller = Controller(avoid=not args.no_avoid)
    outcomes = {"arrived": 0, "collided": 0, "out_of_bounds": 0, "stuck": 0, "timeout": 0}
    distances, headings, drifts, lengths, rewards = [], [], [], [], []

    for i in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + i)
        controller._commit_steps = 0
        total = 0.0
        steps = 0
        while True:
            obs, reward, terminated, truncated, info = env.step(controller(obs))
            total += reward
            steps += 1
            if terminated or truncated:
                break
        for key in ("arrived", "collided", "out_of_bounds", "stuck"):
            if info[key]:
                outcomes[key] += 1
                break
        else:
            outcomes["timeout"] += 1
        distances.append(info["distance"])
        headings.append(info["heading_error_deg"])
        drifts.append(info["odometry_error"])
        lengths.append(steps)
        rewards.append(total)

    label = "pure pursuit, no avoidance" if args.no_avoid else "pure pursuit + camera avoidance"
    print(f"\n{label} -- {args.episodes} episodes on the full task")
    for name, count in outcomes.items():
        print(f"  {name:13s} {count:5d}  ({count / args.episodes:6.1%})")
    print(f"  mean final distance : {np.mean(distances):.2f} m")
    print(f"  mean heading error  : {np.mean(headings):.0f} deg")
    print(f"  mean odometry drift : {np.mean(drifts):.2f} m")
    print(f"  mean episode length : {np.mean(lengths):.0f} steps")
    print(f"  mean episode reward : {np.mean(rewards):.1f}")


if __name__ == "__main__":
    main()
