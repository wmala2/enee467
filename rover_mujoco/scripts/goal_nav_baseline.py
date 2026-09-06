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
# Only spin in place when the goal is genuinely behind. At 50 degrees the controller spent 40%
# of its steps pirouetting -- avoidance manoeuvres leave a large bearing error, which triggered
# a spin, which was then undone by the next avoidance. Instrumented over 40 episodes: the only
# episodes that arrived were ones where avoidance never fired at all.
FACE_FIRST_RAD = math.radians(110.0)

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
# Keep real forward speed while arcing around. Crawling makes the manoeuvre long, and every
# extra step of manoeuvring is extra odometry drift -- the mechanism that made avoidance cost
# more arrivals than it saved.
AVOID_CRUISE = 0.55 * CRUISE
# Below this the obstacle is close enough to straight ahead that the image has no useful
# opinion about which side to pass it.
CENTROID_DEAD_AHEAD = 0.2
# Having picked a side, keep turning that way until the near ground is actually clear, then a
# few steps more. A fixed-length blind arc was the earlier design and it fails for a reason
# worth recording: measured reaction distance is 1.11 m and turning out of the way needs only
# 0.09 m, so margin was never the problem -- but arcing blind for a fixed 8 steps through a
# field of 3-8 obstacles simply finds a different one. Clearing the view is the right exit
# condition; the extra steps stop it re-triggering on the obstacle it just passed.
AVOID_CLEAR_STEPS = 6
AVOID_MAX_STEPS = 40  # give up on the manoeuvre rather than circling forever


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
        self._avoid_elapsed = 0

    def __call__(self, obs):
        """Returns a normalized action. The controller reasons in rad/s because that is what
        the motor limits are expressed in; the env's action space is [-1, 1], so divide at the
        boundary rather than scattering the scale through the control law."""
        wheels = control(obs, avoid=self.avoid, state=self)
        return np.clip(wheels / arena.MAX_WHEEL_SPEED, -1.0, 1.0).astype(np.float32)


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

    bearing = math.atan2(left, forward)

    if avoid:
        fraction, centroid = obstacle_bearing(obs["image"])
        blocked = fraction > OBSTACLE_PIXEL_FRACTION
        committed = state is not None and state._commit_steps > 0
        if committed:
            state._avoid_elapsed += 1
            if state._avoid_elapsed > AVOID_MAX_STEPS:
                state._commit_steps = 0  # bail out; circling is its own failure
            elif blocked:
                state._commit_steps = AVOID_CLEAR_STEPS  # still blocked: keep going
            else:
                state._commit_steps -= 1  # clear: run out the trailing steps
            if state._commit_steps > 0:
                turn = state._commit_turn
                return escape_dead_zone([AVOID_CRUISE + turn, -AVOID_CRUISE + turn])
        if blocked:
            if state is not None:
                state._commit_steps = AVOID_CLEAR_STEPS
                state._avoid_elapsed = 0
                # Which way to go round. The pixels decide it whenever they have an opinion:
                # turn away from where the obstacle mass actually is. Only when it is squarely
                # dead ahead, and the image gives no preferred side, does the goal break the
                # tie -- going round the side the goal is on saves a spin later.
                #
                # Getting this backwards is expensive and not obvious from the code: an earlier
                # version preferred the goal side whenever the bearing exceeded 20 degrees,
                # which steers straight into any obstacle sitting between the rover and the
                # goal. Collisions went from 15% to 85%, worse than no avoidance at all.
                if abs(centroid) >= CENTROID_DEAD_AHEAD:
                    side = -1.0 if centroid > 0 else 1.0
                else:
                    side = 1.0 if bearing > 0 else -1.0
                state._commit_turn = side * AVOID_TURN
            # Steer away from whichever side the obstacle mass sits on. Equal-sign commands
            # spin this rover in place (rover.xml mirrors the right wheel).
            # Obstacle mass to the right (centroid > 0) means turn left, i.e. positive.
            # Arc around it rather than spinning on the spot: a stationary pirouette makes no
            # progress, and the env's stuck detector is right to treat that as failure.
            turn = AVOID_TURN if centroid > 0 else -AVOID_TURN
            crawl = 0.45 * CRUISE
            return np.array([crawl + turn, -crawl + turn], dtype=np.float32)

    if abs(bearing) > FACE_FIRST_RAD:
        turn = ALIGN_GAIN * bearing
        return escape_dead_zone([turn, turn])

    # Differential drive: equal and opposite is straight ahead, the offset steers.
    # Drive and steer at once, easing off the throttle as the bearing error grows rather than
    # stopping to turn. cos() keeps forward speed high when nearly aligned and low when not.
    speed = CRUISE * max(math.cos(bearing), 0.25)
    turn = TURN_GAIN * bearing
    return escape_dead_zone([speed + turn, -speed + turn])


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
        controller._avoid_elapsed = 0
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
