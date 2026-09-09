"""Scores a classical PID controller in LineFollowerReal-v0's own observation/action space.

    uv run python scripts/pid_baseline_real.py                    # every track
    uv run python scripts/pid_baseline_real.py --track rover_line_hairpin_real.xml

This exists to keep the RL numbers honest, and to keep the tracks honest. A learned policy's
score means nothing in isolation -- a reward can be high because the policy is good or because
the metric is soft -- so every track and every reward change gets a known-good controller run
through it first. It has caught real problems: a speed-scaled centring reward that made
tracking worse, and a deviation metric that rated a rover driving dead straight better than a
trained policy.

The direction it matters most here: a held-out track the PID cannot drive either is measuring
the rover's kinematics or the camera's field of view, not the policy, and a failure there is
not evidence against the policy.

Unlike scripts/line_follower.py, which runs its own PID on raw camera frames, this consumes
exactly what the policy consumes (the near-band error) and emits exactly what the policy emits
(normalized linear/angular), so the two are directly comparable.
"""

import argparse

import envs  # noqa: F401  (imported for its side effect: registers LineFollowerReal-v0)
from envs.tasks.line_follower_real_env import EVAL_TRACKS_REAL
from envs.tasks.line_follower_real_env import TRACKS_REAL
from evaluate_real import episode_outcome
from evaluate_real import track_deviation
import gymnasium as gym
import numpy as np
from train_real import ENV_ID

KP, KD = 1.6, 0.4
CRUISE = 0.25  # normalized linear command; -1 is stopped, +1 is MAX_LINEAR_VEL


def pid_episode(env, seed):
    obs, _ = env.reset(seed=seed)
    prev_err = 0.0
    devs = []
    while True:
        near_err, _, near_seen = obs["line"][0][:3]
        # Hold the last command when the line is not visible rather than steering on a stale
        # error -- the same thing the classical follower does when it drives off the end.
        if near_seen > 0.5:
            derivative = near_err - prev_err
            prev_err = near_err
        else:
            derivative = 0.0
        angular = float(np.clip(-(KP * prev_err + KD * derivative), -1.0, 1.0))
        cmd = np.array([CRUISE, angular], dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(cmd)
        devs.append(track_deviation(env, env.unwrapped.data.qpos[:2].copy()))
        if terminated or truncated:
            break
    out = episode_outcome(env)
    out["mean_dev"] = float(np.mean(devs))
    out["max_dev"] = float(np.max(devs))
    return out


def score(track, episodes):
    env = gym.make(ENV_ID, track=track)
    outcomes = [pid_episode(env, seed) for seed in range(episodes)]
    env.close()
    finished = [o for o in outcomes if o["completed"]]
    return {
        "completion": sum(o["completed"] for o in outcomes) / episodes,
        # Conditioned on completion for the same reason the sweep conditions it: an episode
        # that dies in the first metre reports a flatteringly small deviation.
        "mean_dev_cm": float(np.mean([o["mean_dev"] for o in finished]) * 100) if finished else 0.0,
        "worst_dev_cm": float(np.max([o["max_dev"] for o in finished]) * 100) if finished else 0.0,
        "progress": float(np.mean([o["progress"] for o in outcomes])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", help="score just this track")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    tracks = [args.track] if args.track else list(TRACKS_REAL) + list(EVAL_TRACKS_REAL)
    for track in tracks:
        held_out = track in EVAL_TRACKS_REAL
        s = score(track, args.episodes)
        print(
            f"{track:32s} {'held-out' if held_out else 'training':9s} "
            f"completion {s['completion']:6.1%}  progress {s['progress']:6.1%}  "
            f"mean {s['mean_dev_cm']:5.1f} cm  worst {s['worst_dev_cm']:6.1f} cm"
        )


if __name__ == "__main__":
    main()
