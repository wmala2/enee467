"""Trains a PPO policy on GoalNav-v0: drive to a commanded (x, y) offset inside a walled 5x5 m
arena while avoiding a random number and mix of obstacles. See docs/rl-goal-nav.md.

Same sim-to-real footing as train_real.py's line follower -- noisy encoders, the rover's
three-beam lidar, wheel-velocity actions, the same domain randomization ranges -- with the
goal handed to the policy as a dead-reckoned vector that drifts rather than as ground truth.

Two choices here are worth knowing about before changing them:

* `include_image=True, include_lidar=False`. Obstacle avoidance is camera-only by design, so
  the policy gets the same monocular view a deployed rover would and no range at all. The
  classical baseline (scripts/goal_nav_baseline.py) reaches 14.5% arrivals without avoidance
  and only 4.5% with it, because avoidance manoeuvres lengthen episodes and odometry drift
  grows with them -- beating that while keeping collisions down is the actual objective.
* `curriculum=True`. Success is much sparser than line following, so each env starts on short
  goals in an empty arena and promotes itself once it arrives reliably. The final level is the
  full task: 3-8 obstacles, goals up to 3.5 m. See CURRICULUM in envs/tasks/goal_nav_env.py.

Dict observations (encoders + lidar + goal) need "MultiInputPolicy", not "MlpPolicy".
"""

import os

import envs  # noqa: F401  (imported for its side effect: registers GoalNav-v0)
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

ENV_ID = "GoalNav-v0"
TOTAL_TIMESTEPS = 2_000_000
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../runs/ppo_goal_nav")

# Without the camera render the rollout is physics-bound and cheap, so this scales further
# than train_real.py's 8 (which was rendering-bound). Measured on this machine before picking.
N_ENVS = 16

# ent_coef: the line-follower sweep found the SB3 default of 0.0 plateaued from
# under-exploration while 0.01 was still climbing at 3M steps (see sweep_hparams.py). This
# task needs more exploration, not less -- a policy that never turns never finds a goal behind
# it -- so start from that sweep's winner rather than from the default.
ENT_COEF = 0.01

# n_steps is the one hyperparameter here that is *not* an SB3 default, and it matters more
# than it looks: the default 2048, times N_ENVS, is a 49k-step rollout, which over this budget
# is only ~160 gradient updates for the whole run. 512 gives ~650 instead, at the cost of a
# shorter advantage horizon -- worth it when the reward is dense (progress every step).
N_STEPS = 512
BATCH_SIZE = 512
# Episodes run up to MAX_EPISODE_STEPS=400, and the SUCCESS_BONUS lands only at the very end,
# so the default gamma=0.99 (~100-step horizon) discounts arrival too heavily to steer early
# decisions. 0.995 roughly doubles that horizon.
GAMMA = 0.995

# Camera on, lidar off: obstacle avoidance here is specified as camera-only, so the policy
# sees what the rover sees and gets no range information. That costs roughly an order of
# magnitude in throughput versus the vector-only configuration -- rendering dominates the step.
ENV_KWARGS = {"include_image": True, "include_lidar": False, "curriculum": True}


class ProgressCallback(BaseCallback):
    """Logs the things that actually say whether this task is being learned -- how often the
    rover arrives, how often it crashes, and how far the curriculum has advanced -- none of
    which are visible in ep_rew_mean alone (a policy that stops just short of every goal and
    one that arrives look similar in reward, and very different here)."""

    def _on_step(self):
        # Only count *finished* episodes. GoalNavEnv puts "arrived"/"collided" in the info dict
        # on every step, so averaging over all infos silently divides the rate by the episode
        # length -- which made a ~100% arrival rate read as 0.006 in the first run of this.
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        finished = [info for info, done in zip(infos, dones, strict=False) if done]
        if finished:
            self.logger.record_mean(
                "task/arrival_rate", float(np.mean([i["arrived"] for i in finished]))
            )
            self.logger.record_mean(
                "task/collision_rate", float(np.mean([i["collided"] for i in finished]))
            )
        levels = [i["curriculum_level"] for i in infos if "curriculum_level" in i]
        if levels:
            self.logger.record_mean("task/curriculum_level", float(np.mean(levels)))
        return True


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = make_vec_env(
        ENV_ID,
        n_envs=N_ENVS,
        env_kwargs=ENV_KWARGS,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"},
    )
    model = PPO(
        "MultiInputPolicy",
        env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        verbose=1,
        tensorboard_log=MODEL_DIR,
    )
    print(f"Training on device: {model.device}")

    callbacks = [
        ProgressCallback(),
        # Checkpoints so a long run is inspectable (and resumable) before it finishes.
        CheckpointCallback(
            save_freq=max(1, 250_000 // N_ENVS), save_path=MODEL_DIR, name_prefix="checkpoint"
        ),
    ]
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

    model.save(os.path.join(MODEL_DIR, "model"))
    print(f"Saved trained model to {MODEL_DIR}/model.zip")
    print(f"View training curves with: uv run tensorboard --logdir {MODEL_DIR}")


if __name__ == "__main__":
    main()
