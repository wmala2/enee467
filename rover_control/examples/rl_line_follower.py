"""Drive the physical rover with the PPO line-following policy trained in rover_mujoco
(pulled from https://huggingface.co/CursedRock17/rover-line-follower-ppo the first time this
runs, then cached). Deployed as-is -- see rover_control/rl_rover.py's docstring for the known
sim<->real translation points (sign convention, units, real velocity clamping).

Never validated on real hardware before this. Watch it closely on the first run and be ready
to Ctrl-C -- every Rover motion method (including this one, via the inherited update() loop)
stops the rover on Ctrl-C or any error.
"""
from rover_control.rl_rover import RLLineFollowerRover


def main():
    rover = RLLineFollowerRover(show_camera=True)
    try:
        print("Running the RL line-follower. Ctrl-C to stop.")
        rover.update()
    except KeyboardInterrupt:
        pass
    finally:
        rover.stop()
        rover.close()


if __name__ == "__main__":
    main()
