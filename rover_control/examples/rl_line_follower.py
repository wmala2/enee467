"""Runs a PPO policy trained in rover_mujoco on the real, physical rover.

The policy has only ever seen simulated camera frames and simulated encoder counts, so this
script's job is to feed it the same shape of observations from the real hardware, and to convert
real encoder counts into the units the policy expects using this specific rover's measured
calibration (counts per wheel revolution, which direction counts up, and the actual wheel size).
"""

import argparse

from rover_control import network_interface
from rover_control.rl_rover import RLLineFollowerRover
from rover_control.rover import Rover


def main():
    # Every flag below has a sensible default already baked into the firmware/rover classes;
    # you only need to override one if your rover was built or calibrated differently.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("manifest")
    # Default to the firmware's own calibration; override for a differently built rover.
    parser.add_argument(
        "--counts-per-revolution", type=float, default=Rover.ENCODER_COUNTS_PER_REVOLUTION
    )
    parser.add_argument(
        "--encoder-signs", type=int, nargs=2, choices=(-1, 1), default=list(Rover.ENCODER_SIGNS)
    )
    parser.add_argument("--wheel-radius-m", type=float, default=0.03435)
    parser.add_argument("--camera-addr", default=RLLineFollowerRover.DEFAULT_CAMERA_ADDR)
    parser.add_argument(
        "--record",
        help="directory for camera.mp4 and observations.csv showing what the detector sees",
    )
    parser.add_argument(
        "--rover-addr",
        default=network_interface.UDP_IP,
        help="rover UDP address; defaults to $ROVER_IP or the built-in",
    )
    args = parser.parse_args()
    if args.wheel_radius_m <= 0:
        parser.error("wheel radius must be positive")
    # Construction verifies simulation evidence before any hardware connection starts.
    rover = RLLineFollowerRover(
        args.model,
        args.manifest,
        counts_per_revolution=args.counts_per_revolution,
        encoder_signs=args.encoder_signs,
        wheel_diameter_m=2 * args.wheel_radius_m,
        camera_addr=args.camera_addr,
        rover_addr=args.rover_addr,
        record_dir=args.record,
        show_camera=True,
    )
    try:
        print("Running the qualified PPO line follower; Ctrl-C stops the rover.")
        rover.update()
    except KeyboardInterrupt:
        pass
    finally:
        rover.close()


if __name__ == "__main__":
    main()
