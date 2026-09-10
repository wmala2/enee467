"""Run a qualified local PPO checkpoint with measured encoder calibration."""

import argparse

from rover_control.rl_rover import RLLineFollowerRover
from rover_control.rover import Rover


def main():
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
