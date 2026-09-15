"""The whole program a student writes to drive the rover through a maze of ArUco tags.

Every control loop, PID, and camera detail lives inside the `Rover` class already -- this script
just calls its high-level methods, the same way you'd call any other library.
"""

from rover_control.rover import Rover

if __name__ == "__main__":
    # Connect to the rover (set show_camera=True to watch what the camera sees)
    rover = Rover(show_camera=True)

    try:
        # Drive to ArUco tag 0, then tag 1, stopping a quarter metre from each
        rover.run_maze([0, 1])

        # The same behaviors are available one at a time, too:
        #   rover.drive_to_tag(2)     # search, center, and drive to tag 2
        #   rover.forward(0.5)        # drive forward half a metre
        #   rover.turn(-90)           # spin 90 degrees to the left
        #   pose = rover.see_tag(0)   # one reading of tag 0, or None if not seen
    finally:
        # Always stop the rover and shut the camera down when the program ends
        rover.stop()
        rover.close()
