"""Prints the rover's raw wheel encoder counts, live, with no driving involved.

This is the simplest possible sanity check that the rover's encoders are wired up and reporting
over the network: no camera, no control loop, just the numbers each wheel is reporting and how
much they've changed since the last print.
"""

# External Libraries
import time

# Local Files to Import
from rover_control.encoder_poller import EncoderPoller

# How often to print the encoder counts
POLL_HZ = 5.0


def main():
    print("Polling encoder counts - press Ctrl-C to quit\n")
    # EncoderPoller talks to the rover on its own background thread, so this loop can just ask
    # for whatever the latest reading is instead of blocking on the network itself.
    poller = EncoderPoller(poll_hz=POLL_HZ).start()
    try:
        while True:
            data = poller.latest()
            if data is None:
                print("(waiting for first reply - is the rover on the network?)")
            else:
                # l/r are the raw running totals; dl/dr are how much each changed since the
                # last print, which is what actually tells you a wheel is turning.
                l, r, dl, dr = data
                dl_str = f"{dl:+d}" if dl is not None else "  ---"
                dr_str = f"{dr:+d}" if dr is not None else "  ---"
                print(f"left={l:8d}  right={r:8d}    Δleft={dl_str:>6}  Δright={dr_str:>6}")
            time.sleep(1.0 / POLL_HZ)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        poller.stop()


if __name__ == "__main__":
    main()
