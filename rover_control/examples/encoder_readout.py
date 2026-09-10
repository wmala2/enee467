# External Libraries
import time

# Local Files to Import
from rover_control.encoder_poller import EncoderPoller

# How often to print the encoder counts
POLL_HZ = 5.0


def main():
    print("Polling encoder counts - press Ctrl-C to quit\n")
    poller = EncoderPoller(poll_hz=POLL_HZ).start()
    try:
        while True:
            data = poller.latest()
            if data is None:
                print("(waiting for first reply - is the rover on the network?)")
            else:
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
