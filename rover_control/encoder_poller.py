import json
import socket
import threading
import time

from rover_control import network_interface


class EncoderPoller:
    """
    Polls the rover's wheel encoders AND lidar in a background thread at a fixed rate.

    Alternates between {"command": "e"} (encoder) and {"command": "l"} (lidar) each tick,
    so each sensor updates at half the poll rate. Owns UDP_REPLY_PORT (9001) for its lifetime
    — do not mix with other code that tries to bind that port in the same process.

    Call start() to begin, latest() / lidar_latest() to read non-blocking, stop() to shut down.
    """

    def __init__(self, poll_hz=10.0):
        # Match the rover's command rate so every drive tick has a fresh encoder reading
        self._interval = 1.0 / poll_hz
        self._enc_query = json.dumps({"command": "e"}).encode("utf-8")
        self._lidar_query = json.dumps({"command": "l"}).encode("utf-8")

        # Bind the reply socket once; timeout slightly under the poll interval so we never stall
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("", network_interface.UDP_REPLY_PORT))
        self._sock.settimeout(self._interval * 0.8)

        self._lock = threading.Lock()

        # Encoder state
        self._left = self._right = None
        self._prev_left = self._prev_right = None
        self._left_delta = self._right_delta = None

        # Lidar state (distances in mm, center_ok = sensor valid flag)
        self._lidar_left = self._lidar_center = self._lidar_right = None
        self._lidar_center_ok = None

        self._running = False
        self._thread = None

    def start(self):
        # Launch the background polling thread; returns self so you can chain: EncoderPoller().start()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="encoder_poller")
        self._thread.start()
        return self

    def stop(self):
        # Signal the thread to exit, wait for it, then close the socket
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._sock.close()

    def _run(self):
        # Alternate between encoder and lidar queries each tick so both stay fresh
        query_enc = True
        while self._running:
            t0 = time.perf_counter()

            network_interface.send_message(self._enc_query if query_enc else self._lidar_query)

            try:
                data, _ = self._sock.recvfrom(512)
                parsed = json.loads(data.decode("utf-8"))

                if "left_encoder" in parsed:
                    # Encoder reply: update counts and compute tick-to-tick deltas
                    left = parsed["left_encoder"]
                    right = parsed["right_encoder"]
                    with self._lock:
                        if self._prev_left is not None:
                            self._left_delta = left - self._prev_left
                            self._right_delta = right - self._prev_right
                        self._left = left
                        self._right = right
                        self._prev_left = left
                        self._prev_right = right

                elif "center_distance" in parsed:
                    # Lidar reply: store distances and center validity flag
                    with self._lock:
                        self._lidar_left = parsed.get("left_distance")
                        self._lidar_center = parsed.get("center_distance")
                        self._lidar_right = parsed.get("right_distance")
                        self._lidar_center_ok = bool(parsed.get("center_ok", False))

            except (TimeoutError, ValueError, KeyError):
                pass  # missed packet or bad JSON - try again next tick

            query_enc = not query_enc

            # Sleep only the time remaining in this interval so we don't drift
            elapsed = time.perf_counter() - t0
            remaining = self._interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def latest(self):
        """
        Non-blocking encoder read.
        Returns (left_count, right_count, left_delta, right_delta) or None until first reply.
        Deltas are counts since the previous encoder poll; None on the very first reading.
        """
        with self._lock:
            if self._left is None:
                return None
            return self._left, self._right, self._left_delta, self._right_delta

    def lidar_latest(self):
        """
        Non-blocking lidar read.
        Returns (left_mm, center_mm, right_mm, center_ok) or None until first reply.
        """
        with self._lock:
            if self._lidar_center is None:
                return None
            return self._lidar_left, self._lidar_center, self._lidar_right, self._lidar_center_ok
