import threading
import time

import cv2
import numpy as np
import requests


class CameraStream:
    """
    Background reader for the ESP32 camera's '/capture' endpoint.

    A worker thread keeps pulling JPEG stills over one reused HTTP connection and always
    holds the newest decoded frame, stamped with the moment it arrived. Two payoffs for the
    control loop: it never blocks on the network to get an image, and it can ask for the next
    frame that arrived *after* a chosen instant - which is how we throw away the stale, laggy
    frames deterministically (e.g. everything captured before the rover stopped) instead of
    guessing a frame count.
    """

    def __init__(self, http_addr, target_hz=15.0, timeout=2.0, stale_after_s=3.0):
        self.http_addr = http_addr
        self.timeout = timeout
        self.stale_after_s = stale_after_s
        self._min_period = (
            1.0 / target_hz
        )  # don't poll the camera faster than this (bounds load/heat)

        # One reused TCP connection for every capture - no per-frame handshake
        self._session = requests.Session()

        # Shared state, guarded by the lock
        self._lock = threading.Lock()
        self._frame = None
        self._stamp = 0.0  # perf_counter time the latest frame arrived
        self._last_ok = 0.0  # last successful grab, for the staleness watchdog

        self._running = False
        self._thread = None

    def start(self):
        # Launch the background grabber (daemon so it dies with the program)
        if self._running:
            return self
        self._running = True
        self._last_ok = time.perf_counter()  # grace period before the watchdog can trip
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        # Continuously fetch, decode, and store the newest frame
        while self._running:
            cycle_start = time.perf_counter()
            try:
                response = self._session.get(self.http_addr, timeout=self.timeout)
                buffer = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                if frame is not None:
                    now = time.perf_counter()
                    with self._lock:
                        self._frame = frame
                        self._stamp = now
                        self._last_ok = now
            except Exception:
                # Network hiccup: keep the old frame, the watchdog tracks how long we've been down
                time.sleep(0.05)

            # Pace the loop so we don't hammer the camera faster than target_hz
            leftover = self._min_period - (time.perf_counter() - cycle_start)
            if leftover > 0:
                time.sleep(leftover)

    def latest(self):
        # Newest frame and its arrival time (frame is None until the first one lands)
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            return frame, self._stamp

    def next_after(self, since, timeout=2.0):
        # Block until a frame that ARRIVED after `since`, then return (frame, stamp).
        # Returns (None, since) if nothing fresh shows up in time (likely a stalled camera).
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self._lock:
                frame = None if self._frame is None else self._frame.copy()
                stamp = self._stamp
            if frame is not None and stamp > since:
                return frame, stamp
            time.sleep(0.005)
        return None, since

    def is_stale(self):
        # True if the grabber hasn't pulled a fresh frame for a worryingly long time
        with self._lock:
            return (time.perf_counter() - self._last_ok) > self.stale_after_s

    def stop(self):
        # Stop the worker thread and close the connection
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._session.close()
