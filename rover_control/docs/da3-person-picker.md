# Choosing who to follow: `da3_person_picker.py`

`rover_control/examples/da3_person_picker.py` is the "choose who to follow" version of
`da3_object_movement.py`. Both ask a GPU server (`depth_anything_server`) for a metric depth
map and drive the rover toward a person using PID control. The difference is what happens
before driving starts: `da3_object_movement.py` just follows the first person YOLO happens to
see, while this one shows you *everyone* currently visible, lets you click on one, and then
keeps following that specific person as they move.

This doc walks through the code section by section - useful both for understanding it and for
checking it does what it claims to, piece by piece.

## Requirements

Same as the rest of `depth_anything_server`'s examples: a GPU server running
`depth_server.py` on the network (see [depth_anything_server/README.md](../../depth_anything_server/README.md#quick-start)),
and a rover laptop that can reach both the server and the rover's ESP32 camera.

`--server`, `--camera-ip`, and `--rover-ip` are command-line flags, not hardcoded constants,
because every setup's IPs are different - one student's rover, camera, and GPU laptop will
never share IPs with another student's. `--rover-ip` is where motor commands get sent (UDP) -
different from `--camera-ip`, which is where camera frames get fetched (HTTP). Omit `--rover-ip`
to fall back to the `ROVER_IP` environment variable, or `network_interface.py`'s own default.

```shell
uv run --extra cu121 rover_control/examples/da3_person_picker.py \
    --server <YOUR_GPU_LAPTOP_IP>:5000 \
    --camera-ip <YOUR_ROVER_CAMERA_IP>:80 \
    --rover-ip <YOUR_ROVER_IP>
```

Both flags fall back to `PersonPicker.SERVER_IP`/`ROVER_CAMERA_IP` if omitted - placeholder
values from this project's example network, not yours.

## The two-phase state machine

The whole script is one loop with a single piece of state, `self.chosen_id`:

- **`chosen_id is None` (picking):** every person currently visible gets shown, labeled, and
  is clickable. Nothing drives yet.
- **`chosen_id` is set (following):** the rover tries to find that specific person in each new
  frame and drives toward them.

Nothing else branches the script's behavior - no separate "modes" to track, just one variable
that's either `None` or an ID.

## Tracking people with persistent IDs

A single detection call (like `YOLOExtractor.process()`, used by `da3_object_movement.py`)
finds people in one frame with no memory of previous frames - if two people are in the shot,
there's no way to know that "person A" in this frame is the same person as "person A" in the
last frame. That's fine for following whoever's simply closest, but not for following one
*specific* chosen person while others are also in view.

`YOLOExtractor.track()` (added in `YOLO_agent/YOLO_extractor.py` alongside the existing
`process()`, which is untouched) solves this by calling Ultralytics' built-in tracker instead
of a plain detection pass:

```python
results = self.model.track(
    frame,
    persist=True,  # remember previous frames' tracks instead of starting fresh each call
    conf=self.confidence,
    imgsz=self.imgsz,
    classes=classes,
    verbose=False,
)[0]
```

`persist=True` is what makes this tracking rather than one-off detection: Ultralytics keeps
its own internal state between calls and assigns the same numeric ID to what it believes is
the same physical person, using their motion and appearance across frames (ByteTrack, under
the hood - a black box as far as this script is concerned). Every returned detection now
carries an `"id"` key alongside the usual `"box"`/`"confidence"`/`"name"`. A detection can come
back with `box.id is None` if the tracker hasn't confirmed an ID for it yet - `track()` just
drops those rather than reporting a person with no usable identity.

In `da3_person_picker.py`, every loop tick calls this regardless of whether we're picking or
following:

```python
people = self.extractor.track(frame, classes=[PERSON_CLASS_ID])
people = [p for p in people if p["confidence"] > self.CONFIDENCE_THRESH]
```

`classes=[PERSON_CLASS_ID]` (COCO class `0`) keeps the tracker from wasting time on other COCO
classes. The confidence filter afterward matches `da3_object_movement.py`'s stricter 0.80
threshold - the extractor's own default is a looser 0.5.

## Turning depth into a pickable image

While `chosen_id is None`, `_show_picker()` builds the image you click on:

```python
def depth_to_colormap(depth_map, max_display_depth=10.0):
    depth_clipped = np.clip(depth_map, 0, max_display_depth)
    depth_normalized = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
```

This is the same normalize-then-colorize recipe as
[`depth_map_local_camera.py`](../../depth_anything_server/examples/depth_map_local_camera.py):
clip anything past 10 m (so one far-away outlier pixel doesn't wash out the color scale),
squash the remaining range into 0-255, then map that onto a color gradient (`INFERNO`: warm
colors are close, cool colors are far).

`_show_picker()` then draws every tracked person's box and ID number directly on that
colormap, and - this is the part that makes clicking work - records each box's pixel
coordinates in `self._visible_boxes`, keyed by track ID:

```python
self._visible_boxes[person["id"]] = (x1, y1, x2, y2)
```

`_on_click()`, registered once via `cv2.setMouseCallback`, is what OpenCV calls when you click
inside that window. It just checks which recorded box (if any) contains the click point, and
if one does, sets `self.chosen_id`:

```python
def _on_click(self, event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    for track_id, (x1, y1, x2, y2) in self._visible_boxes.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            self.chosen_id = track_id
```

Because `_visible_boxes` is refreshed every tick from that tick's live tracker output, a click
always tests against boxes from a genuinely recent frame - there's no separate "freeze the
picker" step.

## Once you've picked someone

The main loop looks up `chosen_id` in this tick's tracked people:

```python
match = next((p for p in people if p["id"] == self.chosen_id), None)
```

If found, distance comes from `calculate_filtered_distance()` - identical logic to
`da3_object_movement.py`'s method of the same name (see that file, or
[depth_anything_server/README.md](../../depth_anything_server/README.md#rationale)'s
"Filtering the Images" writeup, for why a raw average over the box is noisy and a histogram
peak isn't): bucket every depth pixel inside the box into 5 cm bins, find the most common
bucket, and average only the pixels within 15 cm of that peak. That throws out background
pixels that leak in around an irregularly-shaped person inside a rectangular box.

Sideways offset is worked out from how far the box's center sits from the image's center, in
pixels, converted to meters with the camera's focal length - then both numbers feed the same
two-PID steering (`distance_pid` for forward speed, `heading_pid` for turning) that
`da3_object_movement.py` uses, via `wheel_speeds_for()`.

## When the person disappears

If `chosen_id` isn't found in a tick's people (they left frame, or something broke the track),
the rover sends zero velocity and just keeps looping, still calling `track()` every tick:

```python
if match is None:
    print(f"Lost person #{self.chosen_id}, waiting for them to reappear...")
```

This is deliberately the simple option: no fallback state, no re-showing the picker. The
trade-off is that Ultralytics' tracker doesn't always hand back the *same* ID after a real
occlusion or someone leaving and re-entering frame - if that happens, this script just waits
indefinitely at that spot until you restart it and pick again. Worth knowing if the rover
seems to "give up" on someone who's clearly still in the room.

## Controls

- **Click** a person's box in the depth-colored window to choose them.
- **`q`** (inside either video window) aborts.
- **Ctrl-C** stops the rover immediately, same as every other example here - `update()`'s
  `finally` block always stops the motors, the background depth-fetch thread, and closes every
  window, no matter how the loop exits.

## Not yet verified

This hasn't been run against a real GPU depth server, rover, or camera - that requires
hardware this wasn't written on. Once you run it for real, it's worth specifically checking:
whether `persist=True` behaves as expected across the picking→following transition (IDs
shouldn't reset), and whether the click hit-testing in `_on_click` lines up correctly with
what's drawn (screen coordinates vs. the depth map's actual resolution should match, but worth
eyeballing).
