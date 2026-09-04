# Classical line following: PID and bang-bang

`scripts/line_follower.py` drives the rover around one of the black line tracks
(`assets/objects/tracks/`) using nothing but its onboard camera: find where the line is in the
image, steer to keep it centered. This is the classical-control counterpart to
`docs/rl-line-follower.md`'s RL version, and a good place to build intuition before jumping to
RL — same sensor, same steering problem, much simpler controller.

## Two controllers, same error signal

Both controllers steer off `line_error()`: the black line's centroid, projected to a single
number in `[-1, 1]` (0 = centered, negative = line is to the camera's left, positive = right).

- **PID** (`CONTROLLER = "pid"`, the default): proportional-integral-derivative control on
  that error, same as most real line followers. Tune `KP`/`KI`/`KD` at the top of the file.
- **Bang-bang** (`CONTROLLER = "bang_bang"`): the simplest controller that can work at all —
  turn hard one way or the other with no proportional response, plus a small deadband
  (`BANG_DEADBAND`) around zero so it doesn't chatter when already centered. Worth trying
  first: it makes obvious why PID's proportional term (a *gentler* turn for a *smaller* error)
  produces smoother tracking than bang-bang's always-hard turns.

Switch between them by editing `CONTROLLER` and re-running; both share the same camera,
tracks, and starting logic.

## Camera angle

The real rover's camera mount can tilt, and this sim models that: `CAMERA_ANGLE_DEG` at the
top of `line_follower.py` sets it anywhere from 0 (straight down) to 90 (forward-facing,
perpendicular to the rover), via `envs/camera.py`'s `set_camera_tilt()`. That function mutates
the compiled model's `cam_quat` directly (the same technique `LineFollowerEnv`'s domain
randomization uses to jitter the camera each episode), so any script can pick its own angle at
runtime instead of hand-editing `rover.xml`.

Worth knowing before you start changing it:

- The default, `~15.2°`, is this mount's original tuned angle — the one all the tracking
  numbers below were measured at.
- This particular camera is mounted on the chassis's *rear* overhang (see the comment in
  `rover.xml`), not out over the front. It doesn't look ahead in the direction of travel, and
  that's fine: centroid-based steering only needs to see the line somewhere near the rover, not
  predict a path ahead of it. If you build a *forward*-looking mount instead (a real design
  choice worth trying), expect to re-tune the mount position, not just the angle.
- Higher angles put progressively more of the rover's own chassis in frame and less ground,
  since the mount sits close to the chassis rather than out on a boom. Straight down (0°) and
  the default (~15°) both keep a usable strip of floor in view; angles approaching 90° see
  none of the ground at all (there's nothing to find a centroid of), so they're not usable for
  this task even though the code will happily let you set them. Render a frame and look at it
  before assuming a given angle will track anything.

## Tracks

The two tracks (`track_oval`, `track_s_curve`) are generated procedurally by
`scripts/gen_track.py` from simple parametric waypoints, not exported from OnShape. If you'd
rather design a track visually and export it the same way the rover itself gets exported, see
`docs/onshape-to-robot-mjcf.md` for the CAD-to-MJCF workflow; a track is just flat geometry, so
the same pipeline applies. Either way, a new track needs a waypoint function (see
`envs/tracks.py`) so `line_follower.py` knows where to spawn the rover facing the right
direction — that's the only piece tied to *how* the track's shape was authored.

## Running it

```shell
uv run python scripts/line_follower.py
```

Picks a track at random each run and drives it with whichever `CONTROLLER` is set. Verified
end-to-end headlessly on both tracks with both controllers before shipping this: PID completes
the oval with zero line-loss frames, and briefly loses/reacquires the line a handful of times
on the tighter S-curve (its gains are tuned loosely, not to be optimal — a good first thing to
improve); bang-bang completes both tracks with zero line-loss, just less smoothly.
