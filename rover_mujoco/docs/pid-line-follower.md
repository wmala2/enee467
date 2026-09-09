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

The real rover's camera mount can tilt, and the sim models that with the `top_cam` quaternion
in `assets/robots/rover/rover.xml` — the single place the angle is set. It runs from 0
(straight down) to 90 (forward-facing, perpendicular to the rover), written as
`quat="0 0 sin(t/2) cos(t/2)"`; the 60° default is `quat="0 0 0.5 0.866"`.

This used to be applied at runtime by an `envs/camera.py` helper that overwrote `cam_quat`
after compile, so a script could pick its own angle. Every caller passed the same 60°, so the
indirection only obscured where the angle really came from. When the mount is CADed in with a
rotational joint, the tilt becomes that joint's position and the quat becomes its rest pose.

Worth knowing before you start changing it:

- The default is **60°**. The camera sits directly above the caster, so a shallow tilt points
  it at the rover's own chassis rather than the track — 13.2% of the frame is its own body.
  Measured with `line_error()` over 12 poses on each track: 45° finds the line 0/12 times,
  55° finds it 9/12 and 7/12, and 60° finds it 12/12 on both, well centred.
- This replaced a `~15.2°` default that belonged to an older mount, where the camera sat on
  the opposite end of the chassis from the caster and the drive convention was inverted to
  match. See the note in `rover.xml`.

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
