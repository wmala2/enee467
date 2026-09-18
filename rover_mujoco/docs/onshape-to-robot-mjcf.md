# CAD → MJCF with onshape-to-robot

## What this is

If you design a new robot or prop in OnShape (the CAD tool), it doesn't start out usable in
MuJoCo — you have to export its shape, joints, and motors into MJCF, the XML format MuJoCo
reads. This guide is that export pipeline: a tool called `onshape-to-robot` pulls the design
straight out of OnShape and writes MJCF for you, so you don't have to build the file by hand.

Sections 1 to 6 below are the steps that get you from a finished OnShape assembly to a working
`assets/robots/<name>/` folder. Section 7 is a troubleshooting reference, indexed by the exact
error message you'll see — you don't need to read it up front, just come back to it if
something breaks.

## 1. Install

`onshape-to-robot` is included in the
[shared workspace installation](../../README.md#installation). After setup,
run the commands below from `rover_mujoco/` using that environment.

The exporter needs to prove to OnShape's servers that it's allowed to read your CAD file, and it
does that with an API key instead of your normal password. Sign in, open **My account** from the
top-right menu, pick **Developer** in the left sidebar, then the **API keys** tab, name the key
and grant it permissions:

![Creating an OnShape API key](images/create_new_api_key_labelled.png)

The exporter reads **three** environment variables, and all three are required:

```shell
export ONSHAPE_API=https://cad.onshape.com
export ONSHAPE_ACCESS_KEY=...
export ONSHAPE_SECRET_KEY=...
```

Note that the secret is `ONSHAPE_SECRET_KEY`, not `ONSHAPE_ACCESS_SECRET`. Keep all three in
the repo's gitignored `.env` and source it (`set -a; . ./.env; set +a`) rather than putting
them in `config.json`; the exporter still reads them from there but prints a deprecation
warning.

The [OnShape "File Export" app](https://cad.onshape.com/appstore/apps/File%20Export/698b9abb84adb494ca5d5d5a)
can also trigger an export directly from the CAD UI if you'd rather not run the CLI.

## 2. In the OnShape assembly, before exporting

The URL must point at an **assembly**, not a part studio, and that assembly must contain
something. Your browser's URL is whichever tab you are on, which is the most common reason an
export fails before it starts.

- Leave the root part un-fixed if the robot should be free-floating (a rover, an arm on
  wheels). Only apply OnShape's own **Fixed** mate/feature to the root if the robot is meant
  to be bolted to the world (e.g. a stationary arm). The exporter reads that flag to decide
  whether to emit a `<freejoint>` on the root body.
- **Mate connectors** at joint axes become MJCF joints; name them something you'll recognize
  in the XML (`left_axle`, `right_axle`, etc.), same as we did by hand on the rover.

To pick the assembly deliberately rather than trusting the address bar, list the document's
elements and check the one you want is not empty:

```shell
curl -s -u "$ONSHAPE_ACCESS_KEY:$ONSHAPE_SECRET_KEY" -H "Accept: application/json" \
  "https://cad.onshape.com/api/documents/d/<did>/w/<wid>/elements" \
| uv run python -c "import json,sys; [print(e['elementType'], e['id'], e['name']) for e in json.load(sys.stdin)]"

curl -s -u "$ONSHAPE_ACCESS_KEY:$ONSHAPE_SECRET_KEY" -H "Accept: application/json" \
  "https://cad.onshape.com/api/assemblies/d/<did>/w/<wid>/e/<assembly eid>" \
| uv run python -c "import json,sys; print(len(json.load(sys.stdin)['rootAssembly']['instances']), 'instances')"
```

## 3. Props with no joints: skip the exporter

A track, a ramp, or a wall has no joints, no actuators, and no free-floating base, so none of
what `onshape-to-robot` does applies. Pull the mesh straight out of the part studio and write
a few lines of MJCF around it:

```shell
URL="https://cad.onshape.com/api/partstudios/d/<did>/w/<wid>/e/<part studio eid>/stl?mode=binary&units=meter&grouping=true"
curl -s -o /dev/null -D /tmp/h.txt -u "$ONSHAPE_ACCESS_KEY:$ONSHAPE_SECRET_KEY" \
  -H "Accept: application/vnd.onshape.v1+octet-stream" "$URL"
LOC=$(awk 'tolower($1)=="location:"{print $2}' /tmp/h.txt | tr -d '\r')
curl -s -u "$ONSHAPE_ACCESS_KEY:$ONSHAPE_SECRET_KEY" \
  -H "Accept: application/vnd.onshape.v1+octet-stream" "$LOC" -o part.stl
```

The redirect is followed by hand on purpose; see section 7 for why `curl -L` returns a 401
here. Ask for `units=meter`; MuJoCo works in metres and OnShape will happily hand you
millimetres. `assets/objects/tracks/goomba_track.xml` is the result of exactly this, and shows
the two attributes a flat CAD part needs: `inertia="shell"` on the mesh, because a part 0.1 mm
thick is degenerate under MuJoCo's volume-based inertia, and `contype="0" conaffinity="0"` on
the geom, because paint on the floor is not something to drive into.

## 4. config.json

Everything about how the export should behave — which assembly to pull, which joints get
motors, what to name the robot — lives in one file, `config.json`, that you write by hand
before running the exporter:

```jsonc
{
  "url": "<onshape assembly URL>",
  "output_format": "mujoco",
  "robot_name": "rover",

  // Turn on actuators per-joint. "*" is the wildcard default; override specific
  // joints below it if some (like a caster) should stay passive.
  "joint_properties": {
    "*": { "actuated": false },
    "left_axle": { "actuated": true, "type": "velocity", "kv": 0.2, "limits": [-10, 10] },
    "right_axle": { "actuated": true, "type": "velocity", "kv": 0.2, "limits": [-10, 10] }
  }
}
```

Prefer `"type": "velocity"` (or `"position"`) over `"motor"` for small/light parts like
wheels. A raw torque motor is easy to mistune into instability; see section 7.

Names are matched with `fnmatch`, so `"*"` matches every joint and `"*_axle"` matches both
wheels. Matches merge **in the order the keys appear in the JSON**, so a specific joint must
come *after* the wildcard or the wildcard wins. The one exception is the key `"default"`,
which is always applied first no matter where it sits, so it is the safer way to express
"settings for everything, overridden below".

## 5. Run it, bring the output in

The argument is a path to the directory holding `config.json`, which has to be named exactly
that:

```shell
mkdir -p assets/robots/rover
# write config.json into that directory first
uv run onshape-to-robot assets/robots/rover
```

Useful flags when iterating, since each full run hits the OnShape API:

```shell
uv run onshape-to-robot assets/robots/rover --retrieve   # fetch CAD only, write robot.pkl
uv run onshape-to-robot assets/robots/rover --convert    # re-export from robot.pkl, offline
```

Fetch once with `--retrieve`, then tune `config.json` and re-run `--convert` as many times as
you like without touching the network. `--safe` disables config features that run custom
commands or imports, which is what you want when running someone else's `config.json`.

This produces `robot.xml` (the mesh/body/joint/actuator tree), `scene.xml` (floor + light +
an `<include>` of `robot.xml`, the same composition pattern our hand-built
`rover_scene.xml` uses), and an `assets/` folder of STLs. Drop the whole directory under
`assets/robots/<name>/` as-is; no path rewriting needed (unlike the `package://` URDF mesh
URIs we had to `sed` earlier).

A successful export on a single-part prop looks like this:

```
* Found total 0 degrees of freedom
* Found 1 root nodes:
  - Goomba_Track <1>
+ Adding part Goomba_Track <1>
WARNING: part Goomba_Track <1> has no dynamics (maybe it is a surface)
* Writing robot.xml
* Writing scene.xml
```

Read that warning rather than skipping past it. See section 7.

## 6. Quick validation checklist

A successful export doesn't guarantee a *correct* one — it's easy to get a robot that loads
fine but is the wrong size or has a joint pointing the wrong way. Load it in Python and check
the basics before you trust it:

```python
import mujoco

m = mujoco.MjModel.from_xml_path("assets/robots/<name>/scene.xml")
d = mujoco.MjData(m)
print(m.nbody, m.njnt, m.nu)  # sanity-check body/joint/actuator counts
for i in range(m.njnt):
    print(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i), m.jnt_type[i])  # 0 = free

d.ctrl[:] = ...  # a small nonzero command
for _ in range(1000):
    mujoco.mj_step(m, d)  # no "Nan, Inf or huge value" warnings should print
print(d.qpos[:3])  # moved a sensible distance, didn't teleport or freeze
```

Also check scale. OnShape units, if the assembly wasn't authored in metres, can produce a
robot a few orders of magnitude off, which shows up as either a robot floating away instantly
or barely moving under normal-looking actuator commands.

## 7. Troubleshooting (reference — read only if something breaks)

Every entry here is something we hit. They are grouped by the message you actually see,
because in each case the message names something other than the real cause.

### `ERROR: No Onshape API access key are set`

`ONSHAPE_API` is missing. The message blames the key, but the URL is what's absent. All three
variables in section 1 are required.

### `{"message":"Unauthenticated API request", "status":401}`

Check your system clock before your credentials. OnShape signs each request with the `Date`
header and rejects anything more than a few minutes from its own clock, which is
indistinguishable from a bad key. This bit us with the machine running about six minutes fast.

```shell
# what the server thinks the time is, versus what you think it is
curl -sI https://cad.onshape.com/api/users/sessioninfo | grep -i '^date:'
date -u '+%a, %d %b %Y %H:%M:%S GMT'

timedatectl                             # "System clock synchronized: no" is the tell
sudo timedatectl set-ntp true           # the fix
```

HTTP Basic auth does not involve the signature, so it still works when the signed request does
not. That makes it a one-command test:

```shell
curl -s -o /dev/null -w '%{http_code}\n' -u "$ONSHAPE_ACCESS_KEY:$ONSHAPE_SECRET_KEY" \
  https://cad.onshape.com/api/users/sessioninfo
```

200 there plus 401 from the exporter means the credentials are fine and the clock is not.

### 401 when fetching an STL, with credentials that work everywhere else

The STL endpoint answers with a 307 to a *different* host (`cad-usw2.onshape.com`), and
`curl -L` drops the `Authorization` header across hosts. Follow the redirect by hand, as in
section 3.

### `ERROR (400) ... "Element must be an assembly"`

The URL points at a part studio. `onshape-to-robot` reads assemblies only. List the document's
elements (section 2) and use the assembly's id.

### `KeyError: 'occurrences'` from `assembly.py`

The assembly exists but is empty; the API omits that key entirely when nothing has been
inserted. That was the state the `Goomba_Track` document was in: one part in the part studio
and an `Assembly 1` holding nothing. Dropping the part studio into the assembly by itself is
enough.

### `No module named onshape_to_robot.__main__`

`onshape-to-robot` ships as a console script, not a runnable module. Use
`uv run onshape-to-robot assets/robots/rover`, as in section 5.

### `Nan, Inf or huge value in QACC`, shortly after adding actuated joints

Explicit-Euler with velocity or position actuators and tiny inertias can blow up. This is what
bit us on the rover's wheels, where angular velocity diverged to NaN within a few steps.
MuJoCo's own [Numerical Integration docs](https://github.com/google-deepmind/mujoco/blob/main/doc/computation/index.rst)
call out exactly this case, "stiff springs, position servos or strong damping interact with
contacts", and name `implicitfast` as *"the recommended integrator for most models"* because
it has Euler's cost with much better stability. Add `<option integrator="implicitfast"/>` to
`scene.xml` before retuning any gains. That was the actual fix here; neither a `discrete`
integrator swap nor gain retuning was needed once we made that one change.

Do not start by suspecting the mesh. We initially swapped the wheel STLs for cylinder
collision primitives, but per
[XMLreference.rst](https://github.com/google-deepmind/mujoco/blob/main/doc/XMLreference.rst),
"collision detection works with the convex hull of the mesh", so MuJoCo already collides every
mesh via its convex hull and a coarse or non-manifold STL is not by itself a source of
instability. Primitives are still worth using for performance on parts that touch the ground
often; just don't expect them to fix NaNs.

### A part flies away the moment anything touches it

Preceded by `WARNING: part <name> has no dynamics (maybe it is a surface)`. A part with no
volume, a track or a decal or anything modelled as a sheet, exports with `mass="1e-09"` and a
matching `1e-09` inertia. It also gets a `<freejoint>`, because the root was not marked Fixed
in OnShape. On its own it looks fine: it settles 0.2 mm into the floor and sits there with zero
velocity indefinitely. But `f = ma` with `m = 1e-9` means any contact launches it. Measured on
the exported Goomba track, a 1 mN push, roughly a thousandth of the force a 1.5 kg rover
delivers in a collision, accelerated it to 499 km/s and 125 km away in half a second.

Three ways out: mark the root Fixed in OnShape so no freejoint is emitted, delete the
`<freejoint>` and give the body a real mass by hand, or skip the exporter for props
(section 3).

### Parts penetrate each other at rest

OnShape assemblies routinely have brackets, motors, and fasteners flush against each other by
design. The exporter keeps each part as its own body and geom, unlike a merged URDF link, so
those touching surfaces register as penetrating contacts. Use `geom_properties` wildcards in
`config.json`, or hand-edit `contype`/`conaffinity` after export, to disable collision on parts
that are cosmetic or rigidly interior. Only surfaces that actually touch the ground or other
objects need it on.

### Edits to `scene.xml` are ignored, or a re-export changes nothing

`scene.xml` is written only if it is not already there. Your edits survive a re-export, which
is the behaviour you want, but it also means a scene you customized months ago silently keeps
its old contents while `robot.xml` underneath it changes. Delete it if you want the generated
one back. The generated version is minimal: a skybox, a headlight, one directional light, and
a checkerboard ground plane, with no `<option>` block at all, which is why the integrator fix
above is something you add rather than something you change.

### A joint ignored its `joint_properties`

Matching order. A specific joint listed *before* the wildcard is overwritten by it; see
section 4.

### The robot has no free joint

It's silently skipped if the root was marked Fixed in OnShape. The exporter names it after the
root link, as `<root link name>_freejoint`, so you can grep `robot.xml` for it rather than
assuming.

### The exporter emitted two geoms for every mesh

That's intended. They come from the default classes at the top of `robot.xml`: `visual` is
`group="2"` with `contype`/`conaffinity` 0, `collision` is `group="3"`. For something the rover
drives *over* rather than into, drop the collision geom; for something it collides with,
replace the mesh collision with a primitive.
