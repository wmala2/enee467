# Workspace setup

## What this is

This is the checklist for getting `rover_mujoco` (the simulator) running on your machine from
nothing: installing the tools, syncing the shared environment, and confirming MuJoCo can actually
open a window before you try to debug anything more interesting.

## 1. Prerequisites

Follow the [root installation guide](../../README.md#installation) for Git, uv,
and the CPU/GPU choice. The workspace requires Python 3.12; uv selects a compatible
interpreter and downloads one if needed during setup.

- **[VSCode](https://code.visualstudio.com/download)**, optional but recommended so
  everyone's using the same setup. Once you open this repo's folder in VSCode, it should
  prompt you to install the recommended extensions (Python, Python Debugger, Ruff, ty, Even
  Better TOML) from `.vscode/extensions.json`; accept that prompt, or install them manually
  from the Extensions panel. Opening `mini_claw.code-workspace` instead also loads the two
  firmware repos beside this one and recommends PlatformIO. Every command below still works
  from a plain terminal.

## 2. Clone and install

Complete [Repository Setup](../../README.md#repository-setup) from the repository
root. The simulator is a member of that
[uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/): it has its
own `pyproject.toml`, and shares the root `uv.lock` and `.venv` with the rover code.
Simulation and policy training use the CPU build.

Check the workspace's interpreter from the repository root:

```shell
uv run python --version
```

It should report Python 3.12. The viewer command below runs from `rover_mujoco/`;
`uv run` still selects the same shared environment from there. Other guides state
their working directory beside their commands.

If you're using VSCode, the Python extension should pick up the root `.venv` on its own; if it
doesn't, point it there (Command Palette → "Python: Select Interpreter" → the one inside
`matrix_lab_rover_above/.venv`) so linting/autocomplete see the same packages you're running.
**Terminal → Run Task...** lists the commands from these docs (install, viewer, teleop, PID,
training, evaluation, tests, checks), each already set to run in the right folder, and the
Run and Debug panel launches the simulator scripts with breakpoints.

## 3. Sanity check: does MuJoCo actually open a window?

```shell
cd rover_mujoco
uv run python -m mujoco.viewer --mjcf=assets/robots/rover/rover_scene.xml
```

This should pop up MuJoCo's native viewer with the rover sitting on the floor. If it does,
your graphics/OpenGL setup is working end to end. That's worth checking before debugging
anything else, since a blank window or a crash here is almost always a driver/display issue,
not a bug in this repo. Close the window (or Esc) to exit.

If nothing opens: on Linux, make sure you're not in a headless/SSH session without X forwarding
or a display; on all platforms, updating GPU drivers is the most common fix.

## 4. Adding your own packages

Return to the repository root (`cd ..` after the viewer step), then follow
[Adding dependencies](../../README.md#adding-dependencies). Use its
`--package rover_mujoco` command for simulation dependencies and its `--dev`
command for shared development tools. This keeps the package declarations and
shared lockfile in agreement.

## 5. Cross-platform notes (Windows / Mac / Linux)

Everything above is plain Python packaging: no OS-specific paths or shell scripts, so it's
expected to work identically on all three. A few things worth knowing:

- `uv` itself is cross-platform and installs the same way conceptually everywhere (see the
  prerequisite links above for the exact command per OS).
- MuJoCo's Python bindings ship prebuilt wheels for Windows, macOS (Intel + Apple Silicon),
  and Linux. No compiler is needed on any of them.
- The one platform-specific step is the viewer's OpenGL backend, which is handled
  automatically by MuJoCo on all three; the sanity check in step 3 is the way to confirm it's
  actually working on *your* machine rather than assuming it from this doc.
