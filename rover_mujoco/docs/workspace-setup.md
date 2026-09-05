# Workspace setup

Getting this repo running, verifying MuJoCo actually works on your machine, and extending
the dependency set with whatever else you end up needing.

## 1. Prerequisites

- **Python 3.12.** Check with `python3 --version`. If you're outside that range, install
  a matching version: `uv python install 3.12` will fetch one for you without touching your
  system Python. (This floor was 3.10-3.12 before the sim-to-real work in
  `docs/rl-line-follower.md` — `better-actuator-models` requires 3.12, so the range narrowed
  to keep the whole project on one Python version.)
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)**, which manages the
  virtual environment and every dependency below it:
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - Or via `pip install uv` / a package manager (`brew install uv`, `winget install astral-sh.uv`) if you prefer.
- **[VSCode](https://code.visualstudio.com/download)**, optional but recommended so
  everyone's using the same setup. Once you open this repo's folder in VSCode, it should
  prompt you to install the recommended extensions (Python, Ruff, a MuJoCo viewer) from
  `.vscode/extensions.json`; accept that prompt, or install them manually from the Extensions
  panel.

## 2. Clone and install

This directory is a member of the parent repo's
[uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/): it keeps its own
`pyproject.toml`, but the lockfile and virtual environment live at the repository root and are
shared with the high-level rover code. So clone and sync from the root:

```shell
git clone https://github.com/CursedRock17/matrix_lab_rover_above.git
cd matrix_lab_rover_above
uv sync --extra cpu     # or --extra cu121 on a machine with an NVIDIA GPU
```

`uv sync` reads the root `pyproject.toml`/`uv.lock` and builds one `.venv` with every dependency
— this project's *and* the high-level packages' — pinned to the same versions everyone else is
using. The `--extra` picks which PyTorch build to pull; nothing in `rover_mujoco/` needs torch
directly, but the same environment serves both halves of the repo. You don't need `uv init`:
that's for starting a *new* project, and this one already has a `pyproject.toml` from the clone.

The commands in these docs are written relative to this folder, so `cd rover_mujoco` first and
run them as-is — `uv run` walks up to the workspace root to find the environment on its own.

If you're using VSCode, point its Python interpreter at the root `.venv` (Command Palette →
"Python: Select Interpreter" → the one inside `matrix_lab_rover_above/.venv`) so
linting/autocomplete see the same packages you're running.

## 3. Sanity check: does MuJoCo actually open a window?

```shell
uv run python -m mujoco.viewer --mjcf=assets/robots/rover/rover_scene.xml
```

This should pop up MuJoCo's native viewer with the rover sitting on the floor. If it does,
your graphics/OpenGL setup is working end to end. That's worth checking before debugging
anything else, since a blank window or a crash here is almost always a driver/display issue,
not a bug in this repo. Close the window (or Esc) to exit.

If nothing opens: on Linux, make sure you're not in a headless/SSH session without X forwarding
or a display; on all platforms, updating GPU drivers is the most common fix.

## 4. Adding your own packages

`pyproject.toml` already has the baseline this project needs: `mujoco`, `gymnasium`,
`stable-baselines3`, `onshape-to-robot`. The dev tools (`ruff`/`ty`/`pytest`) live in the root
`pyproject.toml` instead, as a workspace-wide `[dependency-groups] dev`, so one `uv sync` installs
them for both halves of the repo. To add something on top of that:

```shell
cd rover_mujoco && uv add some-package   # simulation dependency -> rover_mujoco/pyproject.toml
uv add --dev some-dev-tool               # linting/testing tool -> run from the repo root
```

Run the first from inside `rover_mujoco/` so it lands in *this* project's `pyproject.toml` rather
than the root one; either way the lockfile they update is the shared root `uv.lock`. Commit both
so everyone else picks up the same addition on their next `uv sync`. If you ever need a distributable wheel
(rather than just a local `.venv`), `uv build` packages the project using exactly what's in
`pyproject.toml`, extras included.

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
