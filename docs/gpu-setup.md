# Matching torch to your GPU

This is the main reference for getting torch working on an NVIDIA GPU in this repo. There's no
single CUDA build that works on every machine, so rather than guessing, follow the steps below to
confirm what your specific driver supports and get a torch build that matches it.

## 0. Try the built-in `cu121` extra first

```bash
uv sync --extra cu121
uv run --extra cu121 python -c "import torch; print(torch.cuda.is_available())"
```

`cu121` (CUDA 12.1) is checked into `pyproject.toml` as a backup, and it happens to work on most
NVIDIA machines because CUDA drivers are backward-compatible with older CUDA builds. If the command
above prints `True`, you're done — skip straight to using `--extra cu121` on your `uv run` calls
(see the root README's note on why it needs repeating every time).

If it prints `False`, or you get a "driver too old" warning even though `nvidia-smi` clearly shows a
working GPU, `cu121` isn't a match for your card/driver. Keep reading to add an extra pinned to your
own CUDA version instead of editing the existing `cu121` one.

## 1. Find your driver's max supported CUDA version

```bash
nvidia-smi
```

Look at the top-right of the output, e.g. `CUDA Version: 12.8`. That's the *highest* CUDA version
your installed driver can run -- not necessarily what's currently installed on the system. Any torch
build compiled for that version or older will work.

## 2. Find the matching torch build

Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), and use the
matrix there (Stable, your OS, Pip, Python) to pick the CUDA version closest to (but not above) the
number from step 1. It'll show you an index URL like:

```
https://download.pytorch.org/whl/cu128
```

The `cuXXX` suffix (`cu128` here) is the name you'll use for the new extra. You can also browse
[download.pytorch.org/whl/torch](https://download.pytorch.org/whl/torch/) directly to see every
published build.

## 3. Add the extra to `pyproject.toml`

Using `cu128` as the example, add a block parallel to the existing `cu121` one in three places:

```toml
[project.optional-dependencies]
cpu = ["torch", "torchvision"]
cu121 = ["torch", "torchvision"]
cu128 = ["torch", "torchvision"]          # <- add this line

[[tool.uv.index]]
name = "pytorch-cu128"                    # <- add this whole block
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv]
conflicts = [[{ extra = "cpu" }, { extra = "cu121" }, { extra = "cu128" }]]   # <- add cu128 here

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", extra = "cpu" },
    { index = "pytorch-cu121", extra = "cu121" },
    { index = "pytorch-cu128", extra = "cu128" },      # <- add this line
]
torchvision = [
    { index = "pytorch-cpu", extra = "cpu" },
    { index = "pytorch-cu121", extra = "cu121" },
    { index = "pytorch-cu128", extra = "cu128" },      # <- add this line
]
```

Only `torch` and `torchvision` need a `[tool.uv.sources]` entry -- they're the only libraries this
repo pulls straight from a CUDA-specific PyTorch index. Other CUDA-sensitive packages (e.g.
`xformers`, pulled in transitively by `depth-anything-3`) already have per-CUDA-version release
metadata on PyPI, so `uv lock` picks the build matching whichever torch you resolve automatically --
you don't need to add sources for those yourself.

## 4. Lock and sync

```bash
uv lock
uv sync --extra cu128
```

## 5. Verify

```bash
uv run --extra cu128 python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` should print `True`. Remember `--extra cu128` needs to be repeated on
every future `uv run`/`uv sync` call for GPU-dependent scripts (see the root README's note on why
`uv run` doesn't remember it for you).
