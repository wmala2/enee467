# LLM Hybrid

## What this is

This folder connects the rover to Large Language Models running locally through [Ollama](https://docs.ollama.com/api/introduction). Instead of a purpose-built network like YOLO that only recognizes a fixed list of trained classes, these examples use general-purpose models that can answer open-ended questions about what the camera sees — and even decide how the rover should drive.

## Setup

Ollama is the local server that actually runs the model; the Python code below just talks to it.

1) Install the Ollama server (one time). The recommended method is the official one-line installer, which always fetches the latest release:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
If you cannot run the installer (no admin rights, or Windows), download the latest standalone build from the [Ollama releases page](https://github.com/ollama/ollama/releases/latest) and unpack it:
```bash
# Linux example — replace the filename with the latest release from the page above
curl -sL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst -o /tmp/ollama.tar.zst
mkdir -p ~/.local/ollama && tar --zstd -xf /tmp/ollama.tar.zst -C ~/.local/ollama
```

2) Start the server (leave it running in its own terminal):
```bash
ollama serve                        # if installed via the official installer
# or, for the standalone build:
~/.local/ollama/bin/ollama serve
```

3) The Python client is installed with the main project (`uv sync` from the repository root). The classes below download their models automatically on first use.

## Example Executables

| Executable | Purpose |
|------------|---------|
| [LLM Chat](llm_chat.py) | Lets you type prompts into a local llama3.2:3b model and read its answers, with conversation memory. |
| [Object Identifier](object_identifier.py) | Scans one camera image and returns a structured JSON response naming the main object in frame, like a YOLO detection. |
| [LLM Driver](llm_driver.py) | Proof of concept where the LLM watches the camera stream and outputs the rover's velocity JSON message, so the LLM drives the rover. |

`object_identifier.py` prints one timed line per scan, for example (illustrative, not a captured
real run):

```
(6.8s) {"object": "water bottle"}
```

<!-- TODO: screenshot needed -- a terminal session running llm_chat.py or object_identifier.py
     against a live camera frame, once Ollama and a model are available to capture one. -->

## Choosing a Model
The vision classes default to **moondream** (1.8B), because it is the only vision model that answers in roughly 5–10 seconds per frame on a laptop CPU (measured ~10–13 s on a power-throttled laptop; faster on wall power). The larger **qwen2.5vl:3b** gives noticeably better answers ("water bottle" instead of "bottle") but takes about 2 minutes per new frame on CPU — fine for a one-off photo, not for a control loop. If you have a GPU, pass `model="qwen2.5vl:3b"` to either class and enjoy both speed and accuracy.

## Estimating an Object's Pose (X, Y, Z) from a Single Image
Classic pose estimation needs two cameras (stereo) to triangulate depth, but neural networks can now estimate depth from a single 2-D image: they are trained on millions of photos with known depth, so they learn cues humans use too — how big familiar objects look, how textures shrink with distance, how scenes are usually laid out. The practical recipe for our rover is **monocular metric depth estimation**: run a model like [Depth Anything V2 (metric)](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf) on the ESP32 camera's JPEG (via the Hugging Face `transformers` library's `pipeline("depth-estimation")`), which returns a depth in meters for every pixel. Take the center pixel `(u, v)` of a YOLO bounding box, read its depth `Z`, and back-project with the camera calibration we already use for ArUco: `X = (u - cx) * Z / fx`, `Y = (v - cy) * Z / fy`. That yields the same (X right, Y down, Z forward) pose the ArUco tracker gives — for any object, with no tag. The model runs on the laptop, not the ESP32 (which only streams the image), and at a few seconds per frame on CPU it is slower than ArUco, but it works.
