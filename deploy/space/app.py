"""Launcher (b): Gradio-SDK Space shim (DESIGN.md 5.2). UNTESTED platform behaviour (risk R3).

A Gradio-SDK Space runs `python app.py` and expects a server on 0.0.0.0:7860. This shim runs the same plain aiohttp
app as the Docker Space (app/main.py); there is no Gradio UI. Free personal accounts may host Gradio Spaces only on
ZeroGPU hardware ([HF-OV]); if the platform insists on at least one @spaces.GPU function, set SPACES_ZERO_GPU=1
(ZeroGPU sets it itself) and a no-op one is declared. gradio / spaces are imported only in that case.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

if os.environ.get("SPACES_ZERO_GPU"):
    try:
        import spaces  # noqa: E402  (provided by the ZeroGPU runtime)

        @spaces.GPU
        def _noop_gpu():
            return None
    except Exception as e:  # pragma: no cover
        print(f"[app.py] spaces.GPU stub not declared: {e!r}", flush=True)

# The package directory app/ wins over this module file for `import app` (FileFinder checks packages first).
from app.main import run  # noqa: E402

if __name__ == "__main__":
    run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "7860")))
