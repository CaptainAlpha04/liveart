"""LiveArt dev launcher.

Runs everything needed to bring the app up from a clean clone:

  1. Installs missing backend Python deps into the ``study`` conda env
     (fastapi, uvicorn, websockets, python-multipart, aiofiles).
  2. Runs ``npm install`` inside ``frontend/`` if ``node_modules`` is missing.
  3. Runs ``scripts/download_models.py`` if ``backend/models/`` has no ``*.pth``
     files yet.
  4. Starts the FastAPI backend (uvicorn on :8000) and the Vite frontend
     (:5173) concurrently, streaming both outputs to this terminal with
     colored prefixes.
  5. Ctrl+C shuts both children down cleanly.

Override the Python interpreter with the ``LIVEART_PY`` environment variable
(default: ``C:/Users/Ali/miniconda3/envs/study/python.exe``).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DEFAULT_PY = r"C:\Users\Ali\miniconda3\envs\study\python.exe"
STUDY_PY = Path(os.environ.get("LIVEART_PY", DEFAULT_PY))

BACKEND_DEPS = {
    # module import name -> pip spec
    "fastapi": "fastapi>=0.111.0",
    "uvicorn": "uvicorn[standard]>=0.29.0",
    "websockets": "websockets>=12.0",
    "multipart": "python-multipart>=0.0.9",
    "aiofiles": "aiofiles>=23.2.1",
}

C_CYAN = "\033[96m"
C_BLUE = "\033[94m"
C_MAGENTA = "\033[95m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_RESET = "\033[0m"


def log(prefix: str, msg: str, color: str = C_RESET) -> None:
    print(f"{color}[{prefix}]{C_RESET} {msg}", flush=True)


def enable_windows_ansi() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass


def check_python() -> None:
    if not STUDY_PY.exists():
        log(
            "setup",
            f"Python not found at {STUDY_PY}. Set LIVEART_PY to your env's python.exe.",
            C_RED,
        )
        sys.exit(1)
    log("setup", f"using python: {STUDY_PY}", C_CYAN)


def ensure_backend_deps() -> None:
    missing: list[str] = []
    for mod, pip_spec in BACKEND_DEPS.items():
        r = subprocess.run(
            [
                str(STUDY_PY),
                "-c",
                f"import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('{mod}') else 1)",
            ],
            capture_output=True,
        )
        if r.returncode != 0:
            missing.append(pip_spec)

    if not missing:
        log("setup", "backend deps: OK", C_CYAN)
        return

    log("setup", f"installing missing backend deps: {missing}", C_CYAN)
    subprocess.check_call([str(STUDY_PY), "-m", "pip", "install", *missing])
    log("setup", "backend deps installed", C_CYAN)


def ensure_npm_install() -> None:
    node_mods = ROOT / "frontend" / "node_modules"
    if node_mods.exists():
        log("setup", "frontend/node_modules exists, skipping npm install", C_CYAN)
        return
    log("setup", "running npm install (this may take a few minutes)...", C_CYAN)
    subprocess.check_call(
        "npm install",
        cwd=str(ROOT / "frontend"),
        shell=True,
    )
    log("setup", "npm install complete", C_CYAN)


def ensure_models() -> None:
    models_dir = ROOT / "backend" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    pth = list(models_dir.glob("*.pth"))
    if pth:
        log("setup", f"found {len(pth)} pretrained model(s), skipping download", C_CYAN)
        return
    log("setup", "downloading pretrained models (~20MB)...", C_CYAN)
    try:
        subprocess.check_call(
            [str(STUDY_PY), str(ROOT / "scripts" / "download_models.py")],
            cwd=str(ROOT),
        )
    except subprocess.CalledProcessError as exc:
        log(
            "setup",
            f"download_models.py exited with {exc.returncode} — continuing; "
            "you can train styles via the /training UI.",
            C_YELLOW,
        )


def stream_output(proc: subprocess.Popen, prefix: str, color: str) -> None:
    assert proc.stdout is not None
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip()
        if line:
            print(f"{color}[{prefix}]{C_RESET} {line}", flush=True)


def main() -> int:
    enable_windows_ansi()
    log("main", "LiveArt dev launcher", C_GREEN)
    check_python()
    ensure_backend_deps()
    ensure_npm_install()
    ensure_models()

    log("main", "starting backend on http://localhost:8000", C_GREEN)
    backend = subprocess.Popen(
        [
            str(STUDY_PY),
            "-m",
            "uvicorn",
            "backend.main:app",
            "--reload",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    # Give uvicorn a moment so its startup logs land first in the interleaved output.
    time.sleep(1.0)

    log("main", "starting frontend on http://localhost:5173", C_GREEN)
    frontend = subprocess.Popen(
        "npm run dev",
        cwd=str(ROOT / "frontend"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
    )

    threading.Thread(
        target=stream_output, args=(backend, "backend", C_BLUE), daemon=True
    ).start()
    threading.Thread(
        target=stream_output, args=(frontend, "frontend", C_MAGENTA), daemon=True
    ).start()

    log(
        "main",
        "both servers starting. Open http://localhost:5173 when Vite is ready. "
        "Press Ctrl+C to stop.",
        C_GREEN,
    )

    try:
        while True:
            if backend.poll() is not None:
                log("main", f"backend exited with code {backend.returncode}", C_RED)
                break
            if frontend.poll() is not None:
                log("main", f"frontend exited with code {frontend.returncode}", C_RED)
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        log("main", "shutting down...", C_YELLOW)
    finally:
        for name, proc in (("backend", backend), ("frontend", frontend)):
            if proc.poll() is None:
                try:
                    if os.name == "nt":
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        proc.terminate()
                except Exception:
                    proc.terminate()
        for name, proc in (("backend", backend), ("frontend", frontend)):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log("main", f"force-killing {name}", C_YELLOW)
                proc.kill()
        log("main", "goodbye", C_GREEN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
