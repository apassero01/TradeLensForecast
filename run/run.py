#!/usr/bin/env python3
"""
Dev launcher for TradeLens:
  * Purges stale Celery tasks
  * Starts Daphne (ASGI) + Celery worker
  * Cleans up every child process on exit
"""
import os
import sys
import signal
import subprocess
import logging
import atexit
from pathlib import Path

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("runner")

# ─── Paths / Django env ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "TradeLens.settings")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _spawn(cmd, *, name, cwd=PROJECT_DIR, env=None):
    """Start *cmd* in its own process‑group and return the Popen handle."""
    log.info("Starting %s: %s", name, " ".join(cmd))
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        preexec_fn=os.setsid,  # new group -> easy killpg
    )

def _purge_celery():
    log.info("Purging pending Celery tasks…")
    subprocess.run(
        ["celery", "-A", "TradeLens", "purge", "-f"],
        cwd=PROJECT_DIR,
        check=True,
    )

# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    debug = os.environ.get("DEBUG_MODE") == "1"
    concurrency = "1" if debug else "4"
    pool = "solo" if debug else "threads"

    _purge_celery()

    if debug:
        # In debug mode, run Celery worker in the main process for debugging
        daphne_proc = _spawn(
            ["daphne", "-b", "0.0.0.0", "-p", "8000", "TradeLens.asgi:application"],
            name="daphne",
        )
        # Register cleanup to terminate Daphne when the main process exits
        atexit.register(
            lambda: os.killpg(daphne_proc.pid, signal.SIGTERM) if daphne_proc.poll() is None else None
        )
        # Import and run the Celery worker in the main process
        from TradeLens.celery import app  # Adjust this import if your app is defined elsewhere
        app.worker_main(
            ["worker", "--loglevel=debug", f"--pool={pool}", f"--concurrency={concurrency}"]
        )
    else:
        # Normal mode: run both as subprocesses
        procs = [
            _spawn(
                ["daphne", "-b", "0.0.0.0", "-p", "8000", "TradeLens.asgi:application"],
                name="daphne",
            ),
            _spawn(
                ["celery", "-A", "TradeLens", "worker", "--loglevel=debug", "--pool=" + pool, f"--concurrency={concurrency}"],
                name="celery‑worker",
            ),
        ]

        # Forward ^C / kill to every child process‑group
        def _shutdown(signum, _frame):
            log.info("Shutting down (%s)…", signal.Signals(signum).name)
            for p in procs:
                if p.poll() is None:  # still alive
                    os.killpg(p.pid, signal.SIGTERM)  # whole group
            for p in procs:
                p.wait()
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        # Wait while children run
        try:
            while True:
                for p in procs:
                    if p.poll() is not None:
                        log.warning("%s exited with code %s", p.args[0], p.returncode)
                        _shutdown(signal.SIGTERM, None)
                signal.pause()  # wake up only on signal
        except KeyboardInterrupt:
            _shutdown(signal.SIGINT, None)

if __name__ == "__main__":
    main()