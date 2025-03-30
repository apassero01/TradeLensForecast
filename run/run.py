import os
import sys
import django
import threading
import subprocess
import logging

# Configure logging.
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Determine the current and parent directories.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
logger.debug("Current directory: %s", current_dir)
logger.debug("Parent directory: %s", parent_dir)

# Add the parent directory to sys.path so that the package 'run' is importable.
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    logger.debug("Added parent_dir to sys.path: %s", parent_dir)
else:
    logger.debug("Parent directory already in sys.path: %s", parent_dir)

# Set up Django before any other imports.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TradeLens.settings')
django.setup()

def start_daphne():
    from daphne.cli import CommandLineInterface
    cli = CommandLineInterface()
    logger.info("Starting Daphne on port 8000")
    cli.run(['-b', '0.0.0.0', '-p', '8000', 'TradeLens.asgi:application'])

def start_celery_worker():
    # Check if DEBUG_MODE is set to '1' to determine concurrency.
    concurrency = 1 if os.environ.get('DEBUG_MODE') == '1' else 4  # Default to 4 workers in non-debug mode.
    logger.info("Starting Celery worker with concurrency=%d", concurrency)
    cmd = [
        "python", "-m", "run.celery_worker", "worker",
        "--loglevel=debug", "--pool=threads", f"--concurrency={concurrency}"
    ]
    logger.info("Running command: %s", " ".join(cmd))
    subprocess.call(cmd, cwd=parent_dir)

if __name__ == '__main__':
    logger.info("Starting both Daphne and Celery worker")
    daphne_thread = threading.Thread(target=start_daphne)
    celery_thread = threading.Thread(target=start_celery_worker)

    daphne_thread.start()
    celery_thread.start()

    daphne_thread.join()
    celery_thread.join()