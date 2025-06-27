import multiprocessing
import signal

multiprocessing.set_start_method('spawn', force=True)

import os
from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TradeLens.settings')

# Create the Celery app instance.
app = Celery('TradeLens')

# Load any custom configuration from your Django settings, using a CELERY namespace.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Auto-discover tasks in all installed Django apps.
app.autodiscover_tasks([
    "shared_utils",                # still loads shared_utils.tasks
    "shared_utils.embedding_tasks" # 👍  explicitly load embedding file
])

CELERY_IMPORTS = ("shared_utils.embedding_tasks",)

def worker_process_init(sender=None, **kwargs):
    # Handle shutdown when receiving termination signals
    signal.signal(signal.SIGTERM, worker_process_shutdown)
    signal.signal(signal.SIGINT, worker_process_shutdown)

def worker_process_shutdown(signal, frame):
    # Perform any cleanup needed``
    worker = app.Worker()
    worker.stop()

# Register the init handler
from celery.signals import worker_process_init as worker_init_signal
worker_init_signal.connect(worker_process_init)

# Example task (optional):
@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')