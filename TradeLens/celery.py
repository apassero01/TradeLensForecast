import multiprocessing

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
app.autodiscover_tasks()

# Example task (optional):
@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')