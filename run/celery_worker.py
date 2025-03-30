#!/usr/bin/env python
import multiprocessing
import sys

multiprocessing.set_start_method('spawn', force=True)

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TradeLens.settings')

from celery import Celery

app = Celery('TradeLens')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

if __name__ == '__main__':
    app.start(sys.argv[1:])  # Pass arguments excluding the script path