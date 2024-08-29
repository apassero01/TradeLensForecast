import os
import django

# Set the settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TradeLens.settings')

# Initialize Django
django.setup()

from dataset_manager.services import DatasetManagerService
from sequenceset_manager.services import SequencesetManagerService

tickers = ["AAPL", "SPY", "QQQ", "XOM", "MSFT", "AMZN", "BB"]

for ticker in tickers:
    DatasetManagerService.create_new_stock(ticker)
    SequencesetManagerService.create_new_sequence_set(ticker, "1d", 50)