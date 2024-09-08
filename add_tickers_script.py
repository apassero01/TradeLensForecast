import os
import django

# Set the settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TradeLens.settings')

# Initialize Django
django.setup()

from dataset_manager.services import DatasetManagerService
from sequenceset_manager.services import SequencesetManagerService

tickers = ["AAPL", "SPY", "QQQ", "XOM", "MSFT", "AMZN", "BB"]
sequences_lengths = [10, 20, 50, 75]
# tickers = ["AAPL", 'MSFT']
# sequences_lengths = [10, 20]

for ticker in tickers:
    for sequence_length in sequences_lengths:
        DatasetManagerService.create_new_stock(ticker)
        SequencesetManagerService.create_new_sequence_set(ticker, "1d", sequence_length)