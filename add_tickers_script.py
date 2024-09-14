import os
import django

# Set the settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TradeLens.settings')

# Initialize Django
django.setup()

from dataset_manager.services import StockDataSetService
from sequenceset_manager.services import StockSequenceSetService

# tickers = ["AAPL", "SPY", "QQQ", "XOM", "MSFT", "AMZN", "BB"]
# sequences_lengths = [10, 20, 50, 75]
tickers = ["AAPL"]
sequences_lengths = [50]

for ticker in tickers:
    for sequence_length in sequences_lengths:
        StockDataSetService.create_new_dataset(dataset_type="stock", start_date=None, end_date=None, ticker=ticker, interval="5m")
        StockSequenceSetService.create_sequence_set(sequence_length, dataset_type='stock', ticker=ticker, interval='5m')