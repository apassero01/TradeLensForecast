import os
import django

# Set the settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TradeLens.settings')

# Initialize Django
django.setup()

from dataset_manager.models import DataSet
from sequenceset_manager.models import SequenceSet
from dataset_manager.services import StockDataSetService, DataSetService
from sequenceset_manager.services import StockSequenceSetService, SequenceSetService

# tickers = ["AAPL", "SPY", "QQQ", "XOM", "MSFT", "AMZN", "BB", 'F', 'TSLA', 'GE',
#            "JPM", "BAC", "LLY", "NKE", "WMT", "PEP", 'M', 'KODK',
#            'BA', 'DIS']
# sequences_lengths = [10, 20, 50, 75]
tickers = ["MSFT", "SPY", "QQQ", "TSLA"]
sequences_lengths = [20, 50]

for ticker in tickers:
    for sequence_length in sequences_lengths:
        dataset = DataSetService.get_data_set(dataset_type="stock", ticker=ticker, interval="1d")
        StockDataSetService.update_recent_data(dataset)

        sequences_set = SequenceSetService.get_sequence_set(sequence_length = sequence_length, dataset_type='stock', ticker=ticker, interval='1d').get()
        StockSequenceSetService.update_recent(sequences_set, ticker =  ticker, interval= '1d')
        print(f"Updated {ticker} with sequence length {sequence_length}")

if __name__ == "__main__":
    pass