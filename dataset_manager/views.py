from django.shortcuts import render
from dataset_manager.services import DatasetManagerService

import json
from django.http import JsonResponse
from dataset_manager.models import StockData
from django.views.decorators.http import require_http_methods


# Create your views here.

@require_http_methods(["GET"])
def get_stock_data(request, ticker):
    """
    Retrieve stock data from the database and return it as a JSON response
    """
    ticker = ticker.upper()

    # Get optional parameters with default values
    interval = request.GET.get('interval', '1d')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    # Fetch the stock data using the DatasetManagerService
    df = DatasetManagerService.stockdata_to_dataframe(ticker, interval, start_date, end_date)

    # Convert the index to string for JSON serialization
    df.index = df.index.astype(str)
    # Return the DataFrame as a JSON response
    return JsonResponse(df.to_dict(orient='index'))

def create_stock_data(request):
    '''
    Create stock data and save it to the database
    '''
    if request.method == 'POST':
        data = json.loads(request.body)
        ticker = data['ticker']
        interval = data.get('interval', '1d')
        start_date = data['start_date']
        end_date = data.get('end_date', None)

        # check for ticker in database 
        all_tickers = StockData.objects.values_list('ticker', flat=True)
        if ticker in all_tickers:
            return JsonResponse({'message': 'Data already exists for this ticker'}, status=400)

        df = DatasetManagerService.create_new_stock(ticker, start_date, end_date, interval)
        df.replace({-999: None}, inplace=True)

        df.index = df.index.astype(str)

        return JsonResponse(df.to_dict(orient='index'))
    

