from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from sequenceset_manager.services import SequencesetManagerService
from datetime import datetime
from django.utils.dateparse import parse_date
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.http import require_http_methods


@require_http_methods(["GET"])
def get_sequence_data(request):
    ticker = request.GET.get('ticker')  # Use getlist to handle multiple tickers
    features = request.GET.getlist('features')

    # Get optional parameters with default values
    start_date = request.GET.get('start_date', None)
    end_date = request.GET.get('end_date', None)
    interval = request.GET.get('interval', None)
    sequence_length = request.GET.get('sequence_length', None)

    # Convert start_date and end_date to Python date objects, if provided
    if start_date:
        start_date = parse_date(start_date)
    if end_date:
        end_date = parse_date(end_date)

    # Ensure 'tickers' and 'features' are provided, as they are mandatory
    if not ticker or not features:
        print("Ticker and features are required parameters."),

        return JsonResponse(
            {"error": "Ticker and features are required parameters."},
            status=400
        )

    try:
        # Assuming you want to retrieve data for multiple tickers
        results = []
        result = SequencesetManagerService.retrieve_sequence_slice(
            ticker=ticker.upper(),  # Convert to uppercase
            features=features,
            interval=interval,  # Can be None
            start_date=start_date,  # Can be None
            end_date=end_date,  # Can be None
            sequence_length=sequence_length  # Can be None
        )
        results.extend(result)  # Aggregate results from multiple tickers

    except Exception as e:
        print(f"An error occurred: {e}")
        return JsonResponse(
            {"error": str(e)},
            status=500
        )

    return JsonResponse(results, safe=False)
