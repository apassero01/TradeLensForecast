import json

from django.core.serializers.json import DjangoJSONEncoder
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from sequenceset_manager.models import SequenceSet
from sequenceset_manager.services import SequenceSetService, StockSequenceSetService, SequenceService
from datetime import datetime
from django.utils.dateparse import parse_date
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.http import require_http_methods
from logging import getLogger

logger = getLogger(__name__)

@csrf_exempt
@require_http_methods(["POST"])
def get_sequence_data(request):
    # Attempt to decode the JSON payload
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON payload"}, status=400)

    # Extract parameters from the JSON data
    ticker = data.get('ticker')
    features = data.get('features', [])
    start_date = data.get('start_timestamp')
    end_date = data.get('end_date')
    interval = data.get('interval')
    sequence_length = data.get('sequence_length')

    # Convert date strings to Python date objects if provided
    if start_date:
        start_date = parse_date(start_date)
    if end_date:
        end_date = parse_date(end_date)

    # Validate required parameters
    if not ticker or not features:
        return JsonResponse(
            {"error": "Ticker and features are required parameters."},
            status=400
        )

    try:
        # Generate the result using your service
        result = SequenceSetService.retrieve_sequence_slice(
            sequence_length=sequence_length,
            feature_list=features,
            start_date=start_date,
            end_date=end_date,
            ticker=ticker,
            interval=interval,
            dataset_type="stock",
        )
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return JsonResponse({"error": str(e)}, status=500)

    # Check if the result is empty
    if not result:
        return JsonResponse({"error": "No data available"}, status=204)

    # Define a generator that streams the JSON response in chunks
    def json_generator(data, chunk_size=1024 * 1024):  # 1 MB per chunk
        json_data = json.dumps(data, cls=DjangoJSONEncoder)
        # Log the first 500 characters for debugging purposes
        logger.debug(f"Result data (first 500 chars): {json_data[:500]}")
        for i in range(0, len(json_data), chunk_size):
            yield json_data[i:i + chunk_size]

    return StreamingHttpResponse(json_generator(result), content_type="application/json")

@require_http_methods(["GET"])
def get_sequence_metadata(request):
    response = StockSequenceSetService.get_all_sequence_set_metadata()
    return JsonResponse(response, safe=False)

@require_http_methods(["GET"])
def get_sequence_metadata_by_ids(request):
    sequence_ids = request.GET.getlist('ids')
    response = SequenceService.get_sequence_metadata_by_ids(sequence_ids)
    return JsonResponse(response, safe=False)

@require_http_methods(["POST"])
@csrf_exempt
def get_sequences_by_ids(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    sequence_ids = data.get('sequence_ids', [])
    feature_list = data.get('feature_list', [])

    if not sequence_ids or not feature_list:
        return JsonResponse({"error": "sequence_ids and feature_list are required"}, status=400)

    # Convert sequence_ids to integers if needed
    try:
        sequence_ids = [int(seq_id) for seq_id in sequence_ids]
    except ValueError:
        return JsonResponse({"error": "All sequence_ids must be integers."}, status=400)

    # Call your service to get the sequences
    result = SequenceService.get_sequences(sequence_ids, feature_list)

    # If no sequences found, return an appropriate response
    if not result:
        return JsonResponse({"error": "No sequences found."}, status=204)

    # Define a generator to stream the JSON response in chunks
    def json_generator(data, chunk_size=1024 * 1024):  # 1 MB per chunk
        json_data = json.dumps(data, cls=DjangoJSONEncoder)
        for i in range(0, len(json_data), chunk_size):
            yield json_data[i:i + chunk_size]

    return StreamingHttpResponse(json_generator(result), content_type="application/json")

