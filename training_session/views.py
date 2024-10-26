import json

from django.http import JsonResponse
from django.shortcuts import render
from django.core.cache import cache
from django.views.decorators.csrf import csrf_exempt


# Create your views here.
@csrf_exempt
def start_training_session(request):
    print('start_training_session')
    if request.method == 'POST':
        if cache.get('current_session'):
            return JsonResponse({'error': 'A session is already in progress'}, status=400)

        training_session_service = StockTrainingSessionService()

        sequence_params = request.POST.get('sequence_params')  # This will be a JSON string
        start_timestamp = request.POST.get('start_timestamp')
        X_features = request.POST.get('X_features')  # This will be a JSON string
        y_features = request.POST.get('y_features')  # This will be a JSON string

        # Parse the JSON strings if necessary
        sequence_params = json.loads(sequence_params) if sequence_params else []
        X_features = json.loads(X_features) if X_features else []
        y_features = json.loads(y_features) if y_features else []

        print('X_features:', X_features)
        X_features = [feature['name']for feature in X_features]
        y_features = [feature['name']for feature in y_features]
        y_features = sorted(y_features, key = lambda x: int(x.split("+")[1]))

        # Now you have your sequence_params, X_features, and y_features as Python lists/dicts
        print("Sequence Params:", sequence_params)
        print("Start Timestamp:", start_timestamp)
        print("X Features:", X_features)
        print("Y Features:", y_features)


        for param in sequence_params:
            param['start_timestamp'] = start_timestamp


        if not X_features or not y_features or not sequence_params:
            print('X_features, y_features, and sequence_params are required')
            return JsonResponse({'error': 'X_features, y_features, and sequence_params are required'}, status=400)

        session = None
        try:
            session = training_session_service.create_training_session(
                X_features=X_features, y_features=y_features, sequence_params=sequence_params
            )


            print('session created')
            print("Sequence params: ", sequence_params)
            print("X_features: ", X_features)
            print("y_features: ", y_features)

            training_session_service.retrieve_sequence_sets(session, sequence_params)

            cache.set('current_session', session)

            return JsonResponse({
                'status': 'success',
                'session_id': session.id,
                'created_at': session.created_at,
                'status': session.status,
            })
        except Exception as e:
            if session is not None:
                session.delete()
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'POST method required'}, status=400)