import json
from tkinter.constants import ACTIVE

from django.http import JsonResponse
from django.shortcuts import render
from django.core.cache import cache
from django.views.decorators.csrf import csrf_exempt

from training_session.VizProcessingStrategies import HistVizProcessingStrategy
from training_session.services.TrainingSessionService import TrainingSessionService, TrainingSessionStatus
from training_session.services.StrategyService import ModelSetStrategyService, VizProcessingStrategyService


# Create your views here.
@csrf_exempt
def start_training_session(request):
    print('start_training_session')
    if request.method == 'POST':
        if cache.get('current_session'):
            print('A session is already in progress')
            return JsonResponse({'error': 'A session is already in progress'}, status=400)

        training_session_service = TrainingSessionService()

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
            session = training_session_service.create_training_session()

            print('session created')
            print("Sequence params: ", sequence_params)
            print("X_features: ", X_features)
            print("y_features: ", y_features)

            session = training_session_service.initialize_params(session, X_features, y_features, sequence_params, start_timestamp)

            cache.set('current_session', session)

            return JsonResponse({
                'status': 'success',
                'sessionData': training_session_service.serialize_session(session)
            })
        except Exception as e:
            if session is not None:
                session.delete()
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'POST method required'}, status=400)

@csrf_exempt
def get_sessions(request):
    print('get_sessions')
    if request.method == 'GET':
        training_session_service = TrainingSessionService()
        try:
            sessions = training_session_service.get_sessions()
        except Exception as e:
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
        return JsonResponse(sessions, safe=False)
    else:
        return JsonResponse({'error': 'GET method required'}, status=400)

@csrf_exempt
def get_session(request, session_id):
    print('get_session')
    if request.method == 'GET':
        training_session_service = TrainingSessionService()
        session = training_session_service.get_session(session_id)
        if cache.get('current_session'):
            print('A session is already in progress')
            return JsonResponse({'error': 'A session is already in progress'}, status=400)
        session.status = TrainingSessionStatus.ACTIVE.value
        cache.set('current_session', session)
        return JsonResponse(
            training_session_service.serialize_session(session), safe=False
        )
    else:
        return JsonResponse({'error': 'GET method required'}, status=400)

@csrf_exempt
def save_session(request):
    print('save_session')
    if request.method == 'POST':
        session = cache.get('current_session')

        if not session:
            print('No session in progress')
            return JsonResponse({'error': 'No session in progress'}, status=400)

        training_session_service = TrainingSessionService()
        training_session_service.print_session(session)
        try:
            training_session_service.save_session(session)
            return JsonResponse({'status': 'success'})
        except Exception as e:
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'POST method required'}, status=400)

@csrf_exempt
def remove_session(request):
    print('remove_session')
    if request.method == 'POST':
        session = cache.get('current_session')
        if not session:
            print('No session in progress')
            return JsonResponse({'error': 'No session in progress'}, status=400)

        cache.delete('current_session')
        return JsonResponse({'status': 'success'})

    else:
        return JsonResponse({'error': 'POST method required'}, status=400)

@csrf_exempt
def post_strategy(request):
    print('post_strategy')
    if request.method == 'POST':
        session = cache.get('current_session')
        if not session:
            print('No session in progress')
            return JsonResponse({'error': 'No session in progress'}, status=400)

        strategy = json.loads(request.body)


        if not strategy:
            print('strategy is required')
            return JsonResponse({'error': 'strategy is required'}, status=400)

        training_session_service = TrainingSessionService()
        try:
            ret_val = training_session_service.apply_strategy(session, strategy)
            print("Session ordered model set strategies: ", session.ordered_model_set_strategies)
            session_state = training_session_service.serialize_session(session)
            cache.set('current_session', session)
            return JsonResponse({'status': 'success', 'sessionData': session_state, 'ret_val': ret_val})
        except Exception as e:
            print("exception")
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'POST method required'}, status=400)


## Stategy APIs
@csrf_exempt
def get_model_set_strategies(request):
    print('get model set strategies')
    if request.method == 'GET':
        strategy_service = ModelSetStrategyService()
        try:
            strategies = strategy_service.get_available_strategies()
            return JsonResponse(strategies, safe=False)
        except Exception as e:
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'GET method required'}, status=400)

@csrf_exempt
def get_viz_processing_strategies(request):
    print('get viz strategies')
    if request.method == 'GET':
        strategy_service = VizProcessingStrategyService()
        try:
            strategies = strategy_service.get_available_strategies()
            return JsonResponse(strategies, safe=False)
        except Exception as e:
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'GET method required'}, status=400)