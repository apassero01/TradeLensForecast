import json
from tkinter.constants import ACTIVE

from django.http import JsonResponse
from django.shortcuts import render
from django.core.cache import cache
from django.views.decorators.csrf import csrf_exempt
from html5lib.treewalkers.base import ENTITY

from shared_utils.entities.Entity import Entity
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.models import StrategyRequest
from shared_utils.strategy_executor.service.StrategyExecutorService import StrategyExecutorService
from training_session.VizProcessingStrategies import HistVizProcessingStrategy
from training_session.entities.TrainingSessionEntity import TrainingSessionEntity
from training_session.services.TrainingSessionEntityService import TrainingSessionEntityService
from training_session.services.TrainingSessionService import TrainingSessionService, TrainingSessionStatus
from training_session.services.StrategyService import ModelSetStrategyService, VizProcessingStrategyService
from training_session.strategy.services.TrainingSessionStrategyService import TrainingSessionStrategyService
from training_session.models import TrainingSession


# Create your views here.
@csrf_exempt
def start_training_session(request):
    print('start_training_session')
    if request.method == 'POST':
        if cache.get('current_session'):
            print('A session is already in progress')
            return JsonResponse({'error': 'A session is already in progress'}, status=400)

        training_session_service = TrainingSessionEntityService()

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
            session = training_session_service.create_training_session_entity()

            print('session created')
            print("Sequence params: ", sequence_params)
            print("X_features: ", X_features)
            print("y_features: ", y_features)

            session = training_session_service.initialize_params(X_features, y_features, sequence_params, start_timestamp)

            cache.set('current_session', session)

            return JsonResponse({
                'status': 'success',
                'sessionData': training_session_service.serialize_session()
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

@csrf_exempt
def post_strategy_request(request):

    print('post_strategy_request')
    if request.method == 'POST':
        session_entity = cache.get('current_session')

        if not isinstance(session_entity, Entity):
            session_entity = TrainingSessionEntity.from_db(session_entity)

        if not session_entity:
            print('No session in progress')
            return JsonResponse({'error': 'No session in progress'}, status=400)

        strategy_request_config = json.loads(request.body)["config"]

        if not strategy_request_config:
            print('strategy is required')
            return JsonResponse({'error': 'strategy is required'}, status=400)

        training_session_service = TrainingSessionEntityService()
        training_session_service.set_session(session_entity)

        strategy_name = strategy_request_config['strategy_name']
        strategy_path = strategy_request_config['strategy_path']
        param_config = strategy_request_config['param_config']
        strategy = StrategyRequestEntity()
        strategy.strategy_name = strategy_name
        strategy.strategy_path = strategy_path
        strategy.param_config = param_config
        if strategy_name == 'GetSequenceSetsStrategy':
            strategy.param_config['model_set_configs'] = session_entity.sequence_set_params
            strategy.param_config['X_features'] = session_entity.X_features
            strategy.param_config['y_features'] = session_entity.y_features


        try:
            ret_val = training_session_service.execute_strat_request(strategy, session_entity)
            for key, value in session_entity.get_entity_map().items():
                print(key, value)
            session_state = training_session_service.serialize_session()
            cache.set('current_session', session_entity)
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
def get_training_session_strategies(request):
    print('get training session strategies')
    if request.method == 'GET':
        strategy_service = TrainingSessionStrategyService
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

@csrf_exempt
def get_strategy_registry(request):
    print('get strategy registry')
    if request.method == 'GET':
        try:
            strategies = StrategyExecutorService.get_registry()
            return JsonResponse(strategies, safe=False)
        except Exception as e:
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'GET method required'}, status=400)
    




### Entity Graph APIs

@csrf_exempt
def get_entity_graph(request):
    print('get_entity_graph')
    if request.method == 'GET':
        session_entity = cache.get('current_session')
        
        if not isinstance(session_entity, Entity):
            session_entity = TrainingSessionEntity.from_db(session_entity)

        if not session_entity:
            print('No session in progress')
            return JsonResponse({'error': 'No session in progress'}, status=400)

        try:
            # Use the entity's serialize method to get the hierarchical structure
            graph_data = session_entity.serialize()
            return JsonResponse(graph_data)
        except Exception as e:
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'GET method required'}, status=400)

### New API Endpoints ###

@csrf_exempt
def api_start_session(request):
    print('api_start_session')
    if request.method == 'POST':
        if cache.get('current_session'):
            print('A session is already in progress')
            return JsonResponse({'error': 'A session is already in progress'}, status=400)

        training_session_service = TrainingSessionEntityService()

        try:
            # Create a new session with minimal initialization
            session = training_session_service.create_training_session_entity()
            cache.set('current_session', session)

            return JsonResponse({
                'status': 'success',
                'sessionData': training_session_service.serialize_session()
            })
        except Exception as e:
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'POST method required'}, status=400)

@csrf_exempt
def api_get_entity_graph(request):
    print('api_get_entity_graph')
    if request.method == 'GET':
        session_entity = cache.get('current_session')
        
        if not isinstance(session_entity, Entity):
            session_entity = TrainingSessionEntity.from_db(session_entity)

        if not session_entity:
            print('No session in progress')
            return JsonResponse({'error': 'No session in progress'}, status=400)

        try:
            # Use the entity's serialize method to get the hierarchical structure
            graph_data = session_entity.serialize()
            print(graph_data)
            return JsonResponse(graph_data)
        except Exception as e:
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'GET method required'}, status=400)

@csrf_exempt
def api_stop_session(request):
    print('api_stop_session')
    if request.method == 'POST':
        session = cache.get('current_session')
        if not session:
            return JsonResponse({'error': 'No session in progress'}, status=400)
        
        try:
            cache.delete('current_session')
            return JsonResponse({
                'status': 'success',
                'message': 'Session stopped successfully'
            })
        except Exception as e:
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'POST method required'}, status=400)
    

@csrf_exempt
def api_save_session(request):
    print('api_save_session')
    if request.method == 'POST':
        session_entity = cache.get('current_session')
        if not session_entity:
            return JsonResponse({'error': 'No session in progress'}, status=400)
        
        try:
            # Convert and save to database
            training_session_service = TrainingSessionEntityService()
            training_session_service.set_session(session_entity)
            session_id = training_session_service.save_session()
            cache.set('current_session', session_entity)
            
            return JsonResponse({
                'status': 'success',
                'session_id': session_id,
                'message': 'Session saved successfully'
            })
        except Exception as e:
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'POST method required'}, status=400)

@csrf_exempt
def api_get_saved_sessions(request):
    print('api_get_saved_sessions')
    if request.method == 'GET':
        try:
            sessions = TrainingSession.objects.all().values('id', 'created_at')
            return JsonResponse({
                'sessions': list(sessions)
            })
        except Exception as e:
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'GET method required'}, status=400)
    
@csrf_exempt
def api_load_session(request, session_id):
    print(f'api_load_session: {session_id}')
    if request.method == 'POST':
        try:
            # Load the session from database
            session_model = TrainingSession.objects.get(id=session_id)
            session_entity = TrainingSessionEntity.from_db(session_model)
            
            # Store in cache
            cache.set('current_session', session_entity)
            
            return JsonResponse({
                'status': 'success',
                'message': 'Session loaded successfully',
                'session_data': session_entity.serialize()
            })
        except TrainingSession.DoesNotExist:
            return JsonResponse({'error': f'Session {session_id} not found'}, status=404)
        except Exception as e:
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'POST method required'}, status=400)
    
@csrf_exempt
def api_get_strategy_registry(request):
    print('api_get_strategy_registry')
    if request.method == 'GET':
        try:
            strategies = StrategyExecutorService.get_registry()
            return JsonResponse(strategies, safe=False)
        except Exception as e:
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'GET method required'}, status=400)

def get_available_entities(request):
    entities = {}
    
    # Get all subclasses of Entity
    def get_all_entity_subclasses(cls):
        all_subclasses = []
        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(get_all_entity_subclasses(subclass))
        return all_subclasses
    
    all_entities = get_all_entity_subclasses(Entity)
    
    # Format the response
    for entity_class in all_entities:
        entities[entity_class.entity_name.value] = {
            'name': entity_class.entity_name.value,
            'class_path': entity_class.get_class_path()
        }
    
    return JsonResponse({'entities': entities})

@csrf_exempt
def api_execute_strategy(request):
    print('execute_strategy')
    if request.method == 'POST':
        session = cache.get('current_session')
        if not session:
            print('No session in progress')
            return JsonResponse({'error': 'No session in progress'}, status=400)

        print(json.loads(request.body))
        strategy = json.loads(request.body).get('strategy')
        print('strategy', strategy)

        if not strategy:
            print('strategy is required')
            return JsonResponse({'error': 'strategy is required'}, status=400)

        training_session_service = TrainingSessionEntityService()
        training_session_service.set_session(session)

        strat_request = StrategyRequestEntity()
        strat_request.strategy_name = strategy['strategy_name']
        strat_request.strategy_path = strategy['strategy_path']
        strat_request.param_config = strategy['param_config']
        print(session.serialize())
        try:
            ret_val = training_session_service.execute_strat_request(strat_request, session)
            cache.set('current_session', session)
            print(session.serialize())
            return JsonResponse({
                'status': 'success',
                'message': 'Session loaded successfully',
                'session_data': session.serialize()
            })
        except Exception as e:
            print("exception")
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'POST method required'}, status=400)