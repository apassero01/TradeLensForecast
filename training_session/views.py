import json

from django.http import JsonResponse
from django.shortcuts import render
from django.core.cache import cache
from django.views.decorators.csrf import csrf_exempt

from shared_utils.entities.Entity import Entity
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.models import StrategyRequest
from shared_utils.strategy_executor.service.StrategyExecutorService import StrategyExecutorService
from training_session.entities.TrainingSessionEntity import TrainingSessionEntity
from training_session.services.TrainingSessionEntityService import TrainingSessionEntityService
from training_session.models import TrainingSession





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
            session_entity = cache.get('current_session')

            if session_entity and session_entity.id == session_id:
                cache.set('current_session', session_entity)
            else:
                session_model = TrainingSession.objects.get(id=session_id)
                session_entity = TrainingSessionEntity.from_db(session_model)
            # Store in cache
            
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
    print(all_entities)
    
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
        strat_request.nested_requests = strategy['nested_requests']
        strat_request.add_to_history = strategy['add_to_history']
        print(session.serialize())
        try:
            ret_val = training_session_service.execute_strat_request(strat_request, session)
            cache.set('current_session', session)
            print(session.serialize())
            return JsonResponse({
                'status': 'success',
                'message': 'Session loaded successfully',
                'session_data': session.serialize(),
                'strategy_response': ret_val
            })
        except Exception as e:
            print("exception")
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'POST method required'}, status=400)

@csrf_exempt
def api_get_strategy_history(request):
    print('api_get_strategy_history')
    if request.method == 'GET':
        session = cache.get('current_session')
        if not session:
            return JsonResponse({'error': 'No session in progress'}, status=400)

        try:
            strategy_requests = []
            print(session.strategy_history)
            for strategy_request in session.strategy_history:
                strategy_requests.append(strategy_request.serialize())
            
            return JsonResponse({'strategy_requests': strategy_requests})
        except Exception as e:
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'GET method required'}, status=400)