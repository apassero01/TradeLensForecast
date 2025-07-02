import importlib
import json
from copy import deepcopy

from django.http import JsonResponse
from django.shortcuts import render
from django.core.cache import cache
from django.views.decorators.csrf import csrf_exempt

from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.Entity import Entity
from shared_utils.entities.EntityModel import EntityModel
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.models import StrategyRequest
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from training_session.entities.TrainingSessionEntity import TrainingSessionEntity
from training_session.services.TrainingSessionEntityService import TrainingSessionEntityService
from training_session.models import TrainingSession
from shared_utils.strategy_executor.service.StrategyExecutorService import StrategyExecutorService
from shared_utils.cache.CacheService import CacheService
from shared_utils.entities.service.EntityService import EntityService
from shared_utils.entities import discover_entities

### New API Endpoints ###

@csrf_exempt
def api_start_session(request):
    print('api_start_session')
    if request.method == 'POST':
        entity_service = EntityService()

        try:
            # Create a new session with minimal initialization
            session = TrainingSessionEntity()
            session_db = session.to_db()
            session_db.save()
            # Save session to cache and track current session ID
            entity_service.save_entity(session)
            entity_service.set_session_id(session.entity_id)

            return JsonResponse({
                'status': 'success',
                'sessionData': {
                    session.entity_id : session.serialize()
                }  
            })
        except Exception as e:
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'POST method required'}, status=400)

@csrf_exempt
def api_stop_session(request):
    print('api_stop_session')
    if request.method == 'POST':
        entity_service = EntityService()
        current_session_id = entity_service.get_session_id()
        
        if not current_session_id:
            return JsonResponse({'error': 'No session in progress'}, status=400)
        
        try:
            # Clear all entities from cache
            entity_service.clear_all_entities()
            
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
        entity_service = EntityService()
        session_entity_id = entity_service.get_session_id()
        if not session_entity_id:
            return JsonResponse({'error': 'No session in progress'}, status=400)
        
        try:
            # Convert and save to database
            session_entity = entity_service.get_entity(session_entity_id)

            children = deepcopy(session_entity.get_children())
            session_model = session_entity.to_db()
            traversed_children = []
            while children:
                child_id = children.pop()
                if child_id in traversed_children:
                    continue
                try:
                    child  = entity_service.get_entity(child_id)
                    entity_service.save_entity(child)
                except Exception as e:
                    continue
                children.extend(child.get_children())
                traversed_children.append(child_id)

            
            return JsonResponse({
                'status': 'success',
                'session_id': session_model.entity_id,
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
            sessions = EntityModel.objects.filter(entity_type=EntityEnum.TRAINING_SESSION.value).values('entity_id', 'created_at')
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
    entity_service = EntityService()
    if request.method == 'POST':
        try:
            # Load the session from database
            entity_id = str(EntityModel.objects.get(entity_id=session_id).entity_id)
            cur_session_id = entity_service.get_session_id()

            if cur_session_id != entity_id:
                session_model = EntityModel.objects.get(entity_id=session_id)
                session_entity = TrainingSessionEntity.from_db(session_model)
                entity_service.cache_service.clear_all()
                cache.set('current_session_id', session_entity.entity_id)

            serialized = serialize_entity_and_children(entity_id)
            # Store in cache
            return JsonResponse({
                'status': 'success',
                'message': 'Session loaded successfully',
                'session_data': serialized
            })
        except TrainingSession.DoesNotExist:
            return JsonResponse({'error': f'Session {session_id} not found'}, status=404)
        except Exception as e:
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'POST method required'}, status=400)

@csrf_exempt
def api_delete_session(request):
    print('api_delete_session')
    if request.method == 'POST':
        entity_service = EntityService()
        session_id = entity_service.get_session_id()
        if not session_id:
            return JsonResponse({'error': 'No session in progress'}, status=400)

        try:
            entity_service.delete_session()
            return JsonResponse({
                'status': 'success',
                'message': 'Session deleted successfully'
            })
        except Exception as e:
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'POST method required'}, status=400)

def create_instance_from_path(class_path, *args, **kwargs):
    """
    Dynamically imports and instantiates a class from its class path.

    :param class_path: The dot-separated path to the class (e.g., 'my_module.MyClass').
    :param args: Positional arguments to pass to the constructor.
    :param kwargs: Keyword arguments to pass to the constructor.
    :return: An instance of the class.
    """
    # Split the path into module and class
    module_name, class_name = class_path.rsplit('.', 1)

    # Import the module
    module = importlib.import_module(module_name)

    # Get the class from the module
    cls = getattr(module, class_name)

    # Call the constructor and return the instance
    return cls
    
@csrf_exempt
def api_get_strategy_registry(request):
    print('api_get_strategy_registry')
    if request.method == 'GET':
        try:
            executor_service = StrategyExecutorService()
            strategies = executor_service.get_registry()
            return JsonResponse(strategies, safe=False)
        except Exception as e:
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'GET method required'}, status=400)

def get_available_entities(request):
    entities = {}
    discover_entities()
    
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
        entity_service = EntityService()
        session_id = entity_service.get_session_id()
        if not session_id:
            print('No session in progress')
            return JsonResponse({'error': 'No session in progress'}, status=400)
    
        print(json.loads(request.body))
        strategy = json.loads(request.body).get('strategy')
        print('strategy', strategy)

        if not strategy:
            print('strategy is required')
            return JsonResponse({'error': 'strategy is required'}, status=400)

        strat_request = json_to_StrategyRequestEntity(strategy)

        strategy_executor_service = StrategyExecutorService()

        try:
            ret_val = strategy_executor_service.execute_request(strat_request)

            return JsonResponse({
                'status': 'success',
                'message': 'Session loaded successfully',
                'entities': get_updated_entities(ret_val)
                # 'strategy_response': ret_val
            })
        except Exception as e:
            print("exception")
            print(str(e))
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'POST method required'}, status=400)
    
@csrf_exempt
def api_execute_strategy_list(request):
    print('execute_strategy_list')
    if request.method == 'POST':
        session = cache.get('current_session')
        if not session:
            return JsonResponse({'error': 'No session in progress'}, status=400)

        print(json.loads(request.body))
        strategies = json.loads(request.body).get('strategy_list')
        print('strategies', strategies)

        if not strategies:
            return JsonResponse({'error': 'strategies are required'}, status=400)

        training_session_service = TrainingSessionEntityService()
        training_session_service.set_session(session)

        strategy_executor_service = StrategyExecutorService()
        for strategy in strategies:
            try:
                strat_request = json_to_StrategyRequestEntity(strategy)
                ret_val = strategy_executor_service.execute_by_target_entity_id(session, strat_request)
                add_to_strategy_history(session, ret_val)


            except Exception as e:
                print(str(e))
                cache.set('current_session', session)
                return JsonResponse({'error': str(e)}, status=400)

        cache.set('current_session', session)
        return JsonResponse({
            'status': 'success',
            'message': 'Session loaded successfully',
            'session_data': session.serialize()
        })
    else:
        return JsonResponse({'error': 'POST method required'}, status=400)

# @csrf_exempt
def api_get_strategy_history(request):
    pass
#     print('api_get_strategy_history')
#     if request.method == 'GET':
#         entity_service = EntityService()
#         session_id = entity_service.get_session_id()
#         if not session_id:
#             return JsonResponse({'error': 'No session in progress'}, status=400)
#
#         try:
#             session = entity_service.get_entity(session_id)
#             strategy_requests = []
#             print(session.strategy_history)
#             for strategy_request in session.strategy_history:
#                 strategy_requests.append(strategy_request.serialize())
#
#             return JsonResponse({'strategy_requests': strategy_requests})
#         except Exception as e:
#             print(str(e))
#             return JsonResponse({'error': str(e)}, status=400)
#     else:
#         return JsonResponse({'error': 'GET method required'}, status=400)


# Helper Functions

def json_to_StrategyRequestEntity(json_data):
    if 'entity_id' in json_data:
        strat_request = StrategyRequestEntity(json_data['entity_id'])
    else:
        strat_request = StrategyRequestEntity()

    strat_request.strategy_name = json_data['strategy_name']
    strat_request.param_config = json_data['param_config']
    strat_request.add_to_history = json_data['add_to_history']
    strat_request.target_entity_id = json_data['target_entity_id']
    strat_request.add_to_history = json_data['add_to_history']

    nested_requests = json_data['nested_requests']
    for nested_request in nested_requests:
        strat_request.add_nested_request(json_to_StrategyRequestEntity(nested_request))

    return strat_request

def add_to_strategy_history(session_id, strat_request):
    #TODO later the history will be stored in the entity that the strategy is executed on for now without db we store it in the session
    entity_service = EntityService()
    session_entity = entity_service.get_entity(session_id)
    for i, strategy_request in enumerate(session_entity.strategy_history):
        if strategy_request.entity_id == strat_request.entity_id:
            session_entity.strategy_history[i] = strat_request
            entity_service.save_entity(session_entity)
            return
    session_entity.add_to_strategy_history(strat_request)
    entity_service.save_entity(session_entity)
    return

def get_strategy_tree(strat_request):
    result = [strat_request]
    for nested_request in strat_request.get_nested_requests():
        result.extend(get_strategy_tree(nested_request))
    return result

def get_updated_entities(strat_request):
    all_requests = get_strategy_tree(strat_request)
    all_entities = set([request.target_entity_id for request in all_requests])
    entity_service = EntityService()
    entities_serialized = {}
    for entity_id in all_entities:
        entity = entity_service.get_entity(entity_id)
        if entity:
            if hasattr(entity, 'deleted') and entity.deleted:
                entities_serialized[entity_id] = {'deleted': True}
                entity_service.clear_entity(entity_id)
            else:
                entities_serialized[entity_id] = entity.serialize()

    return entities_serialized

def serialize_entity_and_children(entity_id, return_dict = None):
    entity_service = EntityService()
    try:
        entity = entity_service.get_entity(entity_id)
    except Exception as e:
        return return_dict
    if not entity:
        return None
    entity_dict = entity.serialize()
    children = entity.get_children()
    if return_dict is None:
        return_dict = {}

    if entity_id in return_dict:
        return return_dict
    return_dict[entity_id] = entity_dict

    for child_id in children:
        return_dict = serialize_entity_and_children(child_id, return_dict)

    return return_dict
