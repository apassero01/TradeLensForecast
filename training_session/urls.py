from django.urls import path
from .views import start_training_session, save_session, remove_session, get_sessions, get_session, \
    get_model_set_strategies, post_strategy, get_viz_processing_strategies, get_training_session_strategies, \
    post_strategy_request, get_strategy_registry, api_start_session, api_get_entity_graph, api_stop_session, \
    api_save_session, api_get_saved_sessions, api_load_session, api_get_strategy_registry, get_available_entities, \
    api_execute_strategy

urlpatterns = [
    path('start_training_session/', start_training_session, name='start_training_session'),
    path('save_session/', save_session, name='save_session'),
    path('remove_session/', remove_session, name='remove_session'),
    path('get_sessions/', get_sessions, name='get_sessions'),
    path('get_session/<int:session_id>/', get_session, name='get_session'),
    path('post_strategy', post_strategy, name='post_strategy'),
    path('post_strategy_request', post_strategy_request, name='post_strategy_request'),


    # strategy endpoints
    path('get_model_set_strategies/', get_model_set_strategies, name='get_model_set_strategies'),
    path('get_viz_processing_strategies/', get_viz_processing_strategies, name='get_viz_processing_strategies'),
    path('get_training_session_strategies', get_training_session_strategies, name='get_training_session_strategies'),

    # new strategy end points
    path('get_strategy_registry/', get_strategy_registry, name='get_strategy_registry'),

    #### NEW API ENDPOINTS
    path('api/start_session/', api_start_session, name='api_start_session'),
    path('api/get_entity_graph/', api_get_entity_graph, name='api_get_entity_graph'),
    path('api/stop_session/', api_stop_session, name='api_stop_session'),
    path('api/save_session/', api_save_session, name='api_save_session'),
    path('api/get_saved_sessions/', api_get_saved_sessions, name='api_get_saved_sessions'),
    path('api/load_session/<int:session_id>/', api_load_session, name='api_load_session'),
    path('api/get_strategy_registry/', api_get_strategy_registry, name='api_get_strategy_registry'),
    path('api/get_available_entities/', get_available_entities, name='get_available_entities'),
    path('api/execute_strategy/', api_execute_strategy, name='api_execute_strategy'),
]
