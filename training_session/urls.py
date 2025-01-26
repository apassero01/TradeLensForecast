from django.urls import path
from .views import api_start_session, api_stop_session, \
    api_save_session, api_get_saved_sessions, api_load_session, api_get_strategy_registry, get_available_entities, \
    api_execute_strategy, api_get_strategy_history, api_execute_strategy_list

urlpatterns = [

    #### NEW API ENDPOINTS
    path('api/start_session/', api_start_session, name='api_start_session'),
    path('api/stop_session/', api_stop_session, name='api_stop_session'),
    path('api/save_session/', api_save_session, name='api_save_session'),
    path('api/get_saved_sessions/', api_get_saved_sessions, name='api_get_saved_sessions'),
    path('api/load_session/<str:session_id>/', api_load_session, name='api_load_session'),
    path('api/get_strategy_registry/', api_get_strategy_registry, name='api_get_strategy_registry'),
    path('api/get_available_entities/', get_available_entities, name='get_available_entities'),
    path('api/execute_strategy/', api_execute_strategy, name='api_execute_strategy'),
    path('api/get_strategy_history/', api_get_strategy_history, name='api_get_strategy_history'),

    path('api/execute_strategy_list/', api_execute_strategy_list, name='api_execute_strategy_list'),
]
