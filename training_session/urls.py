from django.urls import path
from .views import start_training_session, save_session, remove_session, get_sessions, get_session, \
    get_model_set_strategies, post_strategy, get_viz_processing_strategies, get_training_session_strategies, \
    post_strategy_request

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
]