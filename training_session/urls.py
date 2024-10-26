from django.urls import path
from .views import start_training_session

urlpatterns = [
    path('start_training_session/', start_training_session, name='start_training_session'),
]