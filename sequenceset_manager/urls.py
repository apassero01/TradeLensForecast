from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    # path('', views.index, name='index'),
    path('get_sequence_data/', views.get_sequence_data, name='get_seqeunce_data'),
]