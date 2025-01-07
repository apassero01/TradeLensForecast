from django.contrib import admin
from django.urls import path, include
from . import views


urlpatterns = [
    # path('', views.index, name='index'),
    path('get_sequence_data/', views.get_sequence_data, name='get_seqeunce_data'),
    path('get_sequence_metadata/', views.get_sequence_metadata, name='get_sequence_metadata'),
    path('get_sequence_metadata_by_ids/', views.get_sequence_metadata_by_ids, name='get_sequence_metadata_by_ids'),
    path('get_sequences_by_ids/', views.get_sequences_by_ids, name='get_sequences_by_ids'),
]