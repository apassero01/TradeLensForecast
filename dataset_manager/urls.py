from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    # path('', views.index, name='index'),
    path('get_stock_data/<str:ticker>/', views.get_stock_data, name='get_stock_data'),
    path('create_stock_data/', views.create_stock_data, name='create_stock_data'),
    path('get_all_x_features/', views.get_all_x_features, name='get_all_x_features'),
    path('get_all_y_features/', views.get_all_y_features, name='get_all_y_features'),

]