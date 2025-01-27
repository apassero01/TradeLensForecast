from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/entities/$', consumers.GlobalEntityConsumer.as_asgi()),
    re_path(r'ws/entity/(?P<entity_id>[^/]+)/$', consumers.EntityConsumer.as_asgi()),
]