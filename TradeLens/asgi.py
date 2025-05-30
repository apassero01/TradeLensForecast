import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TradeLens.settings')

from django.core.asgi import get_asgi_application
django_asgi_app = get_asgi_application()

from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from training_session import routing as training_session_routing

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
        URLRouter(
            training_session_routing.websocket_urlpatterns
        )
    ),
})