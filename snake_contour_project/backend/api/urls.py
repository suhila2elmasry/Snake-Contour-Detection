from django.urls import path
from .views import SnakeProcessView, HealthCheckView

urlpatterns = [
    path('process-snake/', SnakeProcessView.as_view(), name='process-snake'),
    path('health/', HealthCheckView.as_view(), name='health'),
]