from django.urls import path
from .views import ProcessVideoAPIView

urlpatterns = [
    path('/process_video/', ProcessVideoAPIView.as_view(), name='process-video')
]
