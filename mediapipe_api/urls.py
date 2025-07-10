from django.urls import path
from .views import ProcessVideoAPIView, TaskStatusAPIView

urlpatterns = [
    path('/process_video/', ProcessVideoAPIView.as_view(), name='process-video'),
    path('/status/<uuid:task_id>/', TaskStatusAPIView.as_view()),

]
