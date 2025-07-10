
# api/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from django.conf.urls.static import static
from django.conf import settings

from .views.data_views import (
    OperationLogViewSet,
    SubjectViewSet,
    WarningZoneViewSet,
    EventLogViewSet,
    CameraViewSet,
    AlarmLogViewSet,
    UserViewSet,
    RegisterView,
    LoginView
)

from .views.video_views import (
    TaskResultView,
    VideoUploadAndProcessView
)

from .views.feedback_views import FeedbackView

router = DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'operation_logs', OperationLogViewSet)
router.register(r'subjects', SubjectViewSet)
router.register(r'warning_zones', WarningZoneViewSet)
router.register(r'events',EventLogViewSet)
router.register(r'cameras', CameraViewSet)
router.register(r'alarm_logs', AlarmLogViewSet)

urlpatterns = [
    path('login/', LoginView.as_view(), name='login'),
    path('', include(router.urls)),
    path('register/', RegisterView.as_view(), name='register'),
    path('process-video/', VideoUploadAndProcessView.as_view(), name='process-video'),
    path('task-result/<int:pk>/', TaskResultView.as_view(), name='task-result'),
    path('feedbacks/', FeedbackView.as_view(), name='feedback-list-create'),
] #+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

