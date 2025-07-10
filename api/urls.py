
# api/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from django.conf.urls.static import static
from django.conf import settings

from .views.data_views import DoubleNumberView 

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

] #+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# 导入 FeedbackView，因为它在 feedback_views.py 中
from .views.feedback_views import FeedbackView

urlpatterns = [
   
    path('double/<int:number>/', DoubleNumberView.as_view(), name='double-number'),

    # 这是为 Feedback 功能添加的路由，它现在应该能够正常工作
    path('feedbacks/', FeedbackView.as_view(), name='feedback-list-create'),
]

