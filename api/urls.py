from django.urls import path
from .views import DoubleNumberView
from .views.yolo_views import YoloDetectView

#将路由对应到响应的视图
urlpatterns = [
    path('double-num/', DoubleNumberView.as_view(), name='double_num'),
    path('yolo-detect/', YoloDetectView.as_view(), name='yolo-detect'),
]