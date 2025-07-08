# recognition-api/api/urls.py

from django.urls import path
from .views.data_views import DoubleNumberView 

# 导入 FeedbackView，因为它在 feedback_views.py 中
from .views.feedback_views import FeedbackView

urlpatterns = [
   
    path('double/<int:number>/', DoubleNumberView.as_view(), name='double-number'),

    # 这是为 Feedback 功能添加的路由，它现在应该能够正常工作
    path('feedbacks/', FeedbackView.as_view(), name='feedback-list-create'),
]