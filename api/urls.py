from django.urls import path
from .views import DoubleNumberView

urlpatterns = [
    path('double-num/', DoubleNumberView.as_view(), name='double_num'),
]