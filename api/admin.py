# api/admin.py
from django.contrib import admin
from .models import TestNumber, Feedback # 导入Feedback模型

admin.site.register(TestNumber)
admin.site.register(Feedback) # 注册Feedback模型

