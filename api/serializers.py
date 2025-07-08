# api/serializers.py
from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Feedback # 导入Feedback模型

# ... UserSerializer ...

class FeedbackSerializer(serializers.ModelSerializer):
    # 让返回的JSON中包含用户名，而不仅仅是用户ID
    user = serializers.CharField(source='user.username', read_only=True)

    class Meta:
        model = Feedback
        # 定义API应该暴露哪些字段
        fields = ['id', 'user', 'title', 'content', 'created_at']
        # 将user字段设为只读，因为我们会根据当前登录用户自动设置
        read_only_fields = ['user', 'created_at']