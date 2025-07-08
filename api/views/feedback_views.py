# recognition-api/api/views/feedback_views.py

# --- 务必确保以下所有导入都存在！ ---
from rest_framework.views import APIView # 导入APIView
from rest_framework.response import Response # 导入Response
from rest_framework import status # 导入HTTP状态码
# 导入认证和权限类
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated

# 从 api 应用的顶层导入 models 和 serializers
from ..models import Feedback
from ..serializers import FeedbackSerializer

# 如果有其他视图类需要，可以放在这里 ...

class FeedbackView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    # 定义GET方法，用于获取反馈列表
    def get(self, request):
        """获取所有反馈列表"""
        feedbacks = Feedback.objects.all().order_by('-created_at')
        serializer = FeedbackSerializer(feedbacks, many=True)
        return Response({"code": 0, "info": serializer.data})

    # 定义POST方法，用于创建一条新反馈
    def post(self, request):
        """创建一条新反馈"""
        serializer = FeedbackSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save(user=request.user)
            return Response({"code": 0, "info": serializer.data}, status=status.HTTP_201_CREATED)
        # 如果is_valid验证失败且没有raise_exception，则会执行下面这行
        # return Response({"code": 400, "message": "输入数据无效", "details": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)