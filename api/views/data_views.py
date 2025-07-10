# api/views/data_views.py
from rest_framework import permissions, viewsets, status
from rest_framework.exceptions import PermissionDenied
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from ..models import (
    User, OperationLog, Subject, WarningZone, Camera, AlarmLog, EventLog
)
from ..serializers import (
    UserSerializer, OperationLogSerializer, SubjectSerializer, WarningZoneSerializer, CameraSerializer,
    AlarmLogSerializer, RegisterSerializer, EventLogSerializer
)


class IsAdminOrSelf(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        # 管理员全权访问
        if request.user.role_id == 1:
            return True
        # 不是管理员只能访问自己的数据
        return obj.id == request.user.id


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated, IsAdminOrSelf]

    def get_queryset(self):
        user = self.request.user

        # 先检查用户是否已登录认
        if not user.is_authenticated:
            return User.objects.none()
        
        # 只有当用户登录后，才安全地访问 role_id
        if user.role_id == 1:
            return User.objects.all()
        else:
            return User.objects.filter(pk=user.pk)

    def perform_update(self, serializer):
        user = self.request.user
        # 普通用户不能修改role_id和status
        if user.role_id != 1:
            if 'role_id' in self.request.data or 'status' in self.request.data:
                raise PermissionDenied("无权限修改 role_id 或 status")
        serializer.save()

    def perform_destroy(self, instance):
        user = self.request.user
        # 只有管理员能删除用户
        if user.role_id != 1:
            raise PermissionDenied("无权限删除用户")
        instance.delete()


class OperationLogViewSet(viewsets.ModelViewSet):
    queryset = OperationLog.objects.all().order_by('-timestamp')
    serializer_class = OperationLogSerializer
    permission_classes = [IsAuthenticated]


class SubjectViewSet(viewsets.ModelViewSet):
    queryset = Subject.objects.all()
    serializer_class = SubjectSerializer
    permission_classes = [IsAuthenticated]

    def create(self, request, *args, **kwargs):
        print("【调试】当前用户是：", request.user)
        return super().create(request, *args, **kwargs)


class WarningZoneViewSet(viewsets.ModelViewSet):
    queryset = WarningZone.objects.all()
    serializer_class = WarningZoneSerializer
    permission_classes = [IsAuthenticated]


class CameraViewSet(viewsets.ModelViewSet):
    queryset = Camera.objects.all()
    serializer_class = CameraSerializer
    permission_classes = [IsAuthenticated]


class AlarmLogViewSet(viewsets.ModelViewSet):
    queryset = AlarmLog.objects.all().order_by('-time')
    serializer_class = AlarmLogSerializer
    permission_classes = [IsAuthenticated]


class RegisterView(APIView):
    permission_classes = []  # 注册接口允许匿名访问

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({"message": "注册成功！"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class EventLogViewSet(viewsets.ModelViewSet):
    queryset = EventLog.objects.all().order_by('-time')  # 最新的排前面
    serializer_class = EventLogSerializer
    permission_classes = [IsAuthenticated]

    # 可选：你可以添加 create 或 perform_create 钩子来添加日志、自动报警等
    def perform_create(self, serializer):
        event = serializer.save()
        print(f"记录新事件：{event.event_type}，来自摄像头：{event.camera_id}")
        # 如果需要，还可以在这里触发自动报警（比如写入 AlarmLog）