# api/views/data_views.py
from django.contrib.auth import authenticate
from rest_framework import permissions, viewsets, status
from rest_framework.authtoken.models import Token
from rest_framework.exceptions import PermissionDenied
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from ..models import (
    User, OperationLog, Subject, RecognitionLog, DetectionLog,
    WarningZone, IncidentType, IncidentDetectionLog, Camera, AlarmLog
)
from ..serializers import (
    UserSerializer, OperationLogSerializer, SubjectSerializer, RecognitionLogSerializer,
    DetectionLogSerializer, WarningZoneSerializer, IncidentTypeSerializer,
    IncidentDetectionLogSerializer, CameraSerializer, AlarmLogSerializer,
    RegisterSerializer
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


class RecognitionLogViewSet(viewsets.ModelViewSet):
    queryset = RecognitionLog.objects.all().order_by('-time')
    serializer_class = RecognitionLogSerializer
    permission_classes = [IsAuthenticated]

    def create(self, request, *args, **kwargs):
        print("【调试】当前用户是：", request.user)
        return super().create(request, *args, **kwargs)


class DetectionLogViewSet(viewsets.ModelViewSet):
    queryset = DetectionLog.objects.all().order_by('-time')
    serializer_class = DetectionLogSerializer
    permission_classes = [IsAuthenticated]


class WarningZoneViewSet(viewsets.ModelViewSet):
    queryset = WarningZone.objects.all()
    serializer_class = WarningZoneSerializer
    permission_classes = [IsAuthenticated]


class IncidentTypeViewSet(viewsets.ModelViewSet):
    queryset = IncidentType.objects.all()
    serializer_class = IncidentTypeSerializer
    permission_classes = [IsAuthenticated]


class IncidentDetectionLogViewSet(viewsets.ModelViewSet):
    queryset = IncidentDetectionLog.objects.all().order_by('-time')
    serializer_class = IncidentDetectionLogSerializer
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
        print(request.data)
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({"message": "注册成功！"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LoginView(APIView):
    permission_classes = []

    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')

        if not username or not password:
            return Response({'message': '用户名和密码不能为空'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return Response({'message': '用户不存在，请先注册'}, status=status.HTTP_401_UNAUTHORIZED)

        user = authenticate(username=username, password=password)
        if user:
            token, created = Token.objects.get_or_create(user=user)
            return Response({
                'token': token.key,
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'status': user.status,
                    'created_at': user.created_at
                }
            })
        else:
            return Response({'message': '密码错误'}, status=status.HTTP_401_UNAUTHORIZED)