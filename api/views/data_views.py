# api/views/data_views.py
from datetime import datetime, time
from datetime import timedelta

from django.contrib.auth import authenticate
from django.db.models import Count
from django.db.models.functions import TruncDay
from django.utils import timezone
from django.utils.timezone import now
from rest_framework import permissions, viewsets, status
from rest_framework.authtoken.models import Token
from rest_framework.decorators import action
from rest_framework.exceptions import PermissionDenied
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from werkzeug.debug import console

from ..models import (
    User, OperationLog, Subject, WarningZone, Camera, AlarmLog, EventLog, DailyReport
)
from ..serializers import (
    UserSerializer, OperationLogSerializer, SubjectSerializer, WarningZoneSerializer, CameraSerializer,
    AlarmLogSerializer, RegisterSerializer, EventLogSerializer, DailyReportSerializer
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
    queryset = AlarmLog.objects.select_related('event').all()
    serializer_class = AlarmLogSerializer
    permission_classes = [IsAuthenticated]

    @action(detail=True, methods=['patch'], url_path='update_event_status')
    def update_event_status(self, request, pk=None):
        alarm = self.get_object()
        event = alarm.event

        new_status = request.data.get('status')
        description = request.data.get('description', '')

        if new_status not in [0, 1, 2]:
            return Response({'detail': 'Invalid status'}, status=status.HTTP_400_BAD_REQUEST)

        event.status = new_status
        event.description = description
        event.save()

        return Response({'detail': '事件状态更新成功'})

    def get_queryset(self):
        queryset = AlarmLog.objects.select_related('event').all()
        today_param = self.request.query_params.get("today")

        if today_param == "true":
            today = timezone.now().date()
            queryset = queryset.filter(event__time__date=today)

        return queryset

    @action(detail=True, methods=['get'], url_path='event_detail')
    def event_detail(self, request, pk=None):
        alarm = self.get_object()
        event = alarm.event
        return Response({
            'event_type': event.event_type,
            'event_time': event.time,
            'status': event.status,
            'description': event.description,
            'video_clip_path': event.video_clip_path,
            'image_path': event.image_path,
        })

    @action(detail=False, methods=['get'], url_path='trend')
    def trend(self, request):
        today = now().date()
        start_date = today - timedelta(days=6)  # 最近7天

        alarms = (
            AlarmLog.objects
            .filter(time__date__gte=start_date, time__date__lte=today)
            .annotate(day=TruncDay('time'))
            .values('day')
            .annotate(count=Count('id'))
            .order_by('day')
        )

        date_list = [(start_date + timedelta(days=i)).strftime('%m-%d') for i in range(7)]
        count_dict = {}
        for alarm in alarms:
            day = alarm['day']
            if day:
                count_dict[day.strftime('%m-%d')] = alarm['count']
        count_list = [count_dict.get(date, 0) for date in date_list]
        print("🎯 alarm trend 数据原始结果：", list(alarms))
        return Response({
            "dates": date_list,
            "counts": count_list,
        })


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
                    'created_at': user.created_at,
                    'role': user.role_id.id
                }
            })
        else:
            return Response({'message': '密码错误'}, status=status.HTTP_401_UNAUTHORIZED)


class EventLogViewSet(viewsets.ModelViewSet):
    queryset = EventLog.objects.all().order_by('-time')
    serializer_class = EventLogSerializer
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        event = serializer.save()
        print(f"记录新事件：{event.event_type}，来自摄像头：{event.camera_id}")


class DailyReportViewSet(viewsets.ViewSet):
    @action(detail=False, methods=['get'], url_path='today')
    def today(self, request):
        today = now().date()
        try:
            report = DailyReport.objects.get(date=today)
            serializer = DailyReportSerializer(report)
            return Response({
                'date': str(today),
                'content': serializer.data['content'],
            })
        except DailyReport.DoesNotExist:
            return Response({
                'date': str(today),
                'content': '今日暂无生成的 AI 日报。',
            })
