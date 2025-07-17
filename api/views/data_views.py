# api/views/data_views.py
import logging
from datetime import datetime
from datetime import timedelta

from django.contrib.auth import authenticate

from django.db.models import Count
from django.db.models.functions import TruncDay
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone
from django.utils.timezone import now
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import permissions, viewsets, status
from rest_framework.authtoken.models import Token
from rest_framework.decorators import action
from rest_framework.exceptions import PermissionDenied
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView


from ..models import (
    User, OperationLog, Subject, WarningZone, Camera, AlarmLog, EventLog, DailyReport
)
from ..serializers import (
    UserSerializer, OperationLogSerializer, SubjectSerializer, WarningZoneSerializer, CameraSerializer,
    AlarmLogSerializer, RegisterSerializer, EventLogSerializer, DailyReportSerializer
)

logger = logging.getLogger(__name__)
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

    @action(detail=False, methods=['get'], url_path='count')
    def count(self, request):
        user_count = User.objects.count()
        return Response({'count': user_count})


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

    @action(detail=False, methods=['get'], url_path='by-camera/(?P<camera_id>[^/.]+)')
    def get_by_camera(self, request, camera_id=None):
        try:
            camera_id_int = int(camera_id)
        except ValueError:
            return Response({"error": "camera_id必须是整数"}, status=400)

        zones = self.queryset.filter(camera_id=camera_id_int)
        serializer = self.get_serializer(zones, many=True)
        return Response(serializer.data)


class CameraViewSet(viewsets.ModelViewSet):
    queryset = Camera.objects.all()
    serializer_class = CameraSerializer
    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=["get"])
    def count(self, request):
        total_cameras = self.get_queryset().count()
        return Response({"count": total_cameras})

    @action(detail=False, methods=['get'], url_path='my_cameras')
    def my_cameras(self, request):
        user = request.user
        cameras = Camera.objects.filter(user=user)
        serializer = self.get_serializer(cameras, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'], url_path='flow_info')
    def flow_info(self, request, pk=None):
        try:
            camera = self.get_object()
            # 可加权限判断：只能查看自己的摄像头
            if camera.user != request.user:
                return Response({"detail": "没有权限查看该摄像头"}, status=403)
            return Response({
                "url": camera.url,
                "password": camera.password,
            })
        except Camera.DoesNotExist:
            return Response({"detail": "摄像头不存在"}, status=404)

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context.update({"request": self.request})
        return context


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
            today = (timezone.now()+timedelta(hours=8)).date()
            queryset = queryset.filter(time__date=today)

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
        today = (now()+timedelta(hours=8)).date()
        start_date = today - timedelta(days=6)+timedelta(hours=8)  # 最近7天

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
                count_dict[(day+timedelta(hours=8)).strftime('%m-%d')] = alarm['count']
        count_list = [count_dict.get(date, 0) for date in date_list]
        # print("alarm trend 数据原始结果：", list(alarms))
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
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['camera', 'event_type']


class DailyReportDataAPI(APIView):
    """
    为本地AI机提供生成日报所需的数据摘要。
    (已进一步优化查询逻辑和健壮性)
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        try:
            logger.info("DailyReportDataAPI: 开始为日报生成收集数据...")

            today = timezone.localdate()
            start_of_day = timezone.make_aware(datetime.combine(today, datetime.min.time()))
            end_of_day = timezone.make_aware(datetime.combine(today, datetime.max.time()))

            alarms_today = AlarmLog.objects.filter(time__range=(start_of_day, end_of_day))

            summary = {
                '日期': str(today),
                '总告警数': alarms_today.count(),
                '未处理告警数': alarms_today.filter(status=0).count(),
                '处理中告警数': alarms_today.filter(status=1).count(),
                '已处理告警数': alarms_today.filter(status=2).count(),
                '摄像头总数': Camera.objects.count(),
                '在线摄像头': Camera.objects.filter(is_active=True).count(),
                '各类型事件统计': {},
            }

            type_counts = alarms_today.values('event__event_type').annotate(count=Count('id')).order_by()

            event_type_map = dict(EventLog.EVENT_TYPE_CHOICES)

            for item in type_counts:
                event_type_key = item.get('event__event_type')
                count = item.get('count')

                if event_type_key and count:
                    display_name = event_type_map.get(event_type_key, event_type_key)
                    summary['各类型事件统计'][display_name] = count

            logger.info("DailyReportDataAPI: 数据收集成功。")
            return Response(summary, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"DailyReportDataAPI 在生成数据时发生严重错误: {e}", exc_info=True)
            return Response(
                {"error": "服务器在生成报告数据时发生内部错误。", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SubmitDailyReportAPI(APIView):
    """
    接收本地AI机生成的报告文本，并存入数据库。
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        """
        当接收到POST请求时，将报告内容写入数据库。
        """
        report_content = request.data.get('content')
        report_date_str = request.data.get('date')

        if not report_content or not report_date_str:
            return Response(
                {"error": "请求体中必须包含 'content' 和 'date' 字段。"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # 将字符串格式的日期转换回date对象
            report_date = datetime.strptime(report_date_str, '%Y-%m-%d').date()
        except ValueError:
            return Response(
                {"error": "日期格式无效，请使用 'YYYY-MM-DD' 格式。"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 使用 update_or_create 避免重复创建当天的报告
        report, created = DailyReport.objects.update_or_create(
            date=report_date,
            defaults={'content': report_content}
        )

        message = "日报更新成功。" if not created else "日报创建成功。"
        return Response({"status": "success", "message": message}, status=status.HTTP_201_CREATED)

@receiver(post_save, sender=EventLog)
def create_alarm_on_event(sender, instance, created, **kwargs):
    """
    当一个新的EventLog被创建时，检查是否需要生成告警。
    """
    if created:
        event = instance
        ALARM_WORTHY_EVENTS = ['fire', 'intrusion', 'conflict', 'face_match', 'audio_screaming']

        if event.event_type in ALARM_WORTHY_EVENTS:
            AlarmLog.objects.create(
                title=f"触发告警：{event.get_event_type_display()}",
                event=event,
                time=event.time,
                status=0,
                description=f"检测到 {event.get_event_type_display()}，摄像头ID：{event.camera.id if event.camera else '未知'}"
            )
            print(f"信号触发：为事件 {event.id} 创建了新的告警记录。")