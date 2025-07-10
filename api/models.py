from django.contrib.contenttypes.fields import GenericForeignKey
from django.db import models
from django.contrib.auth.models import AbstractUser


class Role(models.Model):
    id = models.IntegerField(primary_key=True)
    role_name = models.CharField(null=False)
    
    class Meta:
        db_table = 'role'
        verbose_name = '角色'
        verbose_name_plural = '角色'


class User(AbstractUser):
    id = models.BigAutoField(primary_key=True)
    role_id = models.ForeignKey(Role, on_delete=models.CASCADE(), db_column='role_id')
    status = models.PositiveSmallIntegerField(null=False, default=1)
    created_at = models.DateTimeField(auto_now_add=True, null=False)

    class Meta:
        db_table = 'users'
        verbose_name = '用户'
        verbose_name_plural = '用户'

    def __str__(self):
        return self.username


class OperationLog(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, db_column='user_id')
    action = models.CharField(max_length=64, null=False)
    ip = models.CharField(max_length=45, null=False)
    timestamp = models.DateTimeField(auto_now_add=True, null=False)
    description = models.CharField(max_length=255, null=True, blank=True)

    class Meta:
        db_table = 'operation_log'
        verbose_name = '操作日志'
        verbose_name_plural = '操作日志'

    def __str__(self):
        return f"{self.user.username} - {self.action} @ {self.timestamp}"


class Subject(models.Model):
    id = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=64, null=False)
    state = models.IntegerField(null=False)
    face_image_path = models.CharField(max_length=255, null=False)
    face_embedding = models.JSONField(null=True)

    class Meta:
        db_table = 'person'
        verbose_name = '主体'
        verbose_name_plural = '主体'

    def __str__(self):
        return self.name


class Camera(models.Model):
    id = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=64, null=False)
    location = models.CharField(max_length=128, null=True)
    camera_type = models.CharField(max_length=32, null=True)
    is_active = models.BooleanField(null=False)

    class Meta:
        db_table = 'cameras'
        verbose_name = '摄像头信息'
        verbose_name_plural = '摄像头信息'

    def __str__(self):
        return self.name


class EventLog(models.Model):
    EVENT_TYPE_CHOICES = [
        ('face_match', '识别危险人员'),
        ('person_fall', '人员跌倒'),
        ('fire', '火灾'),
        ('intrusion', '区域入侵'),
        ('conflict', '打架冲突'),
    ]

    event_type = models.CharField(max_length=32, choices=EVENT_TYPE_CHOICES)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, null=True)
    time = models.DateTimeField()
    confidence = models.FloatField()

    image_path = models.CharField(max_length=255, null=True, blank=True)  # 报警时截图
    video_clip_path = models.CharField(max_length=255, null=True, blank=True)  # 报警时视频片段

    person = models.ForeignKey(Subject, null=True, blank=True, on_delete=models.SET_NULL)
    # 仅人脸识别类型填写 person（关联黑名单人员）

    STATUS_CHOICES = [
        (0, '未处理'),
        (1, '处理中'),
        (2, '已处理'),
    ]
    status = models.PositiveSmallIntegerField(default=0, choices=STATUS_CHOICES)

    description = models.TextField(null=True, blank=True)

    class Meta:
        db_table = 'event_logs'


class WarningZone(models.Model):
    id = models.BigAutoField(primary_key=True)
    # 修改：使用ForeignKey
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, db_column='camera_id')
    name = models.CharField(max_length=64, null=True)
    zone_type = models.PositiveSmallIntegerField(null=False)
    zone_points = models.JSONField(null=False)
    is_active = models.BooleanField(null=False)

    class Meta:
        db_table = 'warning_zones'
        verbose_name = '预警区域'
        verbose_name_plural = '预警区域'

    def __str__(self):
        return self.name


class AlarmLog(models.Model):
    id = models.BigAutoField(primary_key=True)
    # 注意：这里的source_id可能指向不同的表，是一个通用外键场景，暂时保留
    event = models.ForeignKey(EventLog, on_delete=models.CASCADE, db_column='event_id')

    time = models.DateTimeField(null=False)
    method = models.CharField(max_length=32, null=False)
    receiver = models.CharField(max_length=64, null=False)
    result = models.CharField(max_length=64, null=True, blank=True)

    class Meta:
        db_table = 'alarm_logs'
        verbose_name = '报警记录'
        verbose_name_plural = '报警记录'

    def __str__(self):
        return f"{self.source_type} @ {self.time}"