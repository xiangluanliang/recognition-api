from django.contrib.contenttypes.fields import GenericForeignKey
from django.db import models
from django.contrib.auth.models import AbstractUser

from django.contrib.auth.models import User 


class Role(models.Model):
    id = models.IntegerField(primary_key=True)
    # 普通用户-1
    # 管理员-2
    role_name = models.CharField(null=False)
    
    class Meta:
        db_table = 'role'
        verbose_name = '角色'
        verbose_name_plural = '角色'


class User(AbstractUser):
    id = models.BigAutoField(primary_key=True)
    role_id = models.ForeignKey(Role, on_delete=models.CASCADE, db_column='role_id')
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
    user = models.ForeignKey(User, on_delete=models.CASCADE, db_column='user_id', null=True)
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
    state = models.IntegerField(null=False, default=1)
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
    url = models.CharField(max_length=64, null=True)
    password = models.CharField(max_length=64, null=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)

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

    image_path = models.CharField(max_length=255, null=True, blank=True)
    video_clip_path = models.CharField(max_length=255, null=True, blank=True)

    person = models.ForeignKey(Subject, null=True, blank=True, on_delete=models.SET_NULL)
    # 仅人脸识别类型填写 person（关联黑名单人员）

    class Meta:
        db_table = 'event_logs'


class WarningZone(models.Model):
    id = models.BigAutoField(primary_key=True)
    # 修改：使用ForeignKey
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, db_column='camera_id', null=True)
    name = models.CharField(max_length=64, null=True)
    zone_type = models.PositiveSmallIntegerField(null=False)
    zone_points = models.JSONField(null=False)
    is_active = models.BooleanField(null=False)
    safe_distance = models.FloatField(null=True, blank=True, help_text="安全距离（米）")
    safe_time = models.PositiveIntegerField(null=True, blank=True, help_text="安全时间（秒）")

    class Meta:
        db_table = 'warning_zones'
        verbose_name = '预警区域'
        verbose_name_plural = '预警区域'

    def __str__(self):
        return self.name


class AlarmLog(models.Model):
    id = models.BigAutoField(primary_key=True)
    # 注意：这里的source_id可能指向不同的表，是一个通用外键场景，暂时保留
    title = models.CharField(null=True, blank=True)
    event = models.ForeignKey(EventLog, on_delete=models.CASCADE, db_column='event_id', default="")

    time = models.DateTimeField(null=False)
    result = models.CharField(max_length=64, null=True, blank=True)

    STATUS_CHOICES = [
        (0, '未处理'),
        (1, '处理中'),
        (2, '已处理'),
    ]
    status = models.PositiveSmallIntegerField(default=0, choices=STATUS_CHOICES)

    description = models.TextField(null=True, blank=True)

    class Meta:
        db_table = 'alarm_logs'
        verbose_name = '报警记录'
        verbose_name_plural = '报警记录'


#  Feedback 模型 
class Feedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='feedbacks')
    title = models.CharField(max_length=100)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self): 
        return f"'{self.title}' by {self.user.username}"

    updated_at = models.DateTimeField(auto_now=True)


class VideoAnalysisTask(models.Model):
    """
    专门用于跟踪用户上传视频并进行AI分析的任务。
    """
    # 任务状态的选项
    STATUS_CHOICES = [
        (0, '等待处理'),
        (1, '正在处理'),
        (2, '处理成功'),
        (-1, '处理失败'),
    ]

    id = models.BigAutoField(primary_key=True)

    # 关联到上传视频的用户
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, db_column='user_id')

    # 上传的原始视频文件路径
    original_video = models.FileField(upload_to='uploaded_videos/')

    # 处理后、画上了框的视频文件路径
    processed_video = models.FileField(upload_to='processed_videos/', null=True, blank=True)

    # 任务的状态
    status = models.SmallIntegerField(choices=STATUS_CHOICES, default=0)

    # 存储AI分析的原始结果（JSON格式）
    analysis_result = models.JSONField(null=True, blank=True)

    # 任务创建和更新的时间
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'video_analysis_tasks'
        verbose_name = '视频分析任务'
        verbose_name_plural = '视频分析任务'

    def __str__(self):
        return f"任务 {self.id} - 状态: {self.get_status_display()}"


# api/models.py

class DailyReport(models.Model):
    id = models.BigAutoField(primary_key=True)
    date = models.DateField(auto_now_add=True, unique=True)  # 每天一条
    content = models.TextField()  # 日报正文内容

    class Meta:
        db_table = 'daily_report'
        verbose_name = 'AI 日报'
