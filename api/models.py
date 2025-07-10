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
    # AbstractUser已包含id字段，此处可省略或保留
    id = models.BigAutoField(primary_key=True)
    role_id = models.ForeignKey(Role,on_delete=models.CASCADE, db_column='role_id',null=True,blank=True)
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
    # 修改：使用ForeignKey关联到User模型
    user = models.ForeignKey(User, on_delete=models.CASCADE, db_column='user_id',null=True, blank=True)
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
    face_image_path = models.CharField(max_length=255, null=False)
    face_embedding = models.TextField(null=False)

    class Meta:
        db_table = 'person'
        verbose_name = '主体'
        verbose_name_plural = '主体'

    def __str__(self):
        return self.name


class Camera(models.Model):
    id = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=64, null=False)
    location = models.CharField(max_length=128, null=False)
    stream_url = models.CharField(max_length=255, null=False)
    camera_type = models.CharField(max_length=32, null=False)
    is_active = models.BooleanField(null=False)

    class Meta:
        db_table = 'cameras'
        verbose_name = '摄像头信息'
        verbose_name_plural = '摄像头信息'

    def __str__(self):
        return self.name


class RecognitionLog(models.Model):
    id = models.BigAutoField(primary_key=True)
    # 修改：使用ForeignKey
    person = models.ForeignKey(Subject, on_delete=models.CASCADE, db_column='person_id',null=True, blank=True)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, db_column='camera_id',null=True, blank=True)
    time = models.DateTimeField(auto_now_add=False, null=False)
    confidence = models.FloatField(null=False)
    image_path = models.CharField(max_length=255, null=False)

    class Meta:
        db_table = 'recognition_log'
        verbose_name = '识别日志'
        verbose_name_plural = '识别日志'

    def __str__(self):
        return f"{self.person.name} @ {self.time}"


class DetectionLog(models.Model):
    id = models.BigAutoField(primary_key=True)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, db_column='camera_id', null=True, blank=True)
    object_class = models.CharField(max_length=32, null=False)
    confidence = models.FloatField(null=False)
    bbox = models.TextField(null=False)
    time = models.DateTimeField(auto_now_add=False, null=False)
    image_path = models.CharField(max_length=255, null=False)

    class Meta:
        db_table = 'detection_logs'
        verbose_name = '目标检测日志'
        verbose_name_plural = '目标检测日志'

    def __str__(self):
        return f"{self.object_class} @ {self.time}"


class WarningZone(models.Model):
    id = models.BigAutoField(primary_key=True)
    # 修改：使用ForeignKey
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, db_column='camera_id',null=True, blank=True)
    name = models.CharField(max_length=64, null=False)
    zone_type = models.PositiveSmallIntegerField(null=False)
    zone_points = models.TextField(null=False)
    is_active = models.BooleanField(null=False)

    class Meta:
        db_table = 'warning_zones'
        verbose_name = '预警区域'
        verbose_name_plural = '预警区域'

    def __str__(self):
        return self.name


class IncidentType(models.Model):
    id = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=64, null=False)
    code = models.CharField(max_length=32, null=False)

    class Meta:
        db_table = 'incident_types'
        verbose_name = '危险行为类型'
        verbose_name_plural = '危险行为类型'

    def __str__(self):
        return self.name


class IncidentDetectionLog(models.Model):
    id = models.BigAutoField(primary_key=True)
    # 修改：使用ForeignKey
    incident_type = models.ForeignKey(IncidentType, on_delete=models.CASCADE, db_column='incident_type_id', null=True, blank=True)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, db_column='camera_id', null=True, blank=True)
    time = models.DateTimeField(null=False)
    video_clip_path = models.CharField(max_length=255, null=False)
    confidence = models.FloatField(null=False)
    status = models.PositiveSmallIntegerField(null=False)

    class Meta:
        db_table = 'incident_detection_logs'
        verbose_name = '危险行为检测日志'
        verbose_name_plural = '危险行为检测日志'

    def __str__(self):
        return f"{self.incident_type.name} @ {self.time}"


class AlarmLog(models.Model):
    id = models.BigAutoField(primary_key=True)
    # 注意：这里的source_id可能指向不同的表，是一个通用外键场景，暂时保留
    source_type = models.CharField(max_length=32, null=False)
    source_id = models.BigIntegerField(null=False)
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