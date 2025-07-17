from django.conf import settings
import os
from rest_framework import serializers
from .models import (
    User, OperationLog, Subject, WarningZone, Camera, AlarmLog, EventLog, VideoAnalysisTask, Feedback, Role, DailyReport
)


class UserSerializer(serializers.ModelSerializer):
    role_name = serializers.CharField(source='role_id.name', read_only=True)

    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'role_id', 'role_name', 'status', 'created_at']
        read_only_fields = ['id', 'created_at']


class OperationLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = OperationLog
        # 修改：将 user_id 改为 user
        fields = ['id', 'user', 'action', 'ip', 'timestamp', 'description']
        read_only_fields = ['id', 'timestamp']


class SubjectSerializer(serializers.ModelSerializer):
    face_image = serializers.ImageField(write_only=True, required=True)
    face_image_path = serializers.CharField(read_only=True)

    class Meta:
        model = Subject
        fields = ['id', 'name', 'face_image', 'face_image_path', 'face_embedding']
        read_only_fields = ['id', 'face_image_path']

    def create(self, validated_data):
        image_file = validated_data.pop('face_image')
        # 优化：使用settings.MEDIA_ROOT构建路径
        save_dir = os.path.join(settings.MEDIA_ROOT, 'subject_images')
        os.makedirs(save_dir, exist_ok=True)
        # 注意：这里我们只取文件名，而不是完整的路径
        filename = image_file.name
        full_path = os.path.join(save_dir, filename)

        with open(full_path, 'wb+') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        # 使用MEDIA_URL构建返回给前端的URL
        validated_data['face_image_path'] = os.path.join(settings.MEDIA_URL, 'subject_images', filename).replace('\\',
                                                                                                                 '/')
        return Subject.objects.create(**validated_data)


class WarningZoneSerializer(serializers.ModelSerializer):
    # 直接传 camera_id
    camera = serializers.PrimaryKeyRelatedField(queryset=Camera.objects.all())

    class Meta:
        model = WarningZone
        fields = ['id', 'camera', 'name', 'zone_type', 'zone_points', 'is_active', 'safe_distance', 'safe_time']
        read_only_fields = ['id']

class CameraSerializer(serializers.ModelSerializer):
    class Meta:
        model = Camera
        ffields = ['id', 'name', 'location', 'camera_type', 'is_active', 'url', 'password', 'active_detectors']
        read_only_fields = ['id']

    def create(self, validated_data):
        # 自动绑定当前登录用户
        user = self.context['request'].user
        camera = Camera.objects.create(user=user, **validated_data)
        if not camera.name:
            camera.name = f"摄像头{camera.id}"
            camera.save()
        return camera

class AlarmLogSerializer(serializers.ModelSerializer):
    event_type = serializers.CharField(source='event.event_type', read_only=True)

    class Meta:
        model = AlarmLog
        fields = ['id', 'title', 'time', 'result', 'status', 'description', 'event_id', 'event_type']
        read_only_fields = ['id']


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    role_id = serializers.PrimaryKeyRelatedField(queryset=Role.objects.all())  # 新增
    first_name = serializers.CharField(required=False, allow_blank=True)
    last_name = serializers.CharField(required=False, allow_blank=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'first_name', 'last_name', 'role_id']

    def create(self, validated_data):
        role = validated_data.pop('role_id')
        user = User(
            username=validated_data['username'],
            email=validated_data['email'],
            first_name=validated_data.get('first_name', ''),
            last_name=validated_data.get('last_name', ''),
            is_active=True,
            status=1,
            role_id=role
        )
        user.set_password(validated_data['password'])
        user.save()
        return user


class EventLogSerializer(serializers.ModelSerializer):
    # event_type_display = serializers.CharField(source='get_event_type_display', read_only=True)
    # status_display = serializers.CharField(source='get_status_display', read_only=True)

    camera_name = serializers.CharField(source='camera.name', read_only=True)
    person_name = serializers.CharField(source='person.name', read_only=True)

    camera = serializers.PrimaryKeyRelatedField(queryset=Camera.objects.all())
    person = serializers.PrimaryKeyRelatedField(queryset=Subject.objects.all(), allow_null=True, required=False)

    class Meta:
        model = EventLog
        fields = [
            'id', 'event_type', 'time', 'confidence',
            'image_path', 'video_clip_path',
            'camera', 'camera_name',
            'person', 'person_name',
        ]


class VideoAnalysisTaskSerializer(serializers.ModelSerializer):
    # 让前端能看到可读的状态名，而不是数字
    status = serializers.CharField(source='get_status_display', read_only=True)
    # 让前端能看到关联的用户名
    user = serializers.StringRelatedField(read_only=True)
    class Meta:
        model = VideoAnalysisTask
        fields = ['id', 'user', 'original_video', 'status', 'analysis_result', 'created_at', 'updated_at']
        read_only_fields = ['id', 'status', 'analysis_result', 'created_at', 'updated_at', 'user']


class FeedbackSerializer(serializers.ModelSerializer):
    # 让返回的JSON中包含用户名，而不仅仅是用户ID
    user = serializers.CharField(source='user.username', read_only=True)

    class Meta:
        model = Feedback
        # 定义API应该暴露哪些字段
        fields = ['id', 'user', 'title', 'content', 'created_at']
        # 将user字段设为只读，因为我们会根据当前登录用户自动设置
        read_only_fields = ['user', 'created_at']


class DailyReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = DailyReport
        fields = ['id', 'date', 'content']
        read_only_fields = ['id', 'date', 'content']

class KnownFaceSerializer(serializers.ModelSerializer):
    """
    一个只读的序列化器，仅包含AI人脸比对所必需的字段。
    """
    class Meta:
        model = Subject
        # 只选择AI Worker需要的最少字段，减少网络传输
        fields = ['id', 'name', 'state', 'face_embedding']
        read_only_fields = fields


class EventLogCreateSerializer(serializers.ModelSerializer):
    """
    一个只写的序列化器，用于接收来自AI Worker的事件数据并创建EventLog记录。
    它接收的是ID，而不是复杂的对象。
    """
    # 让序列化器接受来自AI Worker的摄像头ID
    # 我们期望AI Worker发送 'camera': 1 这样的数据
    camera = serializers.PrimaryKeyRelatedField(queryset=Camera.objects.all())

    # person字段是可选的，因为并非所有事件都关联到person
    # AI Worker可以发送 'person': 5 或者不发送此字段
    person = serializers.PrimaryKeyRelatedField(queryset=Subject.objects.all(), allow_null=True, required=False)

    class Meta:
        model = EventLog
        # 定义AI Worker需要提交的字段
        fields = [
            'event_type',
            'camera',
            'time',
            'confidence',
            'image_path',
            'video_clip_path',
            'person'
        ]
