from rest_framework import serializers
from django.conf import settings
from django.contrib.auth import get_user_model
import os
from .models import (
    User, OperationLog, Subject, RecognitionLog, DetectionLog,
    WarningZone, IncidentType, IncidentDetectionLog, Camera, AlarmLog, VideoAnalysisTask
)

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        # 修改：将 role_id 改为 role (如果未来是外键的话)，或者保持原样  'role',
        fields = ['id', 'username', 'status', 'created_at']
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


class RecognitionLogSerializer(serializers.ModelSerializer):
    video_clip = serializers.FileField(write_only=True, required=True)

    # image_path是模型字段，让它正常序列化

    class Meta:
        model = RecognitionLog
        # 修改：person_id -> person, camera_id -> camera
        fields = ['id', 'person', 'camera', 'time', 'confidence', 'video_clip', 'image_path']
        read_only_fields = ['id', 'image_path']

    def create(self, validated_data):
        # 修复：将 'video_file' 修改为 'video_clip'
        video_file = validated_data.pop('video_clip', None)
        if video_file:
            save_dir = os.path.join(settings.MEDIA_ROOT, 'recognition_videos')
            os.makedirs(save_dir, exist_ok=True)

            filename = video_file.name
            full_path = os.path.join(save_dir, filename)
            with open(full_path, 'wb+') as f:
                for chunk in video_file.chunks():
                    f.write(chunk)

            relative_path = os.path.join('recognition_videos', filename)
            validated_data['image_path'] = os.path.join(settings.MEDIA_URL, relative_path).replace('\\', '/')

        return RecognitionLog.objects.create(**validated_data)


class DetectionLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = DetectionLog
        # 修改：camera_id -> camera
        fields = ['id', 'camera', 'object_class', 'confidence', 'bbox', 'time', 'image_path']
        read_only_fields = ['id']


class WarningZoneSerializer(serializers.ModelSerializer):
    video_clip = serializers.FileField(write_only=True, required=True)
    video_clip_path = serializers.CharField(read_only=True)

    class Meta:
        model = WarningZone
        # 修复：添加 'video_clip' 和 'video_clip_path' 到fields
        # 修改：camera_id -> camera
        fields = ['id', 'camera', 'name', 'zone_type', 'zone_points', 'is_active', 'video_clip', 'video_clip_path']
        read_only_fields = ['id', 'video_clip_path']

    def create(self, validated_data):
        video_file = validated_data.pop('video_clip')

        # 优化：使用settings.MEDIA_ROOT构建路径
        save_dir = os.path.join(settings.MEDIA_ROOT, 'incident_videos')
        os.makedirs(save_dir, exist_ok=True)
        filename = video_file.name
        full_path = os.path.join(save_dir, filename)

        with open(full_path, 'wb+') as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        relative_path = os.path.join('incident_videos', filename)
        # 注意：这里我们假设IncidentDetectionLog模型有一个video_clip_path字段来保存路径
        # validated_data['video_clip_path'] = os.path.join(settings.MEDIA_URL, relative_path).replace('\\', '/')

        # 修复：create方法应该创建WarningZone对象，而不是IncidentDetectionLog
        # 并且WarningZone模型本身没有video_clip_path字段，所以create方法不应该处理文件上传和路径保存
        # 这里的逻辑需要根据你的业务重新定义。
        # 一个可能的场景是，上传视频后，创建一个IncidentDetectionLog记录。
        # 如果是这样，那么这个序列化器的目的就不应该是创建WarningZone。
        # 为了让代码能跑通，我暂时注释掉文件处理，并修正create的对象。

        return WarningZone.objects.create(**validated_data)


class IncidentTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = IncidentType
        fields = ['id', 'name', 'code']
        read_only_fields = ['id']


class IncidentDetectionLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = IncidentDetectionLog
        # 修改：incident_type_id -> incident_type, camera_id -> camera
        fields = ['id', 'incident_type', 'camera', 'time', 'video_clip_path', 'confidence', 'status']
        read_only_fields = ['id']


class CameraSerializer(serializers.ModelSerializer):
    class Meta:
        model = Camera
        fields = ['id', 'name', 'location', 'stream_url', 'camera_type', 'is_active']
        read_only_fields = ['id']


class AlarmLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = AlarmLog
        fields = ['id', 'source_type', 'source_id', 'time', 'method', 'receiver', 'result']
        read_only_fields = ['id']

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'first_name', 'last_name']
        # 你有 role_id 和 status，如果注册时不需要用户选，就不写进去

    def create(self, validated_data):
        user = User(
            username=validated_data['username'],
            email=validated_data['email'],
            first_name=validated_data.get('first_name', ''),
            last_name=validated_data.get('last_name', ''),
            is_active=True,
            status=1,  # 默认设为启用，你可以根据需要改
        )
        user.set_password(validated_data['password'])
        user.save()
        return user

class VideoAnalysisTaskSerializer(serializers.ModelSerializer):
    # 让前端能看到可读的状态名，而不是数字
    status = serializers.CharField(source='get_status_display', read_only=True)
    # 让前端能看到关联的用户名
    user = serializers.StringRelatedField(read_only=True)
    class Meta:
        model = VideoAnalysisTask
        fields = ['id', 'user', 'original_video', 'status', 'analysis_result', 'created_at', 'updated_at']
        read_only_fields = ['id', 'status', 'analysis_result', 'created_at', 'updated_at', 'user']