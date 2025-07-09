from rest_framework import serializers
from .models import (
    User, OperationLog, Subject, RecognitionLog, DetectionLog,
    WarningZone, IncidentType, IncidentDetectionLog, Camera, AlarmLog
)


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'role_id', 'status', 'created_at']
        read_only_fields = ['id', 'created_at']


class OperationLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = OperationLog
        fields = ['id', 'user_id', 'action', 'ip', 'timestamp', 'description']
        read_only_fields = ['id', 'timestamp']


class SubjectSerializer(serializers.ModelSerializer):
    face_image = serializers.ImageField(write_only=True, required=True)
    face_image_path = serializers.CharField(read_only=True)

    class Meta:
        model = Subject
        fields = ['id', 'name', 'face_image', 'face_image_path', 'face_embedding']
        read_only_fields = ['id']

    def create(self, validated_data):
        image_file = validated_data.pop('face_image')
        filename = f"subject_images/{image_file.name}"

        with open(f"media/{filename}", 'wb+') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        validated_data['face_image_path'] = f"/media/{filename}"
        return Subject.objects.create(**validated_data)


class RecognitionLogSerializer(serializers.ModelSerializer):
    video_clip = serializers.FileField(write_only=True, required=True)
    image_path = serializers.CharField(read_only=True)

    class Meta:
        model = RecognitionLog
        fields = ['id', 'person_id', 'camera_id', 'time', 'confidence', 'video_clip', 'image_path']
        read_only_fields = ['id', 'image_path']

    def create(self, validated_data):
        video_file = validated_data.pop('video_file', None)
        if video_file:
            import os
            from django.conf import settings
            save_dir = os.path.join(settings.MEDIA_ROOT, 'recognition_videos')
            os.makedirs(save_dir, exist_ok=True)

            filename = os.path.join(save_dir, video_file.name)
            with open(filename, 'wb+') as f:
                for chunk in video_file.chunks():
                    f.write(chunk)

            relative_path = os.path.join('recognition_videos', video_file.name)
            validated_data['image_path'] = os.path.join(settings.MEDIA_URL, relative_path).replace('\\', '/')

        return RecognitionLog.objects.create(**validated_data)


class DetectionLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = DetectionLog
        fields = ['id', 'camera_id', 'object_class', 'confidence', 'bbox', 'time', 'image_path']
        read_only_fields = ['id']


class WarningZoneSerializer(serializers.ModelSerializer):
    video_clip = serializers.FileField(write_only=True, required=True)
    video_clip_path = serializers.CharField(read_only=True)

    class Meta:
        model = WarningZone
        fields = ['id', 'camera_id', 'name', 'zone_type', 'zone_points', 'is_active']
        read_only_fields = ['id']

    def create(self, validated_data):
        video_file = validated_data.pop('video_clip')
        filename = f"incident_videos/{video_file.name}"

        with open(f"media/{filename}", 'wb+') as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        validated_data['video_clip_path'] = f"/media/{filename}"

        return IncidentDetectionLog.objects.create(**validated_data)


class IncidentTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = IncidentType
        fields = ['id', 'name', 'code']
        read_only_fields = ['id']


class IncidentDetectionLogSerializer(serializers.ModelSerializer):

    class Meta:
        model = IncidentDetectionLog
        fields = ['id', 'incident_type_id', 'camera_id', 'time', 'video_clip_path', 'confidence', 'status']
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
