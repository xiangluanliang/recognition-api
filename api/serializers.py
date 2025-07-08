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
    class Meta:
        model = Subject
        fields = ['id', 'name', 'face_image_path', 'face_embedding']
        read_only_fields = ['id']


class RecognitionLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = RecognitionLog
        fields = ['id', 'person_id', 'camera_id', 'time', 'confidence', 'image_path']
        read_only_fields = ['id']


class DetectionLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = DetectionLog
        fields = ['id', 'camera_id', 'object_class', 'confidence', 'bbox', 'time', 'image_path']
        read_only_fields = ['id']


class WarningZoneSerializer(serializers.ModelSerializer):
    class Meta:
        model = WarningZone
        fields = ['id', 'camera_id', 'name', 'zone_type', 'zone_points', 'is_active']
        read_only_fields = ['id']


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
