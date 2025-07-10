# api/admin.py
from django.contrib import admin
from .models import (
    User,
    OperationLog,
    Subject,
    RecognitionLog,
    DetectionLog,
    WarningZone,
    IncidentType,
    IncidentDetectionLog,
    Camera,
    AlarmLog,
    VideoAnalysisTask
)

admin.site.register(User)
admin.site.register(OperationLog)
admin.site.register(Subject)
admin.site.register(RecognitionLog)
admin.site.register(DetectionLog)
admin.site.register(WarningZone)
admin.site.register(IncidentType)
admin.site.register(IncidentDetectionLog)
admin.site.register(Camera)
admin.site.register(AlarmLog)
admin.site.register(VideoAnalysisTask)

