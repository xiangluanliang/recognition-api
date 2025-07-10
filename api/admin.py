# api/admin.py
from django.contrib import admin

from .models import (
    User,
    OperationLog,
    Subject,
    WarningZone,
    EventLog,
    Camera,
    AlarmLog,
    TestNumber, 
    Feedback,
)

admin.site.register(User)
admin.site.register(OperationLog)
admin.site.register(Subject)
admin.site.register(WarningZone)
admin.site.register(EventLog)
admin.site.register(Camera)
admin.site.register(AlarmLog)

admin.site.register(TestNumber)
admin.site.register(Feedback) # 注册Feedback模型


