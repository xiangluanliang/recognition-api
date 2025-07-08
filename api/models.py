# recognition-api/api/models.py

from django.contrib.auth.models import User 
from django.db import models              

# TestNumber 模型 
class TestNumber(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)
    number = models.IntegerField(default=0)

#  Feedback 模型 
class Feedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='feedbacks')
    title = models.CharField(max_length=100)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self): 
        return f"'{self.title}' by {self.user.username}" 


