from django.contrib.auth.models import User
from django.db import models

class TestNumber(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)
    number = models.IntegerField(default=0)
