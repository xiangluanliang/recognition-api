# api/admin.py
from django.contrib import admin
from .models import TestNumber

admin.site.register(TestNumber)