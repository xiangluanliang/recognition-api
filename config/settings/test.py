# test.py
from .base import *

# ========== 测试环境特定配置 ==========

# 开启DEBUG模式
DEBUG = True

ALLOWED_HOSTS = ['*']

# 允许所有来源的跨域请求，方便测试
CORS_ALLOW_ALL_ORIGINS = True

STATIC_URL = '/static/test/'
MEDIA_URL = '/media/test/'
