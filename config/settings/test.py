# test.py

from .base import *


# 开启DEBUG模式，方便调试
DEBUG = True

ALLOWED_HOSTS = ['*']

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'rg_db',
        'USER': 'rg_user',
        'PASSWORD': 'Zyjjh0707',
        'HOST': '8.152.101.217',
        'PORT': '5432',
    }
}

CORS_ALLOW_ALL_ORIGINS = True