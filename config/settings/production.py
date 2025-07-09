# production.py

# 从base.py中导入所有通用配置
from .base import *

# ========== 正式环境特定配置 ==========

# 关闭DEBUG模式，增强安全性
DEBUG = False

# 允许你的服务器公网IP和域名访问
ALLOWED_HOSTS = ['8.152.101.217']

# 数据库配置（可以保持不变，使用base.py中的，或者在这里定义独立的生产数据库）
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'rg_db_production',
        'USER': 'rg_user',
        'PASSWORD': 'Zyjjh0707',
        'HOST': '8.152.101.217',
        'PORT': '5432',
    }
}