# production.py
from .base import *

# ========== 正式环境特定配置 ==========

# 关闭DEBUG模式
DEBUG = False

# 允许服务器公网IP和域名访问
# 建议也从环境变量读取，但为简化，先写死
ALLOWED_HOSTS = ['8.152.101.217']