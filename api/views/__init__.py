# api/views/__init__.py

# 从各个文件中导入你的视图类
from .data_views import DoubleNumberView

# 你也可以使用 __all__ 来明确指定哪些可以被外部导入
__all__ = [
    'DoubleNumberView',
]