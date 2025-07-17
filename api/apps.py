from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        """
        当App准备就绪时，导入信号处理器模块。
        这是确保信号能被Django发现和注册的关键步骤。
        """
        import api.views.data_views
