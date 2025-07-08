# config/urls.py

from django.contrib import admin
from django.urls import path, include
from rest_framework import permissions
from rest_framework.authtoken.views import obtain_auth_token
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

# --- Swagger/OpenAPI 配置 ---
schema_view = get_schema_view(
    openapi.Info(
        title="Recognition API",
        default_version='v1',
        description="API documentation for the recognition project",
        contact=openapi.Contact(email="contact@yourdomain.com"),
        license=openapi.License(name="MIT License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    # Django Admin 路由
    path('admin/', admin.site.urls),

    # API 路由，确保连接到 api.urls
    path('api/', include('api.urls')),

    # Swagger UI 路由
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),

    # ReDoc UI 路由
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),

    # Token 认证 API 路由
    path('api/token-auth/', obtain_auth_token, name='api_token_auth'),
]
