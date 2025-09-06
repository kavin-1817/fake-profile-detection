from django.urls import path
from . import views

app_name = 'detection_app'

urlpatterns = [
    # Main pages
    path('', views.index, name='index'),
    path('result/<int:profile_id>/', views.result, name='result'),
    
    # Admin pages
    path('admin/dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin/flag/<int:profile_id>/', views.flag_profile, name='flag_profile'),
    path('admin/resolve/<int:flag_id>/<str:status>/', views.resolve_flag, name='resolve_flag'),
    
    # API endpoints
    path('api/analytics/', views.analytics_api, name='analytics_api'),
    path('api/model-status/', views.model_status, name='model_status'),
]
