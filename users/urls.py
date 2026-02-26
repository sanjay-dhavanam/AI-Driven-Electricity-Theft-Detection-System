"""
URL Configuration for User Portal
"""

from django.urls import path
from . import views

app_name = 'users'

urlpatterns = [
    # Authentication
    path('', views.custom_login, name='login'),
    path('logout/', views.custom_logout, name='logout'),
    
    # Main pages
    path('dashboard/', views.dashboard, name='dashboard'),
    path('daily/', views.daily_view, name='daily'),
    path('daily/<str:date_str>/', views.daily_view, name='daily_with_date'),
    path('timeline/', views.anomaly_timeline, name='timeline'),
    path('settings/', views.settings, name='settings'),
    path('feedback/', views.feedback_view, name='feedback'),
    path('feedback/history/', views.feedback_history, name='feedback_history'),
    path('feedback/<int:feedback_id>/', views.feedback_detail, name='feedback_detail'),
    path('feedback/export/<str:format>/', views.feedback_export, name='feedback_export'),
    # API endpoints (AJAX)
    path('api/latest/', views.api_latest_reading, name='api_latest'),
    path('api/daily/<str:date_str>/', views.api_daily_data, name='api_daily'),
]