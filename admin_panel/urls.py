# Update existing urls.py with new paths
from django.urls import path
from . import views

urlpatterns = [
        # Authentication paths
    path('login/', views.admin_login, name='admin_login'),
    path('logout/', views.admin_logout, name='admin_logout'),
    path('overview/', views.overview, name='admin_overview'),
    path('meter/<str:meter_id>/', views.meter_detail, name='meter_detail'),
    path('analytics/', views.analytics, name='admin_analytics'),
    path('retraining/', views.retraining_control, name='retraining_control'),
    
    # New paths for anomaly injection
    path('injections/', views.injection_management, name='injection_management'),
    path('injections/create/', views.create_injection, name='create_injection'),
    path('injections/<int:injection_id>/edit/', views.edit_injection, name='edit_injection'),
    path('injections/<int:injection_id>/delete/', views.delete_injection, name='delete_injection'),
    path('injections/preview/', views.injection_preview, name='injection_preview'),
    path('injections/report/', views.injection_impact_report, name='injection_report'),
        # Feedback Management
    path('feedback/', views.feedback_management, name='admin_feedback_management'),
    path('feedback/<int:feedback_id>/', views.feedback_detail, name='admin_feedback_detail'),
    path('feedback/analytics/', views.feedback_analytics, name='admin_feedback_analytics'),
    path('feedback/analytics/chart-data/', views.get_chart_data, name='admin_get_chart_data'),
    path('feedback/analytics/export/', views.feedback_analytics_export, name='admin_feedback_analytics_export'),    
    # Simple status update functions (with page reload)
    path('feedback/<int:feedback_id>/mark-pending/', views.admin_mark_pending, name='admin_mark_pending'),
    path('feedback/<int:feedback_id>/mark-investigation/', views.admin_mark_investigation, name='admin_mark_investigation'),
    path('feedback/<int:feedback_id>/mark-completed/', views.admin_mark_completed, name='admin_mark_completed'),
    path('feedback/<int:feedback_id>/mark-completed-response/', views.admin_mark_completed_with_response, name='admin_mark_completed_response'),
    
    # AJAX endpoint (optional)
    path('feedback/update-status-ajax/', views.update_feedback_status_ajax, name='admin_update_feedback_status_ajax'),
    # Enhanced adaptive learning
    path('enhanced-retraining/', views.enhanced_retraining_control, name='enhanced_retraining'),
    path('api/retraining/trigger/', views.trigger_retraining, name='trigger_retraining'),
    path('api/retraining/status/<int:job_id>/', views.retraining_status, name='retraining_status'),
    
    # Default
    path('api/injections/<int:injection_id>/details/', views.api_injection_details, name='api_injection_details'),
    path('api/meter-info/<str:meter_id>/', views.api_meter_info, name='api_meter_info'),
    path('api/injections/deactivate-expired/', views.api_deactivate_expired, name='api_deactivate_expired'),
    path('api/injections/bulk-deactivate/', views.api_bulk_deactivate, name='api_bulk_deactivate'),
]