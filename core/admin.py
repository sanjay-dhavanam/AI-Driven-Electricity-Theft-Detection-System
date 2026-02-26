from django.contrib import admin

# Register your models here.
from django.contrib import admin
from django.utils.html import format_html
from .models import Meter, DailyConsumption, ImportLog, AnomalyResult, ModelVersion, RetrainingLog, UserProfile, HalfHourReading
from core.adaptive_learning import AdaptiveLearningEngine

@admin.register(Meter)
class MeterAdmin(admin.ModelAdmin):
    list_display = ('meter_id', 'location', 'acorn_group', 'is_active', 'created_date')
    list_filter = ('is_active', 'acorn_group', 'stdorToU')
    search_fields = ('meter_id', 'location', 'customer_name')
    readonly_fields = ('created_date', 'updated_date')
    fieldsets = (
        ('Basic Information', {
            'fields': ('meter_id', 'location', 'customer_name', 'is_active')
        }),
        ('ACORN Details', {
            'fields': ('acorn_group', 'acorn_grouped'),
            'classes': ('collapse',)
        }),
        ('Tariff Information', {
            'fields': ('stdorToU', 'file'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_date', 'updated_date'),
            'classes': ('collapse',)
        }),
    )

@admin.register(DailyConsumption)
class DailyConsumptionAdmin(admin.ModelAdmin):
    list_display = ('meter', 'date', 'total_consumption', 'average_consumption', 'is_holiday')
    list_filter = ('date', 'is_holiday', 'meter__acorn_group')
    search_fields = ('meter__meter_id',)
    readonly_fields = ('created_date', 'updated_date', 'get_consumption_preview')
    date_hierarchy = 'date'
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('meter', 'date', 'is_holiday', 'temperature')
        }),
        ('Consumption Data', {
            'fields': ('total_consumption', 'average_consumption', 'max_consumption', 'min_consumption')
        }),
        ('Consumption Preview', {
            'fields': ('get_consumption_preview',),
            'classes': ('collapse',)
        }),
        ('Full Consumption Data', {
            'fields': ('consumption_data',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_date', 'updated_date'),
            'classes': ('collapse',)
        }),
    )
    
    def get_consumption_preview(self, obj):
        if obj.consumption_data:
            preview = {}
            # Show first 5 and last 5 half-hour readings
            for i in range(5):
                key = f'hh_{i}'
                if key in obj.consumption_data:
                    preview[key] = obj.consumption_data[key]
            
            preview['...'] = '...'
            
            for i in range(43, 48):
                key = f'hh_{i}'
                if key in obj.consumption_data:
                    preview[key] = obj.consumption_data[key]
            
            return format_html('<pre>{}</pre>', str(preview))
        return "No consumption data"
    
    get_consumption_preview.short_description = 'Consumption Preview'

@admin.register(ImportLog)
class ImportLogAdmin(admin.ModelAdmin):
    list_display = ('file_name', 'status', 'meters_imported', 'records_imported', 'started_at', 'duration_seconds')
    list_filter = ('status',)
    readonly_fields = ('started_at', 'completed_at', 'duration_seconds', 'get_errors_display')
    fieldsets = (
        ('Import Details', {
            'fields': ('file_name', 'status', 'meters_imported', 'records_imported')
        }),
        ('Timing', {
            'fields': ('started_at', 'completed_at', 'duration_seconds'),
            'classes': ('collapse',)
        }),
        ('Errors', {
            'fields': ('get_errors_display',),
            'classes': ('collapse',)
        }),
    )
    
    def get_errors_display(self, obj):
        if obj.errors:
            return format_html('<pre>{}</pre>', str(obj.errors))
        return "No errors"
    
    get_errors_display.short_description = 'Errors'


# Add to existing admin.py
@admin.register(AnomalyResult)
class AnomalyResultAdmin(admin.ModelAdmin):
    list_display = ('meter', 'date', 'classification', 'anomaly_score', 
                    'confidence', 'predicted_at')
    list_filter = ('classification', 'date', 'meter__acorn_group')
    search_fields = ('meter__meter_id',)
    readonly_fields = ('predicted_at', 'updated_at', 'get_color_display')
    date_hierarchy = 'date'
    list_per_page = 50
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('meter', 'date', 'daily_consumption')
        }),
        ('Prediction Results', {
            'fields': ('classification', 'anomaly_score', 'confidence', 'get_color_display')
        }),
        ('Thresholds', {
            'fields': ('threshold_normal', 'threshold_suspicious', 'threshold_theft'),
            'classes': ('collapse',)
        }),
        ('Insights', {
            'fields': ('peak_hour', 'peak_consumption', 'total_deviation'),
            'classes': ('collapse',)
        }),
        ('Model Information', {
            'fields': ('model_version', 'model_name'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('predicted_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def get_color_display(self, obj):
        color = obj.get_color_code()
        icon = obj.get_icon()
        return format_html(
            '<span style="color: {}; font-size: 1.5em;">{} {}</span>',
            'green' if color == 'success' else 'red' if color == 'danger' else 'orange',
            icon,
            obj.get_classification_display()
        )
    
    get_color_display.short_description = 'Status Display'

@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ('meter', 'version_number', 'is_active', 'is_trained', 
                    'trained_on', 'window_size', 'training_samples')
    list_filter = ('is_active', 'is_trained', 'model_name')
    search_fields = ('meter__meter_id',)
    readonly_fields = ('created_at', 'updated_at')
    actions = ['activate_selected']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('meter', 'version_number', 'model_name', 'is_active', 'is_trained')
        }),
        ('Files', {
            'fields': ('model_file', 'scaler_file'),
            'classes': ('collapse',)
        }),
        ('Model Parameters', {
            'fields': ('window_size', 'contamination', 'threshold_normal', 'threshold_theft'),
            'classes': ('collapse',)
        }),
        ('Performance Metrics', {
            'fields': ('training_samples', 'validation_accuracy', 'validation_f1',
                      'false_positive_rate', 'false_negative_rate'),
            'classes': ('collapse',)
        }),
        ('Training Information', {
            'fields': ('trained_on', 'training_logs'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def activate_selected(self, request, queryset):
        """Admin action to activate selected model versions"""
        for model_version in queryset:
            model_version.activate()
        self.message_user(request, f"{queryset.count()} model(s) activated.")
    
    activate_selected.short_description = "Activate selected models"


# Update the RetrainingLogAdmin class in core/admin.py
@admin.register(RetrainingLog)
class RetrainingLogAdmin(admin.ModelAdmin):
    list_display = ('name', 'status', 'total_meters', 'success_count', 
                   'improved_count', 'started_at', 'duration_display')
    list_filter = ('status', 'started_at')
    readonly_fields = ('started_at', 'completed_at', 'duration_seconds',
                      'success_rate_display', 'improvement_rate_display', 
                      'status_display', 'meters_lists_display')
    date_hierarchy = 'started_at'
    list_per_page = 20
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'triggered_by', 'status')
        }),
        ('Scope & Settings', {
            'fields': ('window_days', 'validation_days', 'improvement_threshold')
        }),
        ('Results', {
            'fields': ('total_meters', 'success_count', 'failed_count', 
                      'improved_count', 'unchanged_count', 'average_improvement')
        }),
        ('Timing', {
            'fields': ('started_at', 'completed_at', 'duration_seconds'),
            'classes': ('collapse',)
        }),
        ('Logs', {
            'fields': ('detailed_logs', 'error_logs'),
            'classes': ('collapse',)
        }),
    )
    
    actions = ['rerun_selected_jobs']
    
    def duration_display(self, obj):
        if obj.duration_seconds:
            minutes = obj.duration_seconds / 60
            return f"{minutes:.1f} min"
        return "N/A"
    
    duration_display.short_description = 'Duration'
    
    def status_display(self, obj):
        color = obj.get_status_color()
        status_display = obj.get_status_display()
        
        colors = {
            'success': 'green',
            'warning': 'orange',
            'info': 'blue',
            'danger': 'red',
        }
        
        return format_html(
            '<span style="color: {}; font-weight: bold;">● {}</span>',
            colors.get(color, 'black'),
            status_display
        )
    
    status_display.short_description = 'Status'
    
    def success_rate_display(self, obj):
        return f"{obj.get_success_rate():.1f}%"
    
    success_rate_display.short_description = 'Success Rate'
    
    def improvement_rate_display(self, obj):
        return f"{obj.get_improvement_rate():.1f}%"
    
    improvement_rate_display.short_description = 'Improvement Rate'
    
    def meters_lists_display(self, obj):
        html = []
        
        if obj.meters_improved:
            html.append(f"<h4>Improved ({len(obj.meters_improved)}):</h4>")
            html.append(f"<p>{', '.join(obj.meters_improved[:10])}")
            if len(obj.meters_improved) > 10:
                html.append(f"<br>... and {len(obj.meters_improved) - 10} more</p>")
        
        if obj.meters_unchanged:
            html.append(f"<h4>Unchanged ({len(obj.meters_unchanged)}):</h4>")
            html.append(f"<p>{', '.join(obj.meters_unchanged[:10])}")
            if len(obj.meters_unchanged) > 10:
                html.append(f"<br>... and {len(obj.meters_unchanged) - 10} more</p>")
        
        if obj.meters_failed:
            html.append(f"<h4>Failed ({len(obj.meters_failed)}):</h4>")
            html.append(f"<p>{', '.join(obj.meters_failed[:10])}")
            if len(obj.meters_failed) > 10:
                html.append(f"<br>... and {len(obj.meters_failed) - 10} more</p>")
        
        return format_html(''.join(html)) if html else "No data"
    
    meters_lists_display.short_description = 'Meter Results'



# Add to existing admin.py
@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'meter', 'simulation_speed', 'current_pointer', 'is_simulation_active')
    list_filter = ('is_simulation_active', 'chart_theme')
    search_fields = ('user__username', 'meter__meter_id')
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        ('User Information', {
            'fields': ('user', 'meter')
        }),
        ('Simulation Settings', {
            'fields': ('simulation_speed', 'current_pointer', 'is_simulation_active')
        }),
        ('UI Preferences', {
            'fields': ('chart_theme',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

@admin.register(HalfHourReading)
class HalfHourReadingAdmin(admin.ModelAdmin):
    list_display = ('meter', 'timestamp', 'consumption', 'get_time_label', 'anomaly_result')
    list_filter = ('timestamp', 'meter__acorn_group', 'is_peak', 'is_off_peak')
    search_fields = ('meter__meter_id',)
    date_hierarchy = 'timestamp'
    readonly_fields = ('created_at',)
    
    fieldsets = (
        ('Reading Information', {
            'fields': ('meter', 'user_profile', 'timestamp', 'consumption', 'hour', 'minute')
        }),
        ('Analysis', {
            'fields': ('anomaly_result', 'is_peak', 'is_off_peak')
        }),
        ('Metadata', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
    
    def get_time_label(self, obj):
        return obj.get_time_label()
    
    get_time_label.short_description = 'Time'


