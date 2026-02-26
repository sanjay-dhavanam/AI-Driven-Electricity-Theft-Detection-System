from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User
import json
from datetime import timezone
from electricity_app import settings


class Meter(models.Model):
    """Represents a single electricity meter (household)"""
    meter_id = models.CharField(max_length=50, unique=True, primary_key=True)
    location = models.CharField(max_length=200, blank=True, null=True)
    acorn_group = models.CharField(max_length=50, blank=True, null=True)
    acorn_grouped = models.CharField(max_length=50, blank=True, null=True)
    stdorToU = models.CharField(max_length=10, blank=True, null=True)  # Standard or Time of Use
    customer_name = models.CharField(max_length=200, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    created_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)
    
    # Optional fields from household info
    file = models.CharField(max_length=100, blank=True, null=True)  # Original block file
    
    class Meta:
        ordering = ['meter_id']
        verbose_name = 'Meter'
        verbose_name_plural = 'Meters'
    
    def __str__(self):
        return f"{self.meter_id} - {self.location or 'No location'}"
    
from django.db import models

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Feedback(models.Model):
    STATUS_CHOICES = [
        ('pending', '⏳ Pending'),
        ('investigation', '🔍 Under Investigation'),
        ('completed', '✅ Completed'),
    ]

    FEEDBACK_TYPES = [
        ('general', 'General Feedback'),
        ('bug', 'Bug Report'),
        ('suggestion', 'Suggestion'),
        ('complaint', 'Complaint'),
        ('appreciation', 'Appreciation')
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='feedbacks'
    )

    meter = models.ForeignKey(
        Meter,
        on_delete=models.SET_NULL,
        related_name='feedbacks',
        null=True,
        blank=True
    )

    id_number = models.CharField(max_length=50)
    feedback_text = models.TextField()
    feedback_type = models.CharField(max_length=20, choices=FEEDBACK_TYPES, default='general')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    admin_response = models.TextField(blank=True, null=True)
    is_resolved = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Feedback'
        verbose_name_plural = 'Feedbacks'

    def __str__(self):
        return f"Feedback #{self.id} - {self.user.username}"

    # Add these methods for user side
    def get_status_display(self):
        """Get human-readable status"""
        status_map = dict(self.STATUS_CHOICES)
        return status_map.get(self.status, self.status.capitalize())

    def get_feedback_type_display(self):
        """Get human-readable feedback type"""
        type_map = dict(self.FEEDBACK_TYPES)
        return type_map.get(self.feedback_type, self.feedback_type.capitalize())

    def get_status_badge(self):
        """Return Bootstrap badge class based on status"""
        badge_map = {
            'pending': 'warning',
            'investigation': 'info',
            'completed': 'success',
        }
        return badge_map.get(self.status, 'secondary')

    def get_status_icon(self):
        """Return icon for status"""
        icon_map = {
            'pending': '⏳',
            'investigation': '🔍',
            'completed': '✅',
        }
        return icon_map.get(self.status, '❓')

    def has_admin_response(self):
        """Check if admin has responded"""
        return bool(self.admin_response)

    def get_days_since_submission(self):
        """Get number of days since feedback was submitted"""
        delta = timezone.now() - self.created_at
        return delta.days
    
    


class DailyConsumption(models.Model):
    """Stores daily consumption data with 48 half-hour readings"""
    meter = models.ForeignKey(Meter, on_delete=models.CASCADE, related_name='consumption')
    date = models.DateField()
    
    # Store 48 half-hour readings as JSON for flexibility
    consumption_data = models.JSONField(
        help_text='Dictionary with keys hh_0 to hh_47 and consumption values'
    )
    
    # Aggregated fields for faster querying
    total_consumption = models.FloatField(default=0.0, help_text='Sum of 48 half-hour readings')
    average_consumption = models.FloatField(default=0.0, help_text='Average of 48 readings')
    max_consumption = models.FloatField(default=0.0, help_text='Maximum half-hour reading')
    min_consumption = models.FloatField(default=0.0, help_text='Minimum half-hour reading')
    
    # Additional context
    is_holiday = models.BooleanField(default=False, help_text='Is this day a bank holiday?')
    temperature = models.FloatField(null=True, blank=True, help_text='Average temperature for the day')
    
    # Metadata
    created_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['meter', 'date']
        ordering = ['-date', 'meter']
        indexes = [
            models.Index(fields=['meter', 'date']),
            models.Index(fields=['date']),
            models.Index(fields=['total_consumption']),
        ]
        verbose_name = 'Daily Consumption'
        verbose_name_plural = 'Daily Consumptions'
    
    def __str__(self):
        return f"{self.meter.meter_id} - {self.date}"
    
    def save(self, *args, **kwargs):
        # Calculate aggregated fields from consumption_data
        if self.consumption_data:
            values = list(self.consumption_data.values())
            if values:
                self.total_consumption = sum(values)
                self.average_consumption = self.total_consumption / len(values)
                self.max_consumption = max(values)
                self.min_consumption = min(values)
        
        super().save(*args, **kwargs)
    
    def get_half_hour_readings(self):
        """Return half-hour readings in chronological order"""
        readings = []
        for i in range(48):
            key = f'hh_{i}'
            if key in self.consumption_data:
                readings.append({
                    'hour': i,
                    'time': f'{i//2:02d}:{(i%2)*30:02d}',
                    'consumption': self.consumption_data[key]
                })
        return readings

class ImportLog(models.Model):
    """Tracks CSV import operations"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('partial', 'Partial'),
    ]
    
    file_name = models.CharField(max_length=255)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    meters_imported = models.IntegerField(default=0)
    records_imported = models.IntegerField(default=0)
    errors = models.JSONField(default=list, blank=True)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    duration_seconds = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['-started_at']
        verbose_name = 'Import Log'
        verbose_name_plural = 'Import Logs'
    
    def __str__(self):
        return f"{self.file_name} - {self.status}"
    

# Add to existing models.py
class AnomalyResult(models.Model):
    """
    Stores daily anomaly detection results for each meter
    """
    CLASSIFICATION_CHOICES = [
        ('normal', 'Normal'),
        ('suspicious', 'Suspicious'),
        ('theft', 'Theft'),
    ]
    
    meter = models.ForeignKey(Meter, on_delete=models.CASCADE, related_name='anomaly_results')
    date = models.DateField()
    daily_consumption = models.ForeignKey(DailyConsumption, on_delete=models.CASCADE, 
                                         related_name='anomaly_results', null=True, blank=True)
    
    # ML Prediction Results
    anomaly_score = models.FloatField(help_text='Negative scores are more anomalous (Isolation Forest)')
    classification = models.CharField(max_length=20, choices=CLASSIFICATION_CHOICES)
    confidence = models.FloatField(default=0.0, help_text='Confidence in classification (0-1)')
    
    is_injected = models.BooleanField(default=False, help_text='Whether this result was manually injected')
    injection_reason = models.TextField(blank=True, help_text='Reason for injection if applicable')
    injection_source = models.ForeignKey('AnomalyInjection',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='applied_results'
    )

    # Model Thresholds (for transparency)
    threshold_normal = models.FloatField(help_text='Score > this = Normal')
    threshold_suspicious = models.FloatField(help_text='Score between thresholds = Suspicious')
    threshold_theft = models.FloatField(help_text='Score < this = Theft')
    
    # Additional Insights
    peak_hour = models.IntegerField(null=True, blank=True, help_text='Hour (0-47) with highest consumption')
    peak_consumption = models.FloatField(null=True, blank=True, help_text='Consumption at peak hour')
    total_deviation = models.FloatField(null=True, blank=True, help_text='Total deviation from expected pattern')
    
    # Model Metadata
    model_version = models.CharField(max_length=100, blank=True, null=True)
    model_name = models.CharField(max_length=100, default='IsolationForest')
    
    # Timestamps
    predicted_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['meter', 'date']
        ordering = ['-date', 'meter']
        indexes = [
            models.Index(fields=['meter', 'date']),
            models.Index(fields=['date']),
            models.Index(fields=['classification']),
            models.Index(fields=['anomaly_score']),
        ]
        verbose_name = 'Anomaly Result'
        verbose_name_plural = 'Anomaly Results'
    
    def __str__(self):
        return f"{self.meter.meter_id} - {self.date} - {self.classification}"
    
    def get_color_code(self):
        """Return color for visualization"""
        colors = {
            'normal': 'success',      # Green
            'suspicious': 'warning',  # Yellow
            'theft': 'danger',        # Red
        }
        return colors.get(self.classification, 'secondary')
    
    def get_icon(self):
        """Return icon for visualization"""
        icons = {
            'normal': '🟢',
            'suspicious': '🟡',
            'theft': '🔴',
        }
        return icons.get(self.classification, '⚪')
    
    def is_recent(self):
        """Check if prediction is recent (less than 24 hours)"""
        from django.utils.timezone import now
        return (now() - self.predicted_at).total_seconds() < 86400  # 24 hours
    
    def save(self, *args, **kwargs):
        # Check for injection before saving
        InjectionManager.apply_injection_to_result(self)
        super().save(*args, **kwargs)
    

# Add to existing models.py
class ModelVersion(models.Model):
    """
    Tracks different versions of ML models for each meter
    """
    meter = models.ForeignKey(Meter, on_delete=models.CASCADE, related_name='model_versions')
    version_number = models.IntegerField(default=1)
    model_name = models.CharField(max_length=100, default='IsolationForest')
    
    # File paths
    model_file = models.FileField(upload_to='models/', null=True, blank=True,
                                  help_text='Path to .pkl or .joblib file')
    scaler_file = models.FileField(upload_to='scalers/', null=True, blank=True,
                                   help_text='Path to scaler .pkl file')
    
    # Model parameters
    window_size = models.IntegerField(default=30, help_text='Days used for training')
    contamination = models.FloatField(default=0.1, help_text='Expected proportion of outliers')
    threshold_normal = models.FloatField(default=0.0)
    threshold_theft = models.FloatField(default=-0.1)
    
    # Performance metrics
    training_samples = models.IntegerField(default=0)
    validation_accuracy = models.FloatField(null=True, blank=True)
    validation_f1 = models.FloatField(null=True, blank=True)
    false_positive_rate = models.FloatField(null=True, blank=True)
    false_negative_rate = models.FloatField(null=True, blank=True)
    
    # Status
    is_active = models.BooleanField(default=False, help_text='Currently used for predictions')
    is_trained = models.BooleanField(default=False)
    
    # Metadata
    trained_on = models.DateField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    training_logs = models.JSONField(default=dict, blank=True)
    
    class Meta:
        unique_together = ['meter', 'version_number']
        ordering = ['meter', '-version_number']
        verbose_name = 'Model Version'
        verbose_name_plural = 'Model Versions'
    
    def __str__(self):
        status = "✓" if self.is_active else "✗"
        return f"{self.meter.meter_id} v{self.version_number} {status}"
    
    def get_absolute_url(self):
        return f"/admin/core/modelversion/{self.id}/"
    
    def activate(self):
        """Activate this model version and deactivate others for this meter"""
        # Deactivate all other versions for this meter
        ModelVersion.objects.filter(meter=self.meter).update(is_active=False)
        # Activate this one
        self.is_active = True
        self.save()
    
    def get_file_path(self, file_type='model'):
        """Get absolute file path"""
        import os
        from django.conf import settings
        
        if file_type == 'model' and self.model_file:
            return os.path.join(settings.MEDIA_ROOT, self.model_file.name)
        elif file_type == 'scaler' and self.scaler_file:
            return os.path.join(settings.MEDIA_ROOT, self.scaler_file.name)
        return None


# Add to existing models.py
class RetrainingLog(models.Model):
    """
    Tracks adaptive learning retraining events
    """
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('partial', 'Partial Success'),
    ]
    from electricity_app import settings
    # Basic info
    name = models.CharField(max_length=200, help_text='Description of retraining job')
    triggered_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, 
                                   null=True, blank=True, related_name='retraining_logs')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # Scope
    meters_processed = models.JSONField(default=list, blank=True, 
                                       help_text='List of meter IDs processed')
    meters_success = models.JSONField(default=list, blank=True, 
                                     help_text='List of meter IDs where retraining succeeded')
    meters_failed = models.JSONField(default=list, blank=True, 
                                    help_text='List of meter IDs where retraining failed')
    meters_improved = models.JSONField(default=list, blank=True, 
                                      help_text='List of meter IDs where performance improved ≥5%')
    meters_unchanged = models.JSONField(default=list, blank=True, 
                                       help_text='List of meter IDs where performance improved <5%')
    
    # Performance metrics
    total_meters = models.IntegerField(default=0)
    success_count = models.IntegerField(default=0)
    failed_count = models.IntegerField(default=0)
    improved_count = models.IntegerField(default=0)
    unchanged_count = models.IntegerField(default=0)
    average_improvement = models.FloatField(default=0.0, 
                                           help_text='Average performance improvement across all meters')
    
    # Model settings
    window_days = models.IntegerField(default=30, help_text='Days used for training')
    validation_days = models.IntegerField(default=7, help_text='Days used for validation')
    improvement_threshold = models.FloatField(default=5.0, 
                                             help_text='Minimum improvement required to replace model (%)')
    
    # Detailed logs
    detailed_logs = models.JSONField(default=list, blank=True, 
                                    help_text='Detailed logs for each meter')
    error_logs = models.TextField(blank=True, help_text='Error messages if any')
    
    # Timestamps
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    duration_seconds = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-started_at']
        verbose_name = 'Retraining Log'
        verbose_name_plural = 'Retraining Logs'
    
    def __str__(self):
        return f"Retraining {self.started_at.date() if self.started_at else 'Pending'} - {self.status}"
    
    def get_absolute_url(self):
        return f"/admin/core/retraininglog/{self.id}/"
    
    def get_success_rate(self):
        if self.total_meters == 0:
            return 0
        return (self.success_count / self.total_meters) * 100
    
    def get_improvement_rate(self):
        if self.total_meters == 0:
            return 0
        return (self.improved_count / self.total_meters) * 100
    
    def get_duration_minutes(self):
        if self.duration_seconds:
            return self.duration_seconds / 60
        return None
    
    def get_status_color(self):
        colors = {
            'pending': 'warning',
            'running': 'info',
            'completed': 'success',
            'failed': 'danger',
            'partial': 'warning',
        }
        return colors.get(self.status, 'secondary')
    


    # Add to existing models.py after other models
import random
from django.contrib.auth.models import User

class UserProfile(models.Model):
    """
    Links Django User to a Meter
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    meter = models.ForeignKey(Meter, on_delete=models.CASCADE, related_name='user_profiles')
    
    # Simulation settings
    simulation_speed = models.IntegerField(default=1, help_text='How many half-hours to advance per page load')
    current_pointer = models.IntegerField(default=0, help_text='Current position in historical data')
    is_simulation_active = models.BooleanField(default=True)
    
    # UI preferences
    chart_theme = models.CharField(max_length=20, default='light', choices=[
        ('light', 'Light'),
        ('dark', 'Dark'),
        ('colorblind', 'Colorblind Friendly')
    ])
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'
    
    def __str__(self):
        return f"{self.user.username} -> {self.meter.meter_id}"
    
    def get_next_reading(self):
        """
        Get the next half-hour reading for simulation
        Returns: (consumption_value, hour_index, date)
        """
        # Get all consumption days for this meter
        consumptions = DailyConsumption.objects.filter(meter=self.meter).order_by('date')
        
        if not consumptions.exists():
            return None
        
        # Calculate which day and which half-hour we're at
        total_half_hours = consumptions.count() * 48
        
        if self.current_pointer >= total_half_hours:
            # Reset to beginning if we've gone through all data
            self.current_pointer = 0
            self.save()
        
        day_index = self.current_pointer // 48
        half_hour_index = self.current_pointer % 48
        
        if day_index >= len(consumptions):
            day_index = 0
            self.current_pointer = 0
            self.save()
        
        consumption_day = consumptions[day_index]
        consumption_data = consumption_day.consumption_data
        
        # Get the consumption value
        key = f'hh_{half_hour_index}'
        consumption_value = consumption_data.get(key, 0.0)
        
        # Update pointer for next time
        self.current_pointer += self.simulation_speed
        self.save()
        
        return {
            'consumption': consumption_value,
            'half_hour_index': half_hour_index,
            'hour': half_hour_index // 2,
            'minute': (half_hour_index % 2) * 30,
            'date': consumption_day.date,
            'day_index': day_index,
            'total_half_hours': total_half_hours,
            'progress_percentage': (self.current_pointer / total_half_hours * 100) if total_half_hours > 0 else 0
        }

class HalfHourReading(models.Model):
    """
    Stores individual half-hour readings for simulation
    """
    meter = models.ForeignKey(Meter, on_delete=models.CASCADE, related_name='half_hour_readings')
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='readings', null=True, blank=True)
    timestamp = models.DateTimeField()
    consumption = models.FloatField()
    hour = models.IntegerField(help_text='Hour of day (0-23)')
    minute = models.IntegerField(help_text='Minute (0 or 30)')
    
    # Link to anomaly result if available
    anomaly_result = models.ForeignKey(AnomalyResult, on_delete=models.SET_NULL, null=True, blank=True, 
                                      related_name='half_hour_readings')
    
    # Calculated fields
    is_peak = models.BooleanField(default=False)
    is_off_peak = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['meter', 'timestamp']),
            models.Index(fields=['timestamp']),
        ]
        verbose_name = 'Half Hour Reading'
        verbose_name_plural = 'Half Hour Readings'
    
    def __str__(self):
        time_str = f"{self.hour:02d}:{self.minute:02d}"
        return f"{self.meter.meter_id} - {self.timestamp.date()} {time_str} - {self.consumption:.3f} kWh"
    
    def get_time_label(self):
        """Get formatted time label"""
        return f"{self.hour:02d}:{self.minute:02d}"
    
    def get_color_code(self):
        """Get color based on anomaly result"""
        if self.anomaly_result:
            return self.anomaly_result.get_color_code()
        return 'secondary'
    
# Add to existing models.py
class AnomalyInjection(models.Model):
    """
    Manual injection of anomaly results (admin override of ML predictions)
    """
    INJECTION_TYPE_CHOICES = [
        ('normal', 'Normal'),
        ('suspicious', 'Suspicious'),
        ('theft', 'Theft'),
    ]
    
    TIME_SLOT_CHOICES = [
        ('full_day', 'Full Day'),
        ('00:00', '00:00-00:30'), ('00:30', '00:30-01:00'),
        ('01:00', '01:00-01:30'), ('01:30', '01:30-02:00'),
        ('02:00', '02:00-02:30'), ('02:30', '02:30-03:00'),
        ('03:00', '03:00-03:30'), ('03:30', '03:30-04:00'),
        ('04:00', '04:00-04:30'), ('04:30', '04:30-05:00'),
        ('05:00', '05:00-05:30'), ('05:30', '05:30-06:00'),
        ('06:00', '06:00-06:30'), ('06:30', '06:30-07:00'),
        ('07:00', '07:00-07:30'), ('07:30', '07:30-08:00'),
        ('08:00', '08:00-08:30'), ('08:30', '08:30-09:00'),
        ('09:00', '09:00-09:30'), ('09:30', '09:30-10:00'),
        ('10:00', '10:00-10:30'), ('10:30', '10:30-11:00'),
        ('11:00', '11:00-11:30'), ('11:30', '11:30-12:00'),
        ('12:00', '12:00-12:30'), ('12:30', '12:30-13:00'),
        ('13:00', '13:00-13:30'), ('13:30', '13:30-14:00'),
        ('14:00', '14:00-14:30'), ('14:30', '14:30-15:00'),
        ('15:00', '15:00-15:30'), ('15:30', '15:30-16:00'),
        ('16:00', '16:00-16:30'), ('16:30', '16:30-17:00'),
        ('17:00', '17:00-17:30'), ('17:30', '17:30-18:00'),
        ('18:00', '18:00-18:30'), ('18:30', '18:30-19:00'),
        ('19:00', '19:00-19:30'), ('19:30', '19:30-20:00'),
        ('20:00', '20:00-20:30'), ('20:30', '20:30-21:00'),
        ('21:00', '21:00-21:30'), ('21:30', '21:30-22:00'),
        ('22:00', '22:00-22:30'), ('22:30', '22:30-23:00'),
        ('23:00', '23:00-23:30'), ('23:30', '23:30-00:00'),
    ]
    
    # Basic information
    meter = models.ForeignKey(Meter, on_delete=models.CASCADE, related_name='injections')
    date = models.DateField()
    
    # Time selection
    time_slot = models.CharField(
        max_length=10,
        choices=TIME_SLOT_CHOICES,
        default='full_day',
        help_text='Specific 30-minute slot or full day'
    )
    
    # Injection details
    injection_type = models.CharField(
        max_length=20,
        choices=INJECTION_TYPE_CHOICES,
        help_text='Classification to inject'
    )
    
    # Anomaly parameters (for injection)
    injected_score = models.FloatField(
        default=0.0,
        help_text='Anomaly score to inject (use typical values: Normal>0.1, Suspicious=0.0, Theft<-0.1)'
    )
    injected_confidence = models.FloatField(
        default=0.95,
        help_text='Confidence for injection (0-1)'
    )
    
    # Override consumption data for specific time slot
    consumption_override = models.JSONField(
        default=dict,
        blank=True,
        help_text='Override consumption for the time slot (e.g., {"value": 2.5})'
    )
    
    # Metadata
    reason = models.TextField(
        blank=True,
        help_text='Reason for injection (e.g., "Scheduled maintenance", "Known theft case")'
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name='created_injections'
    )
    is_active = models.BooleanField(default=True, help_text='Whether this injection is active')
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    expires_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text='Automatic deactivation date (optional)'
    )
    
    class Meta:
        unique_together = ['meter', 'date', 'time_slot']
        ordering = ['-date', 'meter', 'time_slot']
        verbose_name = 'Anomaly Injection'
        verbose_name_plural = 'Anomaly Injections'
    
    def __str__(self):
        return f"{self.meter.meter_id} - {self.date} {self.get_time_slot_display()} - {self.injection_type}"
    
    def is_expired(self):
        if self.expires_at and timezone.now() > self.expires_at:
            return True
        return False
    
    def get_detailed_info(self):
        """Get detailed information for display"""
        return {
            'meter_id': self.meter.meter_id,
            'date': self.date,
            'time_slot': self.get_time_slot_display(),
            'injection_type': self.get_injection_type_display(),
            'score': self.injected_score,
            'confidence': self.injected_confidence,
            'reason': self.reason,
            'created_by': self.created_by.username if self.created_by else 'System',
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M'),
            'is_active': self.is_active,
            'is_expired': self.is_expired(),
        }
    def save(self, *args, **kwargs):
        # Auto-deactivate if expired
        if self.expires_at and timezone.now() > self.expires_at:
            self.is_active = False
        
        # Set reasonable scores based on injection type
        if not self.injected_score:
            if self.injection_type == 'normal':
                self.injected_score = 0.15  # Positive score for normal
            elif self.injection_type == 'suspicious':
                self.injected_score = 0.0   # Borderline score
            else:  # theft
                self.injected_score = -0.2  # Negative score for theft
        
        super().save(*args, **kwargs)


# Add this class to models.py (after Injection model)
class InjectionManager:
    """
    Manages anomaly injections and their integration with predictions
    """
    
    @staticmethod
    def get_injection_for_result(meter_id, date, time_slot='full_day'):
        """
        Get active injection for a specific meter, date, and time slot
        Returns: Injection object or None
        """
        try:
            injection = AnomalyInjection.objects.filter(
                meter__meter_id=meter_id,
                date=date,
                time_slot=time_slot,
                is_active=True
            ).first()
            
            if injection and injection.is_expired():
                injection.is_active = False
                injection.save()
                return None
            
            return injection
        except:
            return None
    
    @staticmethod
    def apply_injection_to_result(anomaly_result):
        """
        Apply injection to an anomaly result if exists
        Modifies the anomaly_result in place
        Returns: Boolean (True if injected, False if not)
        """
        # Check for full-day injection first
        injection = InjectionManager.get_injection_for_result(
            anomaly_result.meter.meter_id,
            anomaly_result.date
        )
        
        if not injection:
            return False
        
        # Apply injection
        anomaly_result.classification = injection.injection_type
        anomaly_result.anomaly_score = injection.injected_score
        anomaly_result.confidence = injection.injected_confidence
        anomaly_result.is_injected = True
        anomaly_result.injection_reason = injection.reason
        anomaly_result.injection_source = injection
        
        return True
    
    @staticmethod
    def apply_injection_to_result(anomaly_result):
        """
        Apply injection to an anomaly result if exists
        Modifies the anomaly_result in place
        Returns: Boolean (True if injected, False if not)
        """
        # Check for full-day injection first
        injection = InjectionManager.get_injection_for_result(
            anomaly_result.meter.meter_id,
            anomaly_result.date
        )
        
        if not injection:
            return False
        
        # Apply injection
        anomaly_result.classification = injection.injection_type
        anomaly_result.anomaly_score = injection.injected_score
        anomaly_result.confidence = injection.injected_confidence
        anomaly_result.is_injected = True
        anomaly_result.injection_reason = injection.reason
        
        return True
    
    @staticmethod
    def get_injections_for_meter(meter_id, start_date=None, end_date=None):
        """
        Get all injections for a meter within date range
        """
        filters = {'meter__meter_id': meter_id}
        
        if start_date:
            filters['date__gte'] = start_date
        if end_date:
            filters['date__lte'] = end_date
        
        return AnomalyInjection.objects.filter(**filters).order_by('-date')
    
    @staticmethod
    def get_active_injection_count():
        """Get count of active injections"""
        return AnomalyInjection.objects.filter(is_active=True).count()
    
    @staticmethod
    def deactivate_expired_injections():
        """Deactivate all expired injections"""
        expired = AnomalyInjection.objects.filter(
            expires_at__lt=timezone.now(),
            is_active=True
        )
        count = expired.count()
        expired.update(is_active=False)
        return count
    
    @staticmethod
    def get_anomaly_with_injection(meter_id, date):
        """
        Get anomaly result with injection applied.
        Returns a dict with anomaly data (including injection if exists)
        """
        try:
            # Get the meter
            meter = Meter.objects.get(meter_id=meter_id)
            
            # Get anomaly result if exists
            anomaly_result = None
            try:
                anomaly_result = AnomalyResult.objects.get(
                    meter=meter,
                    date=date
                )
            except AnomalyResult.DoesNotExist:
                pass
            
            # Get injection if exists
            injection = InjectionManager.get_injection_for_result(meter_id, date)
            
            # If injection exists, use injection values
            if injection:
                return {
                    'date': date,
                    'classification': injection.injection_type,
                    'anomaly_score': injection.injected_score,
                    'confidence': injection.injected_confidence,
                    'is_injected': True,
                    'injection_reason': injection.reason,
                    'meter_id': meter_id,
                    'exists': True,
                    'source': 'injection'
                }
            # If no injection but anomaly exists
            elif anomaly_result:
                return {
                    'date': date,
                    'classification': anomaly_result.classification,
                    'anomaly_score': anomaly_result.anomaly_score,
                    'confidence': anomaly_result.confidence,
                    'is_injected': anomaly_result.is_injected,
                    'injection_reason': anomaly_result.injection_reason,
                    'meter_id': meter_id,
                    'exists': True,
                    'source': 'prediction'
                }
            # Neither injection nor anomaly
            else:
                return None
                
        except Meter.DoesNotExist:
            return None
        except Exception as e:
            print(f"Error in get_anomaly_with_injection: {e}")
            return None
    
    @staticmethod
    def get_anomalies_for_meter_with_injections(meter_id, start_date=None, end_date=None):
        """
        Get all anomaly results for a meter, with injections applied.
        Includes synthetic results for dates with injections but no predictions.
        """
        try:
            meter = Meter.objects.get(meter_id=meter_id)
            
            # Get all anomaly results for this meter
            anomaly_results = AnomalyResult.objects.filter(meter=meter)
            
            if start_date:
                anomaly_results = anomaly_results.filter(date__gte=start_date)
            if end_date:
                anomaly_results = anomaly_results.filter(date__lte=end_date)
            
            # Get all injections for this meter
            injections = AnomalyInjection.objects.filter(
                meter=meter,
                is_active=True
            )
            if start_date:
                injections = injections.filter(date__gte=start_date)
            if end_date:
                injections = injections.filter(date__lte=end_date)
            
            # Create a combined list
            combined_results = []
            
            # First, get all dates from anomaly results
            anomaly_dates = set()
            for result in anomaly_results:
                anomaly_dates.add(result.date)
                # Apply injection if exists
                injection = InjectionManager.get_injection_for_result(meter_id, result.date)
                if injection:
                    combined_results.append({
                        'date': result.date,
                        'classification': injection.injection_type,
                        'anomaly_score': injection.injected_score,
                        'confidence': injection.injected_confidence,
                        'is_injected': True,
                        'injection_reason': injection.reason,
                        'source': 'injection_override'
                    })
                else:
                    combined_results.append({
                        'date': result.date,
                        'classification': result.classification,
                        'anomaly_score': result.anomaly_score,
                        'confidence': result.confidence,
                        'is_injected': result.is_injected,
                        'injection_reason': result.injection_reason,
                        'source': 'prediction'
                    })
            
            # Add injections that don't have corresponding anomaly results
            for injection in injections:
                if injection.date not in anomaly_dates:
                    combined_results.append({
                        'date': injection.date,
                        'classification': injection.injection_type,
                        'anomaly_score': injection.injected_score,
                        'confidence': injection.injected_confidence,
                        'is_injected': True,
                        'injection_reason': injection.reason,
                        'source': 'injection_only'
                    })
            
            # Sort by date descending
            combined_results.sort(key=lambda x: x['date'], reverse=True)
            
            return combined_results
            
        except Exception as e:
            print(f"Error in get_anomalies_for_meter_with_injections: {e}")
            return []

