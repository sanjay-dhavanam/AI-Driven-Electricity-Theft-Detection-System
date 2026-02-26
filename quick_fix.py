#!/usr/bin/env python
import os
import sys
import django
import pickle
import shutil

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'electricity_app.settings')
django.setup()

from django.conf import settings
from core.models import Meter, ModelVersion
from django.utils import timezone

# Create media directories
os.makedirs(os.path.join(settings.MEDIA_ROOT, 'models'), exist_ok=True)
os.makedirs(os.path.join(settings.MEDIA_ROOT, 'scalers'), exist_ok=True)

# Get some meter IDs that have models
print("Creating dummy ModelVersion records for testing...")

# Take first 10 meters
meters = Meter.objects.all()[:10]

for meter in meters:
    # Skip if already has model
    if ModelVersion.objects.filter(meter=meter).exists():
        continue
    
    # Create a dummy model version
    ModelVersion.objects.create(
        meter=meter,
        version_number=1,
        model_name='IsolationForest',
        window_size=30,
        contamination=0.1,
        threshold_normal=0.0,
        threshold_theft=-0.1,
        training_samples=30,
        is_active=True,
        is_trained=True,
        trained_on=timezone.now().date(),
        training_logs={
            'note': 'Dummy model for testing',
            'created_for_testing': True,
        }
    )
    print(f"Created dummy model for {meter.meter_id}")

print(f"\nTotal ModelVersion records: {ModelVersion.objects.count()}")
print(f"Active models: {ModelVersion.objects.filter(is_active=True).count()}")