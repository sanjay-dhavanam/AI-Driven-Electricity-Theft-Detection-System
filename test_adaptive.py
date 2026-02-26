#!/usr/bin/env python
import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'electricity_app.settings')
django.setup()

from core.adaptive_learning import AdaptiveLearningManager, RetrainingCommands
from core.models import Meter, ModelVersion, DailyConsumption
from django.utils import timezone
from datetime import timedelta

print("="*60)
print("COMPLETE SYSTEM CHECK")
print("="*60)

# 1. Check database
print("\n1. DATABASE STATUS:")
print(f"Total meters: {Meter.objects.count()}")
print(f"Daily consumption records: {DailyConsumption.objects.count()}")
print(f"ModelVersion records: {ModelVersion.objects.count()}")
print(f"Active models: {ModelVersion.objects.filter(is_active=True).count()}")

# 2. Check specific meters
test_meters = ['MAC000002', 'MAC000048', 'MAC000154']
print(f"\n2. CHECKING TEST METERS:")

for meter_id in test_meters:
    meter = Meter.objects.filter(meter_id=meter_id).first()
    if meter:
        print(f"\n{meter_id}:")
        print(f"  ✓ In database")
        
        # Check consumption data
        consumptions = DailyConsumption.objects.filter(meter=meter)
        print(f"  Consumption days: {consumptions.count()}")
        
        if consumptions.count() > 0:
            dates = consumptions.values_list('date', flat=True).order_by('-date')
            print(f"  Latest date: {dates.first()}")
            print(f"  Oldest date: {dates.last()}")
            
            # Check if we have recent data
            thirty_days_ago = timezone.now().date() - timedelta(days=30)
            recent = consumptions.filter(date__gte=thirty_days_ago).count()
            print(f"  Last 30 days: {recent} records")
        
        # Check models
        models = ModelVersion.objects.filter(meter=meter)
        print(f"  Model versions: {models.count()}")
        for model in models:
            status = "✓ ACTIVE" if model.is_active else "inactive"
            print(f"    v{model.version_number}: {status}, trained: {model.trained_on}")
    else:
        print(f"\n{meter_id}: ✗ NOT IN DATABASE")

# 3. System health
print(f"\n3. SYSTEM HEALTH:")
health = AdaptiveLearningManager.get_system_health()
for key, value in health.items():
    print(f"  {key}: {value}")

# 4. Try a simple retraining if possible
print(f"\n4. TEST RETRAINING:")
candidates = AdaptiveLearningManager.get_meters_needing_retraining()
print(f"  Meters needing retraining: {len(candidates)}")

if candidates:
    # Try the first candidate
    meter_id = candidates[0]['meter_id']
    print(f"\n  Trying to retrain {meter_id}...")
    
    # Check if we have enough data
    meter = Meter.objects.get(meter_id=meter_id)
    consumptions = DailyConsumption.objects.filter(meter=meter)
    
    if consumptions.count() >= 30:
        print(f"  ✓ Enough data ({consumptions.count()} days)")
        
        # Check if model exists
        model = ModelVersion.objects.filter(meter=meter, is_active=True).first()
        if model:
            print(f"  ✓ Active model found (v{model.version_number})")
            print(f"  You can run: python manage.py adaptive_retrain --meters {meter_id}")
        else:
            print(f"  ✗ No active model found")
    else:
        print(f"  ✗ Not enough data ({consumptions.count()} days, need 30)")
else:
    print(f"  No candidates found")

print("\n" + "="*60)
print("RECOMMENDED ACTIONS:")
print("="*60)

if Meter.objects.count() < 100:
    print("1. Import more data:")
    print("   python manage.py import_data --path ../electricity_theft_project/hhblock_dataset --limit 10")

if ModelVersion.objects.filter(is_active=True).count() < 10:
    print("\n2. Import more models:")
    print("   python manage.py import_models --source trained_models --activate")
    print("   python manage.py import_models --source models_batch --activate")

print("\n3. After fixing above, test with:")
print("   python manage.py adaptive_retrain --meters MAC000002 --job 'Test'")