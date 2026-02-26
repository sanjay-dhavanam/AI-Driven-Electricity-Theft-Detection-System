#!/usr/bin/env python
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'electricity_app.settings')
django.setup()

from core.models import Meter, DailyConsumption, ModelVersion, AnomalyResult

print("="*60)
print("DATABASE DEBUG INFO")
print("="*60)

# 1. Check meters
print("\n1. METERS:")
meters = Meter.objects.all()
print(f"Total meters: {meters.count()}")
print(f"Sample meters: {list(meters.values_list('meter_id', flat=True)[:10])}")

# 2. Check specific meter
target_meter = 'MAC000002'
print(f"\n2. CHECKING METER {target_meter}:")
meter = Meter.objects.filter(meter_id=target_meter).first()
if meter:
    print(f"  ✓ Found in database")
    
    # Check daily consumption
    consumptions = DailyConsumption.objects.filter(meter=meter)
    print(f"  Daily consumption records: {consumptions.count()}")
    
    if consumptions.exists():
        dates = consumptions.values_list('date', flat=True).order_by('date')
        print(f"  Date range: {dates.first()} to {dates.last()}")
        print(f"  Latest 5 dates: {list(dates[:5])}")
    
    # Check models
    models = ModelVersion.objects.filter(meter=meter)
    print(f"  Model versions: {models.count()}")
    for model in models:
        print(f"    Version {model.version_number}: active={model.is_active}, "
              f"trained={model.trained_on}")
else:
    print(f"  ✗ NOT FOUND in database")

# 3. Check all ModelVersion records
print(f"\n3. MODEL VERSIONS:")
model_versions = ModelVersion.objects.all()
print(f"Total ModelVersion records: {model_versions.count()}")
print(f"Active models: {model_versions.filter(is_active=True).count()}")

# Show first 10
for mv in model_versions[:10]:
    print(f"  {mv.meter.meter_id} v{mv.version_number}: "
          f"{'✓ ACTIVE' if mv.is_active else 'inactive'}")

# 4. Check if we have enough data for retraining
print(f"\n4. DATA FOR ADAPTIVE LEARNING:")
for meter_id in ['MAC000002', 'MAC000048', 'MAC000154']:
    meter = Meter.objects.filter(meter_id=meter_id).first()
    if meter:
        # Check last 30 days of data
        from datetime import date, timedelta
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        
        recent_data = DailyConsumption.objects.filter(
            meter=meter,
            date__gte=start_date
        ).count()
        
        print(f"  {meter_id}: {recent_data} days of recent data "
              f"(need 30 for retraining)")