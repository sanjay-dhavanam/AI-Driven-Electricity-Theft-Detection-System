#!/usr/bin/env python
import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'electricity_app.settings')
django.setup()

from core.models import Meter, DailyConsumption, ModelVersion, AnomalyResult
from core.adaptive_learning import AdaptiveLearningEngine, AdaptiveLearningManager
from core.ml_logic import AnomalyPredictor, ModelLoader
import pickle
from datetime import datetime, timedelta
from django.utils import timezone

print("="*80)
print("FINAL SYSTEM TEST")
print("="*80)

# 1. Check database
print("\n1. DATABASE CHECK:")
print(f"Meters: {Meter.objects.count()}")
print(f"DailyConsumption: {DailyConsumption.objects.count()}")
print(f"ModelVersion: {ModelVersion.objects.count()}")
print(f"Active models: {ModelVersion.objects.filter(is_active=True).count()}")

# 2. Check a specific meter
test_meter_id = 'MAC000002'
print(f"\n2. CHECKING METER {test_meter_id}:")
meter = Meter.objects.get(meter_id=test_meter_id)

# Check consumption
consumptions = DailyConsumption.objects.filter(meter=meter)
print(f"  Total consumption days: {consumptions.count()}")
print(f"  Date range: {consumptions.order_by('date').first().date} to {consumptions.order_by('-date').first().date}")

# Check model
model_version = ModelVersion.objects.filter(meter=meter, is_active=True).first()
print(f"  Model version: v{model_version.version_number if model_version else 'None'}")
print(f"  Model file exists: {os.path.exists(model_version.get_file_path('model')) if model_version else False}")

# 3. Test prediction
print(f"\n3. TESTING PREDICTION:")
# Get a date with data
sample_date = consumptions.first().date
print(f"  Sample date: {sample_date}")

try:
    result = AnomalyPredictor.predict_for_day(test_meter_id, sample_date)
    if result:
        print(f"  ✓ Prediction successful!")
        print(f"    Classification: {result.classification}")
        print(f"    Score: {result.anomaly_score}")
        print(f"    Confidence: {result.confidence}")
    else:
        print(f"  ✗ Prediction failed")
except Exception as e:
    print(f"  ✗ Prediction error: {e}")

# 4. Test model loading
print(f"\n4. TESTING MODEL LOADING:")
model, scaler, thresholds, model_version_obj = ModelLoader.get_active_model(test_meter_id)
if model:
    print(f"  ✓ Model loaded successfully")
    print(f"    Model type: {type(model).__name__}")
    print(f"    Scaler type: {type(scaler).__name__ if scaler else 'None'}")
else:
    print(f"  ✗ Model loading failed")

# 5. Test adaptive learning for this meter
print(f"\n5. TESTING ADAPTIVE LEARNING:")
# Create a simple retraining log
from core.models import RetrainingLog

# First, let's see if we can retrain manually
print(f"  Trying manual retrain for {test_meter_id}...")

# Get enough data for retraining (30 days)
all_dates = list(consumptions.values_list('date', flat=True).order_by('-date'))
if len(all_dates) >= 30:
    print(f"  ✓ Enough data available ({len(all_dates)} days)")
    
    # Try retraining with simpler parameters
    try:
        # Create retraining log
        retraining_log = RetrainingLog.objects.create(
            name=f"Test retraining for {test_meter_id}",
            status='pending',
            window_days=30,
            validation_days=7,
            improvement_threshold=5.0,
            total_meters=1,
            meters_processed=[test_meter_id]
        )
        
        # Call the retraining engine directly
        from core.adaptive_learning import AdaptiveLearningEngine
        
        result = AdaptiveLearningEngine._retrain_single_meter(
            meter=meter,
            window_days=30,
            validation_days=7,
            improvement_threshold=5.0
        )
        
        print(f"  Retraining result: {result.get('status', 'unknown')}")
        print(f"  Decision: {result.get('decision', 'none')}")
        print(f"  Improvement: {result.get('improvement_percentage', 0):.2f}%")
        
    except Exception as e:
        print(f"  ✗ Retraining error: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"  ✗ Not enough data for retraining (need 30, have {len(all_dates)})")

# 6. Quick fix: Create a simple test with first 30 days
print(f"\n6. QUICK FIX TEST:")
# Get first 30 days of data (chronological order)
first_30 = list(consumptions.order_by('date')[:30])
if len(first_30) == 30:
    print(f"  ✓ Found first 30 days of data")
    print(f"  Dates: {first_30[0].date} to {first_30[-1].date}")
    
    # Check if we can extract features
    from core.adaptive_learning import AdaptiveLearningEngine
    features = AdaptiveLearningEngine._extract_features(first_30[0])
    if features is not None:
        print(f"  ✓ Feature extraction works")
        print(f"    Features shape: {features.shape}")
    else:
        print(f"  ✗ Feature extraction failed")
else:
    print(f"  ✗ Don't have 30 consecutive days")

print("\n" + "="*80)
print("RECOMMENDED NEXT STEPS:")
print("="*80)
print("1. Fix admin error by updating core/admin.py")
print("2. Test prediction for a single day:")
print("   python manage.py predict_anomalies --meter MAC000002 --date 2012-10-13")
print("3. Test adaptive retraining with debug:")
print("   python manage.py adaptive_retrain --meters MAC000002 --job 'Debug' --batch 1")
print("4. Check admin at: http://localhost:8000/admin/core/retraininglog/")