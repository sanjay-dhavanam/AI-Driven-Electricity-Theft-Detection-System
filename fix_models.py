#!/usr/bin/env python
import os
import sys
import pickle
import shutil
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'electricity_app.settings')
django.setup()

from core.models import ModelVersion, Meter
from django.conf import settings

print("Fixing model files...")

# Check all ModelVersion records
for mv in ModelVersion.objects.all():
    model_path = mv.get_file_path('model')

    if not model_path:
        print(f"Skipping {mv.meter.meter_id} (model_file is NULL)")
        continue
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"Model file missing for {mv.meter.meter_id}")
        
        # Try to find original .pkl file
        original_paths = [
            os.path.join('trained_models', f'model_{mv.meter.meter_id}.pkl'),
            os.path.join('models_batch', f'model_{mv.meter.meter_id}.pkl'),
            os.path.join('models_sample', f'model_{mv.meter.meter_id}.pkl'),
        ]
        
        found = False
        for orig_path in original_paths:
            if os.path.exists(orig_path):
                # Copy to media directory
                media_dir = os.path.dirname(model_path)
                os.makedirs(media_dir, exist_ok=True)
                shutil.copy2(orig_path, model_path)
                print(f"  ✓ Copied from {orig_path}")
                found = True
                break
        
        if not found:
            print(f"  ✗ No original .pkl file found")
            # Create a dummy model file
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            # Create dummy model
            dummy_model = IsolationForest(n_estimators=10, contamination=0.1, random_state=42)
            dummy_scaler = StandardScaler()
            
            # Train on dummy data
            dummy_data = np.random.rand(30, 48)  # 30 days, 48 half-hours
            dummy_scaler.fit(dummy_data)
            dummy_model.fit(dummy_scaler.transform(dummy_data))
            
            # Save model
            model_data = {
                'model': dummy_model,
                'scaler': dummy_scaler,
                'threshold_n': 0.0,
                'threshold_s': -0.1,
                'training_samples': 30,
                'training_date': '2026-01-09',
                'note': 'Dummy model for testing'
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"  ✓ Created dummy model")

print("\nDone fixing models!")