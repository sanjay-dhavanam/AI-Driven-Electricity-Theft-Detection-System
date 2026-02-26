"""
ML Logic for Electricity Theft Detection
Handles model loading, prediction, and classification
"""

import os
import pickle
import numpy as np
import pandas as pd
from django.conf import settings
from django.db.models import Q
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from django.db import models
# Import models
from core.models import Meter, DailyConsumption, AnomalyResult, ModelVersion

class ModelLoader:
    """
    Loads and manages ML models for prediction
    """
    
    # Cache loaded models to avoid repeated disk reads
    _model_cache = {}
    _scaler_cache = {}
    
    @classmethod
    def clear_cache(cls):
        """Clear model cache"""
        cls._model_cache = {}
        cls._scaler_cache = {}
    
    @classmethod
    def get_active_model(cls, meter_id):
        """
        Get active model for a meter
        Returns: (model, scaler, thresholds, model_version)
        """
        try:
            # Try to get from ModelVersion first
            model_version = ModelVersion.objects.filter(
                meter__meter_id=meter_id,
                is_active=True,
                is_trained=True
            ).first()
            
            if model_version:
                return cls._load_from_model_version(model_version)
            
            # Fallback to legacy .pkl files
            return cls._load_from_legacy_files(meter_id)
            
        except Exception as e:
            print(f"Error loading model for {meter_id}: {e}")
            return None, None, None, None
    
    @classmethod
    def _load_from_model_version(cls, model_version):
        """Load model from ModelVersion database record"""
        cache_key = f"{model_version.meter.meter_id}_v{model_version.version_number}"
        
        # Check cache first
        if cache_key in cls._model_cache:
            return (
                cls._model_cache[cache_key]['model'],
                cls._model_cache[cache_key]['scaler'],
                cls._model_cache[cache_key]['thresholds'],
                model_version
            )
        
        try:
            # Load model file
            model_path = model_version.get_file_path('model')
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_data.get('model')
            scaler = model_data.get('scaler')
            
            # If scaler not in model file, try to load separately
            if scaler is None and model_version.scaler_file:
                scaler_path = model_version.get_file_path('scaler')
                if scaler_path and os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
            
            # Get thresholds from model_version or model_data
            thresholds = {
                'normal': model_version.threshold_normal or model_data.get('threshold_n', 0.0),
                'theft': model_version.threshold_theft or model_data.get('threshold_s', -0.1),
            }
            
            # Cache the loaded model
            cls._model_cache[cache_key] = {
                'model': model,
                'scaler': scaler,
                'thresholds': thresholds,
            }
            
            return model, scaler, thresholds, model_version
            
        except Exception as e:
            print(f"Error loading model from version {model_version.id}: {e}")
            return None, None, None, model_version
    
    @classmethod
    def _load_from_legacy_files(cls, meter_id):
        """Load model from legacy .pkl files (backward compatibility)"""
        cache_key = f"{meter_id}_legacy"
        
        # Check cache
        if cache_key in cls._model_cache:
            return (
                cls._model_cache[cache_key]['model'],
                cls._model_cache[cache_key]['scaler'],
                cls._model_cache[cache_key]['thresholds'],
                None  # No model_version for legacy
            )
        
        # Try different possible locations
        possible_paths = [
            os.path.join(settings.MODELS_DIR, f'model_{meter_id}.pkl'),
            os.path.join(settings.BASE_DIR, 'trained_models', f'model_{meter_id}.pkl'),
            os.path.join(settings.BASE_DIR, 'models_batch', f'model_{meter_id}.pkl'),
            os.path.join(settings.BASE_DIR, 'models_sample', f'model_{meter_id}.pkl'),
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print(f"No model found for {meter_id}")
            return None, None, None, None
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_data.get('model')
            scaler = model_data.get('scaler')
            
            # Extract thresholds
            thresholds = {
                'normal': model_data.get('threshold_n', 0.0),
                'theft': model_data.get('threshold_s', -0.1),
            }
            
            # Cache the loaded model
            cls._model_cache[cache_key] = {
                'model': model,
                'scaler': scaler,
                'thresholds': thresholds,
            }
            
            return model, scaler, thresholds, None
            
        except Exception as e:
            print(f"Error loading legacy model for {meter_id}: {e}")
            return None, None, None, None
    
    @classmethod
    def check_model_exists(cls, meter_id):
        """Check if model exists for a meter"""
        # Check ModelVersion
        if ModelVersion.objects.filter(meter__meter_id=meter_id, is_active=True).exists():
            return True
        
        # Check legacy files
        possible_paths = [
            os.path.join(settings.MODELS_DIR, f'model_{meter_id}.pkl'),
            os.path.join(settings.BASE_DIR, 'trained_models', f'model_{meter_id}.pkl'),
            os.path.join(settings.BASE_DIR, 'models_batch', f'model_{meter_id}.pkl'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return True
        
        return False

class AnomalyPredictor:
    """
    Predicts anomalies for daily consumption data
    """
    
    @staticmethod
    def prepare_features(daily_consumption):
        """
        Prepare features from DailyConsumption object
        Returns: numpy array of shape (1, 48)
        """
        # Extract half-hour readings
        consumption_data = daily_consumption.consumption_data
        
        # Ensure we have 48 readings
        features = []
        for i in range(48):
            key = f'hh_{i}'
            if key in consumption_data:
                features.append(float(consumption_data[key]))
            else:
                # Fill missing with 0 (or could use median)
                features.append(0.0)
        
        return np.array(features).reshape(1, -1)
    
    @staticmethod
    def classify_anomaly(score, threshold_normal, threshold_theft):
        """
        Classify anomaly score into categories
        Returns: (classification, confidence)
        """
        if score > threshold_normal:
            return 'normal', min(1.0, (score - threshold_normal) / (1 - threshold_normal))
        elif score > threshold_theft:
            # Calculate confidence based on distance from thresholds
            range_size = threshold_normal - threshold_theft
            if range_size > 0:
                distance = (score - threshold_theft) / range_size
                confidence = 1.0 - abs(distance - 0.5) * 2  # Max at middle
                confidence = max(0.1, min(0.9, confidence))
            else:
                confidence = 0.5
            return 'suspicious', confidence
        else:
            return 'theft', min(1.0, (threshold_theft - score) / (threshold_theft + 1))
    
    @staticmethod
    def calculate_additional_insights(features, daily_consumption):
        """Calculate additional insights for the anomaly result"""
        # Find peak hour
        features_flat = features.flatten()
        peak_hour = int(np.argmax(features_flat))
        peak_consumption = float(features_flat[peak_hour])
        
        # Calculate deviation from daily average
        daily_avg = daily_consumption.average_consumption
        total_deviation = np.sum(np.abs(features_flat - daily_avg)) if daily_avg else 0.0
        
        return {
            'peak_hour': peak_hour,
            'peak_consumption': peak_consumption,
            'total_deviation': total_deviation,
        }
    
    @classmethod
    def predict_for_day(cls, meter_id, date):
        """
        Predict anomaly for a specific meter and date
        Returns: AnomalyResult object or None
        """
        # Check if prediction already exists
        existing_result = AnomalyResult.objects.filter(
            meter__meter_id=meter_id,
            date=date
        ).first()
        
        if existing_result and existing_result.is_recent():
            # Return existing prediction if recent
            return existing_result
        
        try:
            # Get daily consumption data
            daily_consumption = DailyConsumption.objects.get(
                meter__meter_id=meter_id,
                date=date
            )
        except DailyConsumption.DoesNotExist:
            print(f"No consumption data for {meter_id} on {date}")
            return None
        
        # Load model
        model, scaler, thresholds, model_version = ModelLoader.get_active_model(meter_id)
        
        if model is None or scaler is None:
            print(f"No model available for {meter_id}")
            return None
        
        # Prepare features
        features = cls.prepare_features(daily_consumption)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict anomaly score (Isolation Forest: lower = more anomalous)
        try:
            anomaly_score = float(model.decision_function(features_scaled)[0])
        except AttributeError:
            # If decision_function not available, use predict
            prediction = model.predict(features_scaled)[0]
            anomaly_score = 1.0 if prediction == 1 else -1.0  # 1 = normal, -1 = anomaly
        
        # Classify
        classification, confidence = cls.classify_anomaly(
            anomaly_score,
            thresholds['normal'],
            thresholds['theft']
        )
        
        # Calculate additional insights
        insights = cls.calculate_additional_insights(features, daily_consumption)
        
        # Create or update AnomalyResult
        anomaly_result, created = AnomalyResult.objects.update_or_create(
            meter=daily_consumption.meter,
            date=date,
            defaults={
                'daily_consumption': daily_consumption,
                'anomaly_score': anomaly_score,
                'classification': classification,
                'confidence': confidence,
                'threshold_normal': thresholds['normal'],
                'threshold_suspicious': (thresholds['normal'] + thresholds['theft']) / 2,
                'threshold_theft': thresholds['theft'],
                'peak_hour': insights['peak_hour'],
                'peak_consumption': insights['peak_consumption'],
                'total_deviation': insights['total_deviation'],
                'model_version': model_version.version_number if model_version else None,
                'model_name': model_version.model_name if model_version else 'IsolationForest',
            }
        )
        
        return anomaly_result
    
    @classmethod
    def batch_predict_for_meter(cls, meter_id, start_date=None, end_date=None, limit=100):
        """
        Predict anomalies for multiple days for a meter
        Returns: list of AnomalyResult objects
        """
        # Get consumption records
        query = DailyConsumption.objects.filter(meter__meter_id=meter_id)
        
        if start_date:
            query = query.filter(date__gte=start_date)
        if end_date:
            query = query.filter(date__lte=end_date)
        
        # Order by date descending
        daily_consumptions = query.order_by('-date')[:limit]
        
        results = []
        for daily_consumption in daily_consumptions:
            result = cls.predict_for_day(meter_id, daily_consumption.date)
            if result:
                results.append(result)
        
        return results
    
    @classmethod
    def predict_latest(cls, meter_id):
        """
        Predict anomaly for the latest available data for a meter
        """
        try:
            latest_consumption = DailyConsumption.objects.filter(
                meter__meter_id=meter_id
            ).latest('date')
            
            return cls.predict_for_day(meter_id, latest_consumption.date)
        except DailyConsumption.DoesNotExist:
            return None

class ModelEvaluator:
    """
    Evaluates model performance
    """
    
    @staticmethod
    def get_prediction_stats(meter_id, days=30):
        """
        Get statistics for recent predictions
        """
        from django.utils.timezone import now
        from datetime import timedelta
        
        end_date = now().date()
        start_date = end_date - timedelta(days=days)
        
        predictions = AnomalyResult.objects.filter(
            meter__meter_id=meter_id,
            date__gte=start_date,
            date__lte=end_date
        )
        
        total = predictions.count()
        if total == 0:
            return None
        
        # Count classifications
        normal_count = predictions.filter(classification='normal').count()
        suspicious_count = predictions.filter(classification='suspicious').count()
        theft_count = predictions.filter(classification='theft').count()
        
        # Calculate average confidence
        avg_confidence = predictions.aggregate(avg=models.Avg('confidence'))['avg'] or 0
        
        return {
            'total_days': total,
            'normal_days': normal_count,
            'suspicious_days': suspicious_count,
            'theft_days': theft_count,
            'normal_percentage': (normal_count / total) * 100,
            'suspicious_percentage': (suspicious_count / total) * 100,
            'theft_percentage': (theft_count / total) * 100,
            'average_confidence': avg_confidence,
            'date_range': {
                'start': start_date,
                'end': end_date,
            }
        }
    
    @staticmethod
    def get_meter_summary(meter_id):
        """
        Get comprehensive summary for a meter
        """
        meter = Meter.objects.get(meter_id=meter_id)
        
        # Get latest prediction
        latest_prediction = AnomalyResult.objects.filter(
            meter=meter
        ).order_by('-date').first()
        
        # Get model info
        model_version = ModelVersion.objects.filter(
            meter=meter,
            is_active=True
        ).first()
        
        # Get stats
        stats_30d = ModelEvaluator.get_prediction_stats(meter_id, 30)
        stats_90d = ModelEvaluator.get_prediction_stats(meter_id, 90)
        
        return {
            'meter': {
                'id': meter.meter_id,
                'location': meter.location,
                'acorn_group': meter.acorn_group,
                'is_active': meter.is_active,
            },
            'latest_prediction': {
                'date': latest_prediction.date if latest_prediction else None,
                'classification': latest_prediction.classification if latest_prediction else None,
                'score': latest_prediction.anomaly_score if latest_prediction else None,
                'color': latest_prediction.get_color_code() if latest_prediction else 'secondary',
            },
            'model': {
                'exists': ModelLoader.check_model_exists(meter_id),
                'version': model_version.version_number if model_version else None,
                'trained_on': model_version.trained_on if model_version else None,
                'window_size': model_version.window_size if model_version else None,
            },
            'stats_30d': stats_30d,
            'stats_90d': stats_90d,
        }

class LegacyModelImporter:
    """
    Imports legacy .pkl files into ModelVersion system
    """
    
    @staticmethod
    def import_legacy_models(legacy_dir=None, activate=True):
        """
        Import all legacy .pkl files as ModelVersion records
        """
        import glob
        
        if legacy_dir is None:
            legacy_dir = os.path.join(settings.BASE_DIR, 'trained_models')
        
        if not os.path.exists(legacy_dir):
            print(f"Legacy directory not found: {legacy_dir}")
            return 0
        
        # Find all .pkl files
        pkl_files = glob.glob(os.path.join(legacy_dir, 'model_*.pkl'))
        
        imported_count = 0
        for pkl_file in pkl_files:
            try:
                # Extract meter ID from filename
                filename = os.path.basename(pkl_file)
                meter_id = filename.replace('model_', '').replace('.pkl', '')
                
                # Check if meter exists
                try:
                    meter = Meter.objects.get(meter_id=meter_id)
                except Meter.DoesNotExist:
                    print(f"Meter {meter_id} not found in database")
                    continue
                
                # Check if model version already exists
                existing = ModelVersion.objects.filter(meter=meter).count()
                version_number = existing + 1
                
                # Load model data to get metadata
                with open(pkl_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Create ModelVersion
                model_version = ModelVersion.objects.create(
                    meter=meter,
                    version_number=version_number,
                    model_name='IsolationForest',
                    model_file=os.path.relpath(pkl_file, settings.MEDIA_ROOT),
                    window_size=model_data.get('training_samples', 30),
                    contamination=0.1,  # Default
                    threshold_normal=model_data.get('threshold_n', 0.0),
                    threshold_theft=model_data.get('threshold_s', -0.1),
                    training_samples=model_data.get('training_samples', 0),
                    is_trained=True,
                    trained_on=model_data.get('training_date').date() if 'training_date' in model_data else None,
                )
                
                # Activate if requested
                if activate:
                    model_version.activate()
                
                imported_count += 1
                print(f"Imported model for {meter_id} (v{version_number})")
                
            except Exception as e:
                print(f"Error importing {pkl_file}: {e}")
        
        return imported_count

# Utility functions
def predict_all_meters_latest(limit=100):
    """
    Predict latest anomaly for all meters with models
    """
    results = []
    
    # Get all meters with active models
    meters_with_models = []
    
    # First check ModelVersion
    active_versions = ModelVersion.objects.filter(is_active=True)
    for version in active_versions:
        meters_with_models.append(version.meter.meter_id)
    
    # Add meters with legacy models
    possible_dirs = [
        settings.MODELS_DIR,
        os.path.join(settings.BASE_DIR, 'trained_models'),
        os.path.join(settings.BASE_DIR, 'models_batch'),
    ]
    
    for model_dir in possible_dirs:
        if os.path.exists(model_dir):
            for filename in os.listdir(model_dir):
                if filename.startswith('model_') and filename.endswith('.pkl'):
                    meter_id = filename.replace('model_', '').replace('.pkl', '')
                    if meter_id not in meters_with_models:
                        meters_with_models.append(meter_id)
    
    # Limit to specified number
    meters_to_process = meters_with_models[:limit]
    
    for meter_id in meters_to_process:
        try:
            result = AnomalyPredictor.predict_latest(meter_id)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error predicting for {meter_id}: {e}")
    
    return results

def generate_sample_predictions():
    """
    Generate predictions for sample meters (for testing/demo)
    """
    sample_meters = ['MAC000002', 'MAC000048', 'MAC000131', 'MAC000174']
    
    for meter_id in sample_meters:
        print(f"\nProcessing {meter_id}:")
        
        # Check if meter exists
        try:
            meter = Meter.objects.get(meter_id=meter_id)
        except Meter.DoesNotExist:
            print(f"  Meter not found in database")
            continue
        
        # Check if model exists
        if not ModelLoader.check_model_exists(meter_id):
            print(f"  No model found")
            continue
        
        # Get latest 5 days
        consumptions = DailyConsumption.objects.filter(
            meter=meter
        ).order_by('-date')[:5]
        
        for consumption in consumptions:
            result = AnomalyPredictor.predict_for_day(meter_id, consumption.date)
            if result:
                print(f"  {consumption.date}: {result.classification} "
                      f"(score: {result.anomaly_score:.3f}, confidence: {result.confidence:.2f})")