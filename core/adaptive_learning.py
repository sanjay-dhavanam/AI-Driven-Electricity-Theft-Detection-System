"""
Adaptive Learning System for Electricity Theft Detection
Implements sliding-window retraining with performance-based model replacement
Uses historical data (2010-2014) instead of current dates
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from django.utils import timezone
from django.conf import settings
from django.db import transaction, models
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings('ignore')

# Import models
from core.models import Meter, DailyConsumption, ModelVersion, RetrainingLog, AnomalyResult

class AdaptiveLearningEngine:
    """
    Core engine for adaptive learning
    Implements sliding-window retraining with performance-based model replacement
    """
    
    # Default settings
    DEFAULT_WINDOW_DAYS = 30
    DEFAULT_VALIDATION_DAYS = 7
    DEFAULT_IMPROVEMENT_THRESHOLD = 5.0  # 5% improvement required
    
    @classmethod
    def create_retraining_job(cls, name, user=None, window_days=None, 
                             validation_days=None, improvement_threshold=None):
        """Create a new retraining job"""
        return RetrainingLog.objects.create(
            name=name,
            triggered_by=user,
            status='pending',
            window_days=window_days or cls.DEFAULT_WINDOW_DAYS,
            validation_days=validation_days or cls.DEFAULT_VALIDATION_DAYS,
            improvement_threshold=improvement_threshold or cls.DEFAULT_IMPROVEMENT_THRESHOLD,
        )
    
    @classmethod
    def retrain_meter_models(cls, retraining_log, meter_ids=None, batch_size=10):
        """
        Main adaptive learning function
        Retrains models for specified meters using sliding window
        Uses historical data (2010-2014) instead of current dates
        """
        start_time = timezone.now()
        retraining_log.started_at = start_time
        retraining_log.status = 'running'
        retraining_log.save()
        
        try:
            # Get meters to retrain
            if meter_ids:
                meters = Meter.objects.filter(meter_id__in=meter_ids, is_active=True)
            else:
                # Retrain all meters with active models
                active_versions = ModelVersion.objects.filter(is_active=True)
                meter_ids_from_models = list(set([v.meter.meter_id for v in active_versions]))
                meters = Meter.objects.filter(meter_id__in=meter_ids_from_models, is_active=True)
            
            retraining_log.total_meters = meters.count()
            retraining_log.meters_processed = list(meters.values_list('meter_id', flat=True))
            retraining_log.save()
            
            detailed_logs = []
            success_meters = []
            failed_meters = []
            improved_meters = []
            unchanged_meters = []
            
            total_improvements = []
            
            # Process meters in batches
            for i in range(0, len(meters), batch_size):
                batch = meters[i:i+batch_size]
                
                for meter in batch:
                    try:
                        result = cls._retrain_single_meter_historical(
                            meter,
                            retraining_log.window_days,
                            retraining_log.validation_days,
                            retraining_log.improvement_threshold
                        )
                        
                        detailed_logs.append(result)
                        
                        if result['status'] == 'success':
                            success_meters.append(meter.meter_id)
                            
                            if result['decision'] == 'replaced':
                                improved_meters.append(meter.meter_id)
                                if 'improvement_percentage' in result:
                                    total_improvements.append(result['improvement_percentage'])
                            else:
                                unchanged_meters.append(meter.meter_id)
                                
                        else:
                            failed_meters.append(meter.meter_id)
                            
                    except Exception as e:
                        error_log = {
                            'meter_id': meter.meter_id,
                            'status': 'error',
                            'error': str(e),
                            'timestamp': timezone.now().isoformat()
                        }
                        detailed_logs.append(error_log)
                        failed_meters.append(meter.meter_id)
            
            # Update retraining log
            end_time = timezone.now()
            duration = (end_time - start_time).total_seconds()
            
            retraining_log.status = 'completed' if not failed_meters else 'partial'
            retraining_log.meters_success = success_meters
            retraining_log.meters_failed = failed_meters
            retraining_log.meters_improved = improved_meters
            retraining_log.meters_unchanged = unchanged_meters
            retraining_log.success_count = len(success_meters)
            retraining_log.failed_count = len(failed_meters)
            retraining_log.improved_count = len(improved_meters)
            retraining_log.unchanged_count = len(unchanged_meters)
            retraining_log.detailed_logs = detailed_logs
            retraining_log.completed_at = end_time
            retraining_log.duration_seconds = duration
            
            if total_improvements:
                retraining_log.average_improvement = sum(total_improvements) / len(total_improvements)
            
            retraining_log.save()
            
            return retraining_log
            
        except Exception as e:
            retraining_log.status = 'failed'
            retraining_log.error_logs = str(e)
            retraining_log.completed_at = timezone.now()
            retraining_log.save()
            raise
    
    @classmethod
    def _retrain_single_meter_historical(cls, meter, window_days, validation_days, improvement_threshold):
        """
        Retrain model for a single meter using historical data
        Uses the meter's historical consumption data (2010-2014)
        """
        result = {
            'meter_id': meter.meter_id,
            'timestamp': timezone.now().isoformat(),
            'window_days': window_days,
            'validation_days': validation_days,
            'improvement_threshold': improvement_threshold,
        }
        
        try:
            # Get all consumption data for this meter, ordered by date
            all_consumptions = DailyConsumption.objects.filter(
                meter=meter
            ).order_by('date')
            
            if all_consumptions.count() < window_days:
                result['status'] = 'skipped'
                result['reason'] = f'Insufficient data: {all_consumptions.count()} days available, need {window_days}'
                return result
            
            # Get the most recent window_days of data (from historical dataset)
            # Reverse to get chronological order (oldest to newest) for training
            recent_consumptions = list(all_consumptions[:window_days])
            recent_consumptions.reverse()  # Now oldest to newest
            
            # Split into training and validation
            split_index = window_days - validation_days
            
            if split_index <= 0:
                result['status'] = 'skipped'
                result['reason'] = f'Invalid split: window={window_days}, validation={validation_days}'
                return result
            
            training_data = []
            validation_data = []
            
            for i, consumption in enumerate(recent_consumptions):
                features = cls._extract_features(consumption)
                if features is not None:
                    if i < split_index:
                        training_data.append(features)
                    else:
                        validation_data.append((features, consumption.date))
            
            if len(training_data) < 10 or len(validation_data) < 5:
                result['status'] = 'skipped'
                result['reason'] = f'Insufficient data after processing: train={len(training_data)}, val={len(validation_data)}'
                return result
            
            X_train = np.array(training_data)
            X_val = [data[0] for data in validation_data]
            val_dates = [data[1] for data in validation_data]
            
            # Step 3: Load current model
            current_version = ModelVersion.objects.filter(
                meter=meter,
                is_active=True
            ).first()
            
            if not current_version:
                result['status'] = 'skipped'
                result['reason'] = 'No active model found'
                return result
            
            # Try to load model
            current_model, current_scaler = cls._load_current_model(current_version)
            
            if current_model is None or current_scaler is None:
                result['status'] = 'skipped'
                result['reason'] = 'Current model could not be loaded'
                return result
            
            # Step 4: Train new model
            new_model, new_scaler = cls._train_new_model(X_train)
            
            # Step 5: Evaluate both models on historical validation data
            current_performance = cls._evaluate_model_historical(
                current_model, current_scaler, X_val, val_dates, meter
            )
            
            new_performance = cls._evaluate_model_historical(
                new_model, new_scaler, X_val, val_dates, meter
            )
            
            # Step 6: Compare performance
            improvement_percentage = 0
            if current_performance['f1_score'] > 0:
                improvement_percentage = ((new_performance['f1_score'] - current_performance['f1_score']) 
                                         / current_performance['f1_score']) * 100
            
            # Step 7: Make replacement decision
            should_replace = improvement_percentage >= improvement_threshold
            
            if should_replace:
                # Step 8: Create new model version
                new_version = cls._create_model_version(
                    meter, new_model, new_scaler, X_train,
                    window_days, validation_days,
                    new_performance, improvement_percentage,
                    current_version.version_number + 1
                )
                
                # Activate new model
                new_version.activate()
                
                result.update({
                    'status': 'success',
                    'decision': 'replaced',
                    'old_version': current_version.version_number,
                    'new_version': new_version.version_number,
                    'current_f1': current_performance['f1_score'],
                    'new_f1': new_performance['f1_score'],
                    'improvement_percentage': improvement_percentage,
                    'improvement_absolute': new_performance['f1_score'] - current_performance['f1_score'],
                    'validation_samples': len(X_val),
                    'training_samples': len(X_train),
                })
            else:
                result.update({
                    'status': 'success',
                    'decision': 'kept',
                    'current_version': current_version.version_number,
                    'current_f1': current_performance['f1_score'],
                    'new_f1': new_performance['f1_score'],
                    'improvement_percentage': improvement_percentage,
                    'validation_samples': len(X_val),
                    'training_samples': len(X_train),
                    'reason': f'Improvement ({improvement_percentage:.2f}%) below threshold ({improvement_threshold}%)'
                })
            
            return result
            
        except Exception as e:
            import traceback
            result['status'] = 'error'
            result['error'] = f"{str(e)}\n{traceback.format_exc()}"
            return result
    
    @staticmethod
    def _load_current_model(model_version):
        """Load current model and scaler"""
        try:
            model_path = model_version.get_file_path('model')
            if not model_path or not os.path.exists(model_path):
                # Try alternative paths
                alternative_paths = [
                    os.path.join('trained_models', f'model_{model_version.meter.meter_id}.pkl'),
                    os.path.join('models_batch', f'model_{model_version.meter.meter_id}.pkl'),
                    os.path.join('models_sample', f'model_{model_version.meter.meter_id}.pkl'),
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        model_path = alt_path
                        break
                else:
                    return None, None
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_data.get('model')
            scaler = model_data.get('scaler')
            
            return model, scaler
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None
    
    @staticmethod
    def _extract_features(daily_consumption):
        """Extract features from DailyConsumption object"""
        try:
            consumption_data = daily_consumption.consumption_data
            features = []
            
            for i in range(48):
                key = f'hh_{i}'
                value = consumption_data.get(key, 0.0)
                features.append(float(value))
            
            return np.array(features)
        except:
            return None
    
    @staticmethod
    def _train_new_model(X_train):
        """Train new Isolation Forest model"""
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train Isolation Forest
        model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            max_samples='auto',
            max_features=1.0,
            bootstrap=False,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled)
        
        return model, scaler
    
    @staticmethod
    def _evaluate_model_historical(model, scaler, X_val, val_dates, meter):
        """
        Evaluate model on historical validation data
        Uses pseudo-labels based on existing anomaly results
        """
        if len(X_val) == 0:
            return {'f1_score': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}
        
        # Scale validation data
        X_val_scaled = scaler.transform(np.array(X_val))
        
        # Get anomaly scores
        try:
            anomaly_scores = model.decision_function(X_val_scaled)
        except:
            # If decision_function not available, use predict
            predictions = model.predict(X_val_scaled)
            anomaly_scores = [1.0 if p == 1 else -1.0 for p in predictions]
        
        # For historical evaluation, we need to create pseudo-labels
        # We'll use existing anomaly results as pseudo-ground truth
        true_labels = []
        pred_labels = []
        
        for i, (features, date) in enumerate(zip(X_val, val_dates)):
            # Get existing anomaly result for this date
            try:
                existing_result = AnomalyResult.objects.get(
                    meter=meter,
                    date=date
                )
                
                # Convert classification to binary: 0 = normal, 1 = suspicious/theft
                is_anomaly = 1 if existing_result.classification in ['suspicious', 'theft'] else 0
                true_labels.append(is_anomaly)
                
            except AnomalyResult.DoesNotExist:
                # If no existing result, skip this sample
                continue
            
            # Predict using current model (simplified threshold)
            threshold = 0.0  # Default threshold
            pred = 1 if anomaly_scores[i] < threshold else 0
            pred_labels.append(pred)
        
        if len(true_labels) < 3:  # Need minimum samples
            # Use statistical measures as fallback
            score_variance = np.var(anomaly_scores)
            mean_score = np.mean(anomaly_scores)
            std_score = np.std(anomaly_scores)
            
            # Calculate pseudo F1 based on score distribution
            if std_score > 0:
                # Normalize variance to 0-1 range
                norm_variance = min(score_variance / 1.0, 1.0)
                # Calculate separation (higher is better)
                separation = np.mean(np.abs((anomaly_scores - mean_score) / std_score)) if std_score > 0 else 0
                separation_norm = min(separation / 2.0, 1.0)
                
                # Combined pseudo-F1
                pseudo_f1 = (1.0 - norm_variance) * 0.5 + separation_norm * 0.5
            else:
                pseudo_f1 = 0.5
            
            return {
                'f1_score': pseudo_f1,
                'accuracy': 1.0 - min(score_variance, 1.0),
                'precision': 0.5,
                'recall': 0.5,
                'samples_evaluated': len(X_val),
                'note': 'Pseudo-evaluation (no ground truth)'
            }
        
        # Calculate actual metrics if we have labels
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(true_labels, pred_labels)
            precision = precision_score(true_labels, pred_labels, zero_division=0)
            recall = recall_score(true_labels, pred_labels, zero_division=0)
            f1 = f1_score(true_labels, pred_labels, zero_division=0)
            
            return {
                'f1_score': f1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'samples_evaluated': len(true_labels),
                'note': 'Evaluation with existing results as pseudo-labels'
            }
        except:
            # Fallback to pseudo-evaluation
            score_variance = np.var(anomaly_scores)
            pseudo_f1 = 1.0 - min(score_variance / 1.0, 1.0)
            
            return {
                'f1_score': pseudo_f1,
                'accuracy': 1.0 - min(score_variance, 1.0),
                'precision': 0.5,
                'recall': 0.5,
                'samples_evaluated': len(X_val),
                'note': 'Fallback pseudo-evaluation'
            }
    
    @staticmethod
    def _create_model_version(meter, model, scaler, X_train, window_days, 
                             validation_days, performance, improvement_percentage, version_number):
        """Create new ModelVersion record and save model files"""
        # Create directory for model files
        models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        scalers_dir = os.path.join(settings.MEDIA_ROOT, 'scalers')
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(scalers_dir, exist_ok=True)
        
        # Generate filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'model_{meter.meter_id}_v{version_number}_{timestamp}.pkl'
        scaler_filename = f'scaler_{meter.meter_id}_v{version_number}_{timestamp}.pkl'
        
        model_path = os.path.join(models_dir, model_filename)
        scaler_path = os.path.join(scalers_dir, scaler_filename)
        
        # Calculate thresholds from training data
        X_train_scaled = scaler.transform(X_train)
        scores = model.decision_function(X_train_scaled)
        
        # Use percentiles for thresholds
        threshold_normal = np.percentile(scores, 75)  # Top 25% = normal
        threshold_theft = np.percentile(scores, 25)   # Bottom 25% = theft
        
        # Save model file (combine model and scaler)
        model_data = {
            'model': model,
            'scaler': scaler,
            'threshold_normal': threshold_normal,
            'threshold_theft': threshold_theft,
            'training_date': datetime.now(),
            'training_samples': len(X_train),
            'window_days': window_days,
            'validation_days': validation_days,
            'improvement_percentage': improvement_percentage,
            'performance': performance,
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save scaler separately as well
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Create ModelVersion record
        model_version = ModelVersion.objects.create(
            meter=meter,
            version_number=version_number,
            model_name='IsolationForest',
            model_file=f'models/{model_filename}',
            scaler_file=f'scalers/{scaler_filename}',
            window_size=window_days,
            contamination=0.1,
            threshold_normal=threshold_normal,
            threshold_theft=threshold_theft,
            training_samples=len(X_train),
            validation_accuracy=performance.get('accuracy'),
            validation_f1=performance.get('f1_score'),
            false_positive_rate=None,
            false_negative_rate=None,
            is_trained=True,
            trained_on=datetime.now().date(),
            training_logs={
                'window_days': window_days,
                'validation_days': validation_days,
                'performance': performance,
                'improvement_percentage': improvement_percentage,
                'created_at': datetime.now().isoformat(),
            }
        )
        
        return model_version

class AdaptiveLearningManager:
    """
    Manager class for adaptive learning operations
    Provides higher-level interface and utilities
    Uses historical data for calculations
    """
    
    @classmethod
    def get_meters_needing_retraining(cls, days_since_retraining=30, min_anomalies=3):
        """
        Identify meters that might benefit from retraining
        Uses historical data for anomaly calculations
        """
        candidates = []
        
        # Get all active meters with models
        active_versions = ModelVersion.objects.filter(is_active=True).select_related('meter')
        
        for version in active_versions:
            meter = version.meter
            
            # Check 1: Time since last retraining (use real time, not historical)
            days_since = (timezone.now().date() - version.trained_on).days if version.trained_on else 999
            
            # Check 2: Recent anomalies in historical data
            # Get the max date from consumption data for this meter
            max_date = DailyConsumption.objects.filter(meter=meter).aggregate(models.Max('date'))['date__max']
            if not max_date:
                continue
            
            # Calculate 30 days back from max historical date
            thirty_days_before = max_date - timedelta(days=30)
            
            recent_anomalies = AnomalyResult.objects.filter(
                meter=meter,
                date__gte=thirty_days_before,
                date__lte=max_date,
                classification__in=['suspicious', 'theft']
            ).count()
            
            # Check 3: Confidence scores (optional)
            recent_results = AnomalyResult.objects.filter(
                meter=meter,
                date__gte=thirty_days_before,
                date__lte=max_date
            )
            avg_confidence = recent_results.aggregate(avg=models.Avg('confidence'))['avg'] or 0
            
            # Scoring system
            score = 0
            reasons = []
            
            if days_since > days_since_retraining:
                score += 2
                reasons.append(f"Not retrained in {days_since} days")
            
            if recent_anomalies >= min_anomalies:
                score += 3
                reasons.append(f"{recent_anomalies} recent anomalies (historical)")
            
            if avg_confidence < 0.6:
                score += 1
                reasons.append(f"Low confidence ({avg_confidence:.2f})")
            
            if score >= 3:  # Threshold for suggesting retraining
                candidates.append({
                    'meter': meter,
                    'meter_id': meter.meter_id,
                    'score': score,
                    'reasons': reasons,
                    'days_since_retraining': days_since,
                    'recent_anomalies': recent_anomalies,
                    'avg_confidence': avg_confidence,
                    'current_version': version.version_number,
                })
        
        # Sort by score descending
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates
    
    @classmethod
    def get_performance_history(cls, meter_id, limit=10):
        """Get retraining performance history for a meter"""
        versions = ModelVersion.objects.filter(
            meter__meter_id=meter_id,
            is_trained=True
        ).order_by('-version_number')[:limit]
        
        history = []
        for version in versions:
            history.append({
                'version': version.version_number,
                'trained_on': version.trained_on,
                'is_active': version.is_active,
                'validation_f1': version.validation_f1,
                'validation_accuracy': version.validation_accuracy,
                'training_samples': version.training_samples,
                'window_size': version.window_size,
            })
        
        return history
    
    @classmethod
    def get_system_health(cls):
        """Get overall system health metrics"""
        total_meters = Meter.objects.filter(is_active=True).count()
        meters_with_models = ModelVersion.objects.filter(is_active=True).count()
        
        # Recent retraining activity (real-time)
        recent_logs = RetrainingLog.objects.filter(
            completed_at__gte=timezone.now() - timedelta(days=30)
        )
        
        successful_retrains = recent_logs.filter(status__in=['completed', 'partial']).count()
        total_retrains = recent_logs.count()
        
        # Model age distribution
        active_versions = ModelVersion.objects.filter(is_active=True)
        model_ages = []
        for version in active_versions:
            if version.trained_on:
                age = (timezone.now().date() - version.trained_on).days
                model_ages.append(age)
        
        avg_model_age = sum(model_ages) / len(model_ages) if model_ages else 0
        
        return {
            'total_meters': total_meters,
            'meters_with_models': meters_with_models,
            'coverage_percentage': (meters_with_models / total_meters * 100) if total_meters > 0 else 0,
            'recent_retrains': total_retrains,
            'successful_retrains': successful_retrains,
            'success_rate': (successful_retrains / total_retrains * 100) if total_retrains > 0 else 0,
            'average_model_age_days': avg_model_age,
            'oldest_model_age': max(model_ages) if model_ages else 0,
            'meters_needing_retraining': len(cls.get_meters_needing_retraining()),
        }

class RetrainingCommands:
    """
    Command-line utilities for adaptive learning
    """
    
    @classmethod
    def run_retraining_job(cls, job_name, meter_ids=None, user_id=None, batch_size=5):
        """Run a retraining job"""
        from django.contrib.auth.models import User
        
        user = User.objects.get(id=user_id) if user_id else None
        
        # Create retraining log
        retraining_log = AdaptiveLearningEngine.create_retraining_job(
            name=job_name,
            user=user,
        )
        
        try:
            # Run retraining
            result = AdaptiveLearningEngine.retrain_meter_models(
                retraining_log=retraining_log,
                meter_ids=meter_ids,
                batch_size=batch_size
            )
            
            return result
            
        except Exception as e:
            print(f"Error running retraining job: {e}")
            return None
    
    @classmethod
    def retrain_specific_meter(cls, meter_id, user_id=None):
        """Retrain a specific meter (for debugging/testing)"""
        from django.contrib.auth.models import User
        
        user = User.objects.get(id=user_id) if user_id else None
        
        retraining_log = AdaptiveLearningEngine.create_retraining_job(
            name=f"Single meter retraining: {meter_id}",
            user=user,
        )
        
        result = AdaptiveLearningEngine.retrain_meter_models(
            retraining_log=retraining_log,
            meter_ids=[meter_id],
            batch_size=1
        )
        
        return result