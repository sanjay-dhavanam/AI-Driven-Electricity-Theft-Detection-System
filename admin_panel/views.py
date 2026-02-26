"""
Admin Panel Views for Electricity Theft Detection System
Only accessible to superusers
"""
from core.models import InjectionManager
from django.contrib.auth.decorators import user_passes_test
from django.db.models import Count, Avg, Max, Min, Q
from django.utils import timezone
from datetime import datetime, timedelta
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.utils import timezone
from datetime import datetime, timedelta
import json
from core.adaptive_learning import AdaptiveLearningManager
from core.models import Meter, AnomalyInjection, RetrainingLog, ModelVersion
from core.adaptive_learning import AdaptiveLearningEngine, AdaptiveLearningManager
from core.models import Meter, DailyConsumption, AnomalyResult, ModelVersion, RetrainingLog

def is_superuser(user):
    """Check if user is superuser"""
    return user.is_authenticated and user.is_superuser


def overview(request):
    """
    Admin Overview Page - Shows all meters with latest data using historical dates
    """
    # Apply date filter if provided
    date_filter = request.GET.get('date')
    if date_filter:
        try:
            filter_date = datetime.strptime(date_filter, '%Y-%m-%d').date()
        except:
            # Use max available date if invalid
            max_consumption_date = DailyConsumption.objects.aggregate(Max('date'))['date__max']
            filter_date = max_consumption_date or datetime(2010, 12, 31).date()
    else:
        # Use max available date from consumption data
        max_consumption_date = DailyConsumption.objects.aggregate(Max('date'))['date__max']
        filter_date = max_consumption_date or datetime(2010, 12, 31).date()
    
    # Get all active meters
    meters = Meter.objects.filter(is_active=True).order_by('meter_id')
    
    # Get latest data for each meter
    meter_data = []
    for meter in meters:
        # Get latest consumption BEFORE OR ON filter_date
        latest_consumption = DailyConsumption.objects.filter(
            meter=meter,
            date__lte=filter_date
        ).order_by('-date').first()
        
        # Get latest anomaly result with injection check BEFORE OR ON filter_date
        latest_anomaly_query = AnomalyResult.objects.filter(
            meter=meter,
            date__lte=filter_date
        ).order_by('-date')
        
        latest_anomaly = latest_anomaly_query.first()
        
        # Check if there's an active injection for this meter
        injection_for_date = AnomalyInjection.objects.filter(
            meter=meter,
            date__lte=filter_date,
            is_active=True
        ).order_by('-date').first()
        
        # If injection exists and no anomaly result for that date, create a synthetic one
        if injection_for_date:
            # Check if we have an anomaly for the injection date
            anomaly_for_injection = AnomalyResult.objects.filter(
                meter=meter,
                date=injection_for_date.date
            ).first()
            
            if not anomaly_for_injection:
                # Create synthetic anomaly from injection
                latest_anomaly = AnomalyResult(
                    meter=meter,
                    date=injection_for_date.date,
                    classification=injection_for_date.injection_type,
                    anomaly_score=injection_for_date.injected_score,
                    confidence=injection_for_date.injected_confidence,
                    is_injected=True,
                    injection_reason=injection_for_date.reason
                )
            elif latest_anomaly and latest_anomaly.date == injection_for_date.date:
                # Apply injection to the anomaly result
                InjectionManager.apply_injection_to_result(latest_anomaly)
        
        # Get model info
        active_model = ModelVersion.objects.filter(
            meter=meter,
            is_active=True
        ).first()
        
        # Get 30-day stats with injection consideration
        thirty_days_before = filter_date - timedelta(days=30)
        
        # Get all anomalies in last 30 days
        recent_anomalies = AnomalyResult.objects.filter(
            meter=meter,
            date__gte=thirty_days_before,
            date__lte=filter_date
        )
        
        # Count statistics
        normal_count = 0
        suspicious_count = 0
        theft_count = 0
        
        for anomaly in recent_anomalies:
            # Check for injection on each anomaly date
            injection = AnomalyInjection.objects.filter(
                meter=meter,
                date=anomaly.date,
                is_active=True
            ).first()
            
            if injection:
                # Use injection classification
                if injection.injection_type == 'normal':
                    normal_count += 1
                elif injection.injection_type == 'suspicious':
                    suspicious_count += 1
                elif injection.injection_type == 'theft':
                    theft_count += 1
            else:
                # Use anomaly classification
                if anomaly.classification == 'normal':
                    normal_count += 1
                elif anomaly.classification == 'suspicious':
                    suspicious_count += 1
                elif anomaly.classification == 'theft':
                    theft_count += 1
        
        # Get stats with injection overrides
        stats = {
            'normal_count': normal_count,
            'suspicious_count': suspicious_count,
            'theft_count': theft_count,
        }
        
        meter_data.append({
            'meter': meter,
            'latest_consumption': latest_consumption,
            'latest_anomaly': latest_anomaly,
            'active_model': active_model,
            'stats': stats,
        })
    
    # System statistics with injection consideration
    total_meters = Meter.objects.count()
    meters_with_models = ModelVersion.objects.filter(is_active=True).count()
    
    # Get recent anomalies count (last 7 days)
    seven_days_before = filter_date - timedelta(days=7)
    
    recent_anomalies_all = AnomalyResult.objects.filter(
        date__gte=seven_days_before,
        date__lte=filter_date
    )
    
    # Count anomalies with injection overrides
    recent_anomalies_count = 0
    recent_suspicious_count = 0
    recent_thefts_count = 0
    
    for anomaly in recent_anomalies_all:
        injection = AnomalyInjection.objects.filter(
            meter=anomaly.meter,
            date=anomaly.date,
            is_active=True
        ).first()
        
        if injection:
            if injection.injection_type == 'suspicious':
                recent_suspicious_count += 1
                recent_anomalies_count += 1
            elif injection.injection_type == 'theft':
                recent_thefts_count += 1
                recent_anomalies_count += 1
        else:
            if anomaly.classification == 'suspicious':
                recent_suspicious_count += 1
                recent_anomalies_count += 1
            elif anomaly.classification == 'theft':
                recent_thefts_count += 1
                recent_anomalies_count += 1
    
    # Get date range for filter
    min_date = DailyConsumption.objects.aggregate(Min('date'))['date__min']
    max_date = DailyConsumption.objects.aggregate(Max('date'))['date__max']
    
    context = {
        'meter_data': meter_data,
        'total_meters': total_meters,
        'meters_with_models': meters_with_models,
        'recent_anomalies': recent_anomalies_count,
        'recent_suspicious': recent_suspicious_count,
        'recent_thefts': recent_thefts_count,
        'current_date': filter_date,
        'min_date': min_date,
        'max_date': max_date,
    }
    
    return render(request, 'admin_panel/overview.html', context)

from datetime import date, datetime, timedelta

def meter_detail(request, meter_id):
    meter = get_object_or_404(Meter, meter_id=meter_id)

    days_param = request.GET.get('days', '30')

    # DEFAULTS (important)
    days = 30
    end_date = timezone.now().date()
    start_date = end_date - timedelta(days=days)

    # -----------------------------
    # HANDLE CUSTOM RANGE
    # -----------------------------
    if days_param == 'custom':
        start_date_str = request.GET.get('start_date')
        end_date_str = request.GET.get('end_date')

        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

            # Calculate days for UI + stats
            days = (end_date - start_date).days or 1

        except (TypeError, ValueError):
            # fallback to last 30 days
            days = 30
            end_date = timezone.now().date()
            start_date = end_date - timedelta(days=days)

    # -----------------------------
    # HANDLE PREDEFINED RANGE
    # -----------------------------
    else:
        try:
            days = int(days_param)
        except ValueError:
            days = 30

        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=days)

    end_date = request.GET.get('end_date', timezone.now().date())
    
    if isinstance(end_date, str):
        try:
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        except:
            end_date = timezone.now().date()
    
    start_date = end_date - timedelta(days=days)
    
    # Get consumption data
    consumption_data = DailyConsumption.objects.filter(
        meter=meter,
        date__gte=start_date,
        date__lte=end_date
    ).order_by('date')
    
    # Get anomaly results
    anomaly_results = AnomalyResult.objects.filter(
        meter=meter,
        date__gte=start_date,
        date__lte=end_date
    ).order_by('date')
    
    # Get model versions
    model_versions = ModelVersion.objects.filter(
        meter=meter
    ).order_by('-version_number')
    
    # Prepare data for charts
    chart_data = {
        'dates': [],
        'consumption': [],
        'anomaly_scores': [],
        'classifications': [],
    }
    
    # Create a dictionary of anomaly results for easy lookup
    anomaly_dict = {ar.date: ar for ar in anomaly_results}
    # Attach anomaly to each consumption object (IMPORTANT)
    consumption_list = []
    for consumption in consumption_data:
        consumption.anomaly = anomaly_dict.get(consumption.date)
        consumption_list.append(consumption)

    
    for consumption in consumption_data:
        chart_data['dates'].append(consumption.date.strftime('%Y-%m-%d'))
        chart_data['consumption'].append(consumption.total_consumption)
        
        # Get anomaly data if exists
        if consumption.date in anomaly_dict:
            anomaly = anomaly_dict[consumption.date]
            chart_data['anomaly_scores'].append(anomaly.anomaly_score)
            
            # Convert classification to numeric for chart
            if anomaly.classification == 'normal':
                chart_data['classifications'].append(0)
            elif anomaly.classification == 'suspicious':
                chart_data['classifications'].append(1)
            else:  # theft
                chart_data['classifications'].append(2)
        else:
            chart_data['anomaly_scores'].append(None)
            chart_data['classifications'].append(None)
    
    # Get statistics
    stats = {
        'total_days': consumption_data.count(),
        'avg_consumption': consumption_data.aggregate(avg=Avg('total_consumption'))['avg'] or 0,
        'max_consumption': consumption_data.aggregate(max=Max('total_consumption'))['max'] or 0,
        'min_consumption': consumption_data.aggregate(min=Min('total_consumption'))['min'] or 0,
        'normal_days': anomaly_results.filter(classification='normal').count(),
        'suspicious_days': anomaly_results.filter(classification='suspicious').count(),
        'theft_days': anomaly_results.filter(classification='theft').count(),
    }
    
    # Half-hour consumption for latest day
    latest_consumption = consumption_data.last()
    half_hour_data = []
    if latest_consumption and latest_consumption.consumption_data:
        for i in range(48):
            key = f'hh_{i}'
            hour = i // 2
            minute = (i % 2) * 30
            time_str = f'{hour:02d}:{minute:02d}'
            
            half_hour_data.append({
                'time': time_str,
                'consumption': latest_consumption.consumption_data.get(key, 0),
                'hour': hour,
                'minute': minute,
            })
    
    context = {
        'meter': meter,
        'consumption_data': consumption_data,
        'anomaly_results': anomaly_results,
        'anomaly_dict': anomaly_dict,   # ✅ ADD THIS
        'model_versions': model_versions,
        'chart_data': json.dumps(chart_data),
        'consumption_data': consumption_list,
        'stats': stats,
        'half_hour_data': half_hour_data,
        'days': days,
        'start_date': start_date,
        'end_date': end_date,
        'date_range': f"{start_date} to {end_date}",
    }

    
    return render(request, 'admin_panel/meter_detail.html', context)


def analytics(request):
    """
    System Analytics Page - Shows overall system statistics and trends using historical data
    """
    # Get date filter from request or use max available date
    date_filter = request.GET.get('date')
    if date_filter:
        try:
            filter_date = datetime.strptime(date_filter, '%Y-%m-%d').date()
        except:
            max_date = DailyConsumption.objects.aggregate(Max('date'))['date__max']
            filter_date = max_date or datetime(2010, 12, 31).date()
    else:
        max_date = DailyConsumption.objects.aggregate(Max('date'))['date__max']
        filter_date = max_date or datetime(2010, 12, 31).date()
    
    # Calculate date ranges relative to filter_date
    last_7_days = filter_date - timedelta(days=7)
    last_30_days = filter_date - timedelta(days=30)
    last_90_days = filter_date - timedelta(days=90)

    total_meters = Meter.objects.count()
    active_meters = Meter.objects.filter(is_active=True).count()

    # -----------------------------
    # Helper function with injection consideration
    # -----------------------------
    def get_anomaly_stats(start_date, end_date=None):
        if end_date is None:
            end_date = filter_date
        
        anomalies = AnomalyResult.objects.filter(
            date__gte=start_date,
            date__lte=end_date
        )
        
        normal_count = 0
        suspicious_count = 0
        theft_count = 0
        total_confidence = 0
        total_score = 0
        count = 0
        
        for anomaly in anomalies:
            # Check for injection
            injection = AnomalyInjection.objects.filter(
                meter=anomaly.meter,
                date=anomaly.date,
                is_active=True
            ).first()
            
            if injection:
                classification = injection.injection_type
                confidence = injection.injected_confidence
                score = injection.injected_score
            else:
                classification = anomaly.classification
                confidence = anomaly.confidence
                score = anomaly.anomaly_score
            
            if classification == 'normal':
                normal_count += 1
            elif classification == 'suspicious':
                suspicious_count += 1
            elif classification == 'theft':
                theft_count += 1
            
            total_confidence += confidence
            total_score += score
            count += 1
        
        total = normal_count + suspicious_count + theft_count
        detected = suspicious_count + theft_count
        
        stats = {
            'total': total,
            'normal': normal_count,
            'suspicious': suspicious_count,
            'theft': theft_count,
            'detection_rate': round((detected / total) * 100, 1) if total > 0 else 0.0,
            'avg_confidence': round(total_confidence / count, 3) if count > 0 else 0.0,
            'avg_score': round(total_score / count, 3) if count > 0 else 0.0,
        }
        
        return stats

    # -----------------------------
    # Stats by period
    # -----------------------------
    stats_7d = get_anomaly_stats(last_7_days, filter_date)
    stats_30d = get_anomaly_stats(last_30_days, filter_date)
    stats_90d = get_anomaly_stats(last_90_days, filter_date)

    # -----------------------------
    # Model statistics
    # -----------------------------
    model_stats = {
        'total_models': ModelVersion.objects.count(),
        'active_models': ModelVersion.objects.filter(is_active=True).count(),
        'avg_training_samples': ModelVersion.objects.filter(is_trained=True)
            .aggregate(avg=Avg('training_samples'))['avg'] or 0,
        'models_retrained_recently': RetrainingLog.objects.filter(
            completed_at__gte=filter_date - timedelta(days=30)
        ).count(),
    }

    # -----------------------------
    # Daily anomaly trend (30 days) with injection consideration
    # -----------------------------
    daily_trend = []
    for i in range(30, 0, -1):
        date = filter_date - timedelta(days=i)
        daily_anomalies = AnomalyResult.objects.filter(date=date)
        
        normal = 0
        suspicious = 0
        theft = 0
        
        for anomaly in daily_anomalies:
            injection = AnomalyInjection.objects.filter(
                meter=anomaly.meter,
                date=date,
                is_active=True
            ).first()
            
            if injection:
                if injection.injection_type == 'normal':
                    normal += 1
                elif injection.injection_type == 'suspicious':
                    suspicious += 1
                elif injection.injection_type == 'theft':
                    theft += 1
            else:
                if anomaly.classification == 'normal':
                    normal += 1
                elif anomaly.classification == 'suspicious':
                    suspicious += 1
                elif anomaly.classification == 'theft':
                    theft += 1
        
        daily_trend.append({
            'date': date.strftime('%Y-%m-%d'),
            'total': normal + suspicious + theft,
            'normal': normal,
            'suspicious': suspicious,
            'theft': theft,
        })

    # -----------------------------
    # Top anomaly meters (last 30 days)
    # -----------------------------
    top_anomaly_meters = []
    all_meters = Meter.objects.all()
    
    for meter in all_meters:
        # Get anomaly counts for last 30 days
        anomalies_30d = AnomalyResult.objects.filter(
            meter=meter,
            date__gte=last_30_days,
            date__lte=filter_date
        )
        
        anomaly_count = anomalies_30d.count()
        suspicious_count = 0
        theft_count = 0
        
        for anomaly in anomalies_30d:
            injection = AnomalyInjection.objects.filter(
                meter=meter,
                date=anomaly.date,
                is_active=True
            ).first()
            
            if injection:
                if injection.injection_type == 'suspicious':
                    suspicious_count += 1
                elif injection.injection_type == 'theft':
                    theft_count += 1
            else:
                if anomaly.classification == 'suspicious':
                    suspicious_count += 1
                elif anomaly.classification == 'theft':
                    theft_count += 1
        
        if anomaly_count > 0:
            top_anomaly_meters.append({
                'meter_id': meter.meter_id,
                'anomaly_count': anomaly_count,
                'suspicious_count': suspicious_count,
                'theft_count': theft_count,
                'latest_anomaly_date': anomalies_30d.order_by('-date').first().date if anomalies_30d.exists() else None,
                'is_active': meter.is_active
            })
    
    # Sort by anomaly count descending
    top_anomaly_meters.sort(key=lambda x: x['anomaly_count'], reverse=True)
    top_anomaly_meters = top_anomaly_meters[:10]  # Top 10 only

    # Get date range for filter
    min_date = DailyConsumption.objects.aggregate(Min('date'))['date__min']
    max_date = DailyConsumption.objects.aggregate(Max('date'))['date__max']
    
    context = {
        'filter_date': filter_date,
        'total_meters': total_meters,
        'active_meters': active_meters,
        'stats_7d': stats_7d,
        'stats_30d': stats_30d,
        'stats_90d': stats_90d,
        'meters_without_models': total_meters - model_stats['active_models'],
        'model_stats': model_stats,
        'daily_trend': json.dumps(daily_trend),
        'top_anomaly_meters': top_anomaly_meters,
        'detection_rate_30d': stats_30d['detection_rate'],
        'min_date': min_date,
        'max_date': max_date,
    }
    
    return render(request, 'admin_panel/analytics.html', context)


def retraining_control(request):
    """
    Adaptive Learning Control Panel
    """
    # Get recent retraining logs
    recent_logs = RetrainingLog.objects.all().order_by('-started_at')[:10]
    
    # Meters needing retraining
    
    meters_needing = AdaptiveLearningManager.get_meters_needing_retraining()
    
    # System health
    system_health = AdaptiveLearningManager.get_system_health()
    
    context = {
        'recent_logs': recent_logs,
        'meters_needing': meters_needing,
        'system_health': system_health,
        'total_meters': Meter.objects.count(),
    }
    
    return render(request, 'admin_panel/retraining_control.html', context)


# Add these views to existing views.py


@user_passes_test(is_superuser)
def injection_management(request):
    """
    Manage anomaly injections
    """
    # Get filter parameters
    meter_id = request.GET.get('meter_id')
    injection_type = request.GET.get('type')
    active_only = request.GET.get('active', 'true') == 'true'
    
    # Build filters
    filters = {}
    if meter_id:
        filters['meter__meter_id'] = meter_id
    if injection_type:
        filters['injection_type'] = injection_type
    if active_only:
        filters['is_active'] = True
    
    # Get injections
    injections = AnomalyInjection.objects.filter(**filters).order_by('-date', '-created_at')
    
    # Get all meters for filter dropdown
    all_meters = Meter.objects.filter(is_active=True).order_by('meter_id')
    
    # Get statistics
    stats = {
        'total': injections.count(),
        'active': injections.filter(is_active=True).count(),
        'normal': injections.filter(injection_type='normal').count(),
        'suspicious': injections.filter(injection_type='suspicious').count(),
        'theft': injections.filter(injection_type='theft').count(),
        'full_day': injections.filter(time_slot='full_day').count(),
        'time_slot': injections.exclude(time_slot='full_day').count(),
    }
    
    context = {
        'injections': injections,
        'all_meters': all_meters,
        'stats': stats,
        'filter_meter_id': meter_id,
        'filter_type': injection_type,
        'filter_active': active_only,
    }
    
    return render(request, 'admin_panel/injection_management.html', context)

@user_passes_test(is_superuser)
def create_injection(request):
    """
    Create a new anomaly injection
    """
    if request.method == 'POST':
        try:
            # Get form data
            meter_id = request.POST.get('meter_id')
            date_str = request.POST.get('date')
            time_slot = request.POST.get('time_slot', 'full_day')
            injection_type = request.POST.get('injection_type')
            reason = request.POST.get('reason', '')
            
            # Validate required fields
            if not all([meter_id, date_str, injection_type]):
                messages.error(request, 'Missing required fields')
                return redirect('injection_management')
            
            # Get meter
            meter = get_object_or_404(Meter, meter_id=meter_id)
            
            # Parse date
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            
            # Check if injection already exists
            existing = AnomalyInjection.objects.filter(
                meter=meter,
                date=date,
                time_slot=time_slot,
                is_active=True
            ).exists()
            
            if existing:
                messages.warning(request, f'Active injection already exists for {meter_id} on {date}')
                return redirect('injection_management')
            
            # Create injection
            injection = AnomalyInjection.objects.create(
                meter=meter,
                date=date,
                time_slot=time_slot,
                injection_type=injection_type,
                reason=reason,
                created_by=request.user,
                is_active=True
            )
            
            messages.success(request, 
                f'Injection created: {meter_id} on {date} marked as {injection_type}'
            )
            return redirect('injection_management')
            
        except Exception as e:
            messages.error(request, f'Error creating injection: {str(e)}')
            return redirect('injection_management')
    
    # GET request - show form
    meters = Meter.objects.filter(is_active=True).order_by('meter_id')
    today = timezone.now().date()
    
    context = {
        'meters': meters,
        'today': today,
        'time_slot_choices': AnomalyInjection.TIME_SLOT_CHOICES,
        'injection_type_choices': AnomalyInjection.INJECTION_TYPE_CHOICES,
    }
    
    return render(request, 'admin_panel/create_injection.html', context)

@user_passes_test(is_superuser)
def edit_injection(request, injection_id):
    """
    Edit an existing injection
    """
    injection = get_object_or_404(AnomalyInjection, id=injection_id)
    
    if request.method == 'POST':
        try:
            # Update fields
            injection.time_slot = request.POST.get('time_slot', injection.time_slot)
            injection.injection_type = request.POST.get('injection_type', injection.injection_type)
            injection.injected_score = float(request.POST.get('injected_score', injection.injected_score))
            injection.injected_confidence = float(request.POST.get('injected_confidence', injection.injected_confidence))
            injection.reason = request.POST.get('reason', injection.reason)
            injection.is_active = request.POST.get('is_active') == 'true'
            
            # Handle expiration
            expires_at = request.POST.get('expires_at')
            if expires_at:
                injection.expires_at = datetime.strptime(expires_at, '%Y-%m-%d')
            
            injection.save()
            
            messages.success(request, 'Injection updated successfully')
            return redirect('injection_management')
            
        except Exception as e:
            messages.error(request, f'Error updating injection: {str(e)}')
    
    context = {
        'injection': injection,
        'time_slot_choices': AnomalyInjection.TIME_SLOT_CHOICES,
        'injection_type_choices': AnomalyInjection.INJECTION_TYPE_CHOICES,
    }
    
    return render(request, 'admin_panel/edit_injection.html', context)

@user_passes_test(is_superuser)
def delete_injection(request, injection_id):
    """
    Delete an injection
    """
    injection = get_object_or_404(AnomalyInjection, id=injection_id)
    
    if request.method == 'POST':
        meter_id = injection.meter.meter_id
        injection.delete()
        messages.success(request, f'Injection for {meter_id} deleted')
        return redirect('injection_management')
    
    context = {'injection': injection}
    return render(request, 'admin_panel/delete_injection.html', context)

@user_passes_test(is_superuser)
def injection_preview(request):
    """
    Preview injection impact
    """
    if request.method == 'POST':
        try:
            meter_id = request.POST.get('meter_id')
            date_str = request.POST.get('date')
            injection_type = request.POST.get('injection_type')
            
            if not all([meter_id, date_str, injection_type]):
                return JsonResponse({'error': 'Missing parameters'}, status=400)
            
            # Get meter
            meter = get_object_or_404(Meter, meter_id=meter_id)
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            
            # Get current anomaly result (if exists)
            current_result = None
            try:
                current_result = AnomalyResult.objects.get(
                    meter=meter,
                    date=date
                )
            except AnomalyResult.DoesNotExist:
                pass
            
            # Get typical values for injection type
            typical_scores = {
                'normal': {'score': 0.15, 'confidence': 0.95},
                'suspicious': {'score': 0.0, 'confidence': 0.85},
                'theft': {'score': -0.2, 'confidence': 0.90},
            }
            
            typical = typical_scores.get(injection_type, {'score': 0.0, 'confidence': 0.9})
            
            # Prepare preview data
            preview = {
                'current': {
                    'exists': current_result is not None,
                    'classification': current_result.classification if current_result else 'No result',
                    'score': current_result.anomaly_score if current_result else 0.0,
                    'confidence': current_result.confidence if current_result else 0.0,
                },
                'injection': {
                    'type': injection_type,
                    'score': typical['score'],
                    'confidence': typical['confidence'],
                    'impact': 'Will override ML prediction' if current_result else 'Will create new result',
                }
            }
            
            return JsonResponse({'preview': preview})
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

@user_passes_test(is_superuser)
def enhanced_retraining_control(request):
    """
    Enhanced adaptive learning control panel
    """
    # Get system health
    system_health = AdaptiveLearningManager.get_system_health()
    
    # Get meters needing retraining
    meters_needing = AdaptiveLearningManager.get_meters_needing_retraining()
    
    # Get recent retraining logs
    recent_logs = RetrainingLog.objects.all().order_by('-started_at')[:10]
    
    # Get all meters with active models
    meters_with_models = Meter.objects.filter(
        model_versions__is_active=True
    ).distinct().order_by('meter_id')

    
    context = {
        'system_health': system_health,
        'meters_needing': meters_needing,
        'recent_logs': recent_logs,
        'meters_with_models': meters_with_models,
    }
    
    return render(request, 'admin_panel/enhanced_retraining.html', context)

@user_passes_test(is_superuser)
def trigger_retraining(request):
    """
    Trigger adaptive retraining (AJAX endpoint)
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            scope = data.get('scope', 'needing')  # 'needing', 'selected', 'all'
            selected_meters = data.get('selected_meters', [])
            job_name = data.get('job_name', f'Manual Retraining - {timezone.now().date()}')
            
            # Determine which meters to retrain
            if scope == 'selected' and selected_meters:
                meter_ids = selected_meters
            elif scope == 'all':
                # Get all meters with active models
                meter_ids = list(
                Meter.objects.filter(
                    model_versions__is_active=True
                ).values_list('meter_id', flat=True).distinct()
            )

            else:  # 'needing'
                meters_needing = AdaptiveLearningManager.get_meters_needing_retraining()
                meter_ids = [m['meter_id'] for m in meters_needing]
            
            # Create retraining job
            retraining_log = AdaptiveLearningEngine.create_retraining_job(
                name=job_name,
                user=request.user,
                window_days=30,
                validation_days=7,
                improvement_threshold=5.0
            )
            
            # Start retraining (this would be async in production)
            # For demo, we'll run it synchronously
            import threading
            
            def run_retraining():
                AdaptiveLearningEngine.retrain_meter_models(
                    retraining_log=retraining_log,
                    meter_ids=meter_ids,
                    batch_size=5
                )
            
            # Run in thread for better UX
            thread = threading.Thread(target=run_retraining)
            thread.daemon = True
            thread.start()
            
            return JsonResponse({
                'success': True,
                'job_id': retraining_log.id,
                'meter_count': len(meter_ids),
                'message': f'Retraining started for {len(meter_ids)} meters'
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@user_passes_test(is_superuser)
def retraining_status(request, job_id):
    """
    Get retraining job status (AJAX endpoint)
    """
    retraining_log = get_object_or_404(RetrainingLog, id=job_id)
    
    return JsonResponse({
        'id': retraining_log.id,
        'status': retraining_log.status,
        'progress': {
            'total': retraining_log.total_meters,
            'success': retraining_log.success_count,
            'failed': retraining_log.failed_count,
            'improved': retraining_log.improved_count,
        },
        'details': {
            'started_at': retraining_log.started_at.strftime('%Y-%m-%d %H:%M:%S') if retraining_log.started_at else None,
            'completed_at': retraining_log.completed_at.strftime('%Y-%m-%d %H:%M:%S') if retraining_log.completed_at else None,
            'duration': retraining_log.duration_seconds,
        }
    })

@user_passes_test(is_superuser)
def injection_impact_report(request):
    """
    Report on injection impact
    """
    # Get all injections
    injections = AnomalyInjection.objects.all().order_by('-date')
    
    # Calculate statistics
    total_injections = injections.count()
    active_injections = injections.filter(is_active=True).count()
    
    # Group by type
    by_type = {}
    for injection in injections:
        injection_type = injection.get_injection_type_display()
        by_type[injection_type] = by_type.get(injection_type, 0) + 1
    
    # Group by meter
    by_meter = {}
    for injection in injections:
        meter_id = injection.meter.meter_id
        by_meter[meter_id] = by_meter.get(meter_id, 0) + 1
    
    # Get top meters with injections
    top_meters = sorted(by_meter.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Recent injections (last 30 days)
    thirty_days_ago = timezone.now().date() - timedelta(days=30)
    recent_injections = injections.filter(date__gte=thirty_days_ago)
    
    context = {
        'total_injections': total_injections,
        'active_injections': active_injections,
        'by_type': by_type,
        'top_meters': top_meters,
        'recent_injections': recent_injections,
        'injections': injections[:50],  # Show first 50
    }
    
    return render(request, 'admin_panel/injection_report.html', context)


# Add these API endpoints
@user_passes_test(is_superuser)
def api_injection_details(request, injection_id):
    """Get injection details (AJAX)"""
    injection = get_object_or_404(AnomalyInjection, id=injection_id)
    
    return JsonResponse({
        'meter_id': injection.meter.meter_id,
        'date': injection.date.strftime('%Y-%m-%d'),
        'time_slot': injection.get_time_slot_display(),
        'type': injection.get_injection_type_display(),
        'score': injection.injected_score,
        'confidence': injection.injected_confidence,
        'reason': injection.reason,
        'consumption_override': injection.consumption_override,
        'is_active': injection.is_active,
        'created_at': injection.created_at.strftime('%Y-%m-%d %H:%M'),
        'created_by': injection.created_by.username if injection.created_by else 'System',
    })

@user_passes_test(is_superuser)
def api_meter_info(request, meter_id):
    """Get meter info (AJAX)"""
    meter = get_object_or_404(Meter, meter_id=meter_id)
    
    return JsonResponse({
        'meter_id': meter.meter_id,
        'location': meter.location or 'N/A',
        'acorn_group': meter.acorn_group or 'N/A',
        'is_active': meter.is_active,
        'has_model': ModelVersion.objects.filter(meter=meter, is_active=True).exists(),
        'consumption_count': DailyConsumption.objects.filter(meter=meter).count(),
    })

@user_passes_test(is_superuser)
def api_deactivate_expired(request):
    """Deactivate expired injections (AJAX)"""
    
    count = InjectionManager.deactivate_expired_injections()
    
    return JsonResponse({'count': count})

@user_passes_test(is_superuser)
def api_bulk_deactivate(request):
    """Deactivate all injections (AJAX)"""
    count = AnomalyInjection.objects.filter(is_active=True).update(is_active=False)
    
    return JsonResponse({'count': count})


# Add at the top of your views.py, after existing imports
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import render, redirect
from django.contrib import messages

# Add these login/logout views
def admin_login(request):
    """
    Admin login view
    """
    # If user is already authenticated, redirect to dashboard
        
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        if username == 'admin' and password == 'admin':
            messages.success(request, 'Login Successful')
            return redirect('admin_overview')
        else:
            messages.error(request, 'Invalid details !')
            return redirect('admin_login')

    return render(request, 'admin_panel/login.html')

def admin_logout(request):
    """
    Admin logout view
    """
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('admin_login')
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.core.paginator import Paginator
from django.db.models import Q, Count
from django.utils import timezone
from django.db import models
import json
from core.models import Feedback, Meter
from users.forms import FeedbackForm

@staff_member_required
def feedback_management(request):
    """Admin view to manage all feedbacks"""
    status_filter = request.GET.get('status', 'all')
    search_query = request.GET.get('search', '')
    
    # Get all feedbacks
    feedbacks = Feedback.objects.all().order_by('-created_at')
    
    # Apply filters
    if status_filter != 'all':
        feedbacks = feedbacks.filter(status=status_filter)
    
    if search_query:
        feedbacks = feedbacks.filter(
            Q(id_number__icontains=search_query) |
            Q(feedback_text__icontains=search_query) |
            Q(user__username__icontains=search_query) |
            Q(user__email__icontains=search_query)
        )
    
    # Pagination
    paginator = Paginator(feedbacks, 15)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Statistics
    stats = {
        'total': Feedback.objects.count(),
        'pending': Feedback.objects.filter(status='pending').count(),
        'investigation': Feedback.objects.filter(status='investigation').count(),
        'completed': Feedback.objects.filter(status='completed').count(),
    }
    
    # Check if users app is installed
    from django.apps import apps
    installed_apps = [app.name for app in apps.get_app_configs()]
    
    return render(request, 'admin_panel/feedback_management.html', {
        'feedbacks': page_obj,
        'stats': stats,
        'status_filter': status_filter,
        'search_query': search_query,
        'active_page': 'feedback',
        'page_obj': page_obj,
        'installed_apps': installed_apps,
    })

# SIMPLE STATUS UPDATE FUNCTIONS (No AJAX, page reload)
@staff_member_required
def admin_mark_pending(request, feedback_id):
    """Mark feedback as pending"""
    try:
        feedback = Feedback.objects.get(id=feedback_id)
        feedback.status = 'pending'
        feedback.save()
        messages.success(request, f'Feedback #{feedback_id} marked as Pending')
    except Feedback.DoesNotExist:
        messages.error(request, f'Feedback #{feedback_id} not found')
    
    return redirect('admin_feedback_management')

@staff_member_required
def admin_mark_investigation(request, feedback_id):
    """Mark feedback as under investigation"""
    try:
        feedback = Feedback.objects.get(id=feedback_id)
        feedback.status = 'investigation'
        feedback.save()
        messages.success(request, f'Feedback #{feedback_id} marked as Under Investigation')
    except Feedback.DoesNotExist:
        messages.error(request, f'Feedback #{feedback_id} not found')
    
    return redirect('admin_feedback_management')

@staff_member_required
def admin_mark_completed(request, feedback_id):
    """Mark feedback as completed"""
    try:
        feedback = Feedback.objects.get(id=feedback_id)
        feedback.status = 'completed'
        feedback.save()
        messages.success(request, f'Feedback #{feedback_id} marked as Completed')
    except Feedback.DoesNotExist:
        messages.error(request, f'Feedback #{feedback_id} not found')
    
    return redirect('admin_feedback_management')

@staff_member_required
def admin_mark_completed_with_response(request, feedback_id):
    """Mark feedback as completed with response"""
    try:
        feedback = Feedback.objects.get(id=feedback_id)
        feedback.status = 'completed'
        
        # Get response from POST data
        admin_response = request.POST.get('admin_response', '')
        if admin_response:
            feedback.admin_response = admin_response
        
        feedback.save()
        messages.success(request, f'Feedback #{feedback_id} marked as Completed with response')
    except Feedback.DoesNotExist:
        messages.error(request, f'Feedback #{feedback_id} not found')
    
    return redirect('admin_feedback_management')

# AJAX version (optional, for no page reload)
@staff_member_required
@require_POST
def update_feedback_status_ajax(request):
    """AJAX endpoint to update feedback status (for no page reload)"""
    try:
        feedback_id = request.POST.get('feedback_id')
        new_status = request.POST.get('status')
        admin_response = request.POST.get('admin_response', '')
        
        if not all([feedback_id, new_status]):
            return JsonResponse({
                'success': False, 
                'error': 'Missing parameters'
            })
        
        feedback = Feedback.objects.get(id=feedback_id)
        feedback.status = new_status
        
        if admin_response:
            feedback.admin_response = admin_response
        
        feedback.save()
        
        # Update statistics
        stats = {
            'total': Feedback.objects.count(),
            'pending': Feedback.objects.filter(status='pending').count(),
            'investigation': Feedback.objects.filter(status='investigation').count(),
            'completed': Feedback.objects.filter(status='completed').count(),
        }
        
        return JsonResponse({
            'success': True,
            'message': f'Status updated to {new_status}',
            'new_status': feedback.get_status_display(),
            'new_status_badge': feedback.get_status_badge(),
            'new_status_icon': feedback.get_status_icon(),
            'stats': stats,
        })
        
    except Feedback.DoesNotExist:
        return JsonResponse({
            'success': False, 
            'error': f'Feedback with ID {feedback_id} not found'
        })
    except Exception as e:
        return JsonResponse({
            'success': False, 
            'error': str(e)
        })

@staff_member_required
def feedback_detail(request, feedback_id):
    """Admin view to see feedback details and update status"""
    feedback = get_object_or_404(Feedback, id=feedback_id)
    
    if request.method == 'POST':
        form = FeedbackForm(request.POST, instance=feedback)
        if form.is_valid():
            form.save()
            messages.success(request, f'Feedback #{feedback.id} updated successfully!')
            return redirect('admin_feedback_management')
    else:
        form = FeedbackForm(instance=feedback)
    
    # Get user's other feedbacks
    user_feedbacks = Feedback.objects.filter(user=feedback.user).exclude(id=feedback.id)[:5]
    
    return render(request, 'admin_panel/feedback_detail.html', {
        'feedback': feedback,
        'form': form,
        'user_feedbacks': user_feedbacks,
        'active_page': 'feedback',
    })

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST
from django.core.paginator import Paginator
from django.db.models import Q, Count, F, ExpressionWrapper, fields
from django.db.models.functions import TruncMonth, TruncDay, ExtractHour
from django.utils import timezone
from django.contrib.auth.models import User
from django.db import models
import json
import csv
from datetime import datetime, timedelta
from collections import defaultdict
import calendar
from core.models import Feedback, Meter
from users.forms import FeedbackForm

@staff_member_required
def feedback_analytics(request):
    """Admin view for feedback analytics with working charts"""
    
    # Get time period from request (default: 30 days)
    period = request.GET.get('period', '30days')
    
    # Calculate date range based on period
    end_date = timezone.now()
    if period == '7days':
        start_date = end_date - timedelta(days=7)
        date_format = '%Y-%m-%d'
    elif period == '30days':
        start_date = end_date - timedelta(days=30)
        date_format = '%Y-%m-%d'
    elif period == '90days':
        start_date = end_date - timedelta(days=90)
        date_format = '%Y-%m'
    elif period == '1year':
        start_date = end_date - timedelta(days=365)
        date_format = '%Y-%m'
    else:
        start_date = end_date - timedelta(days=30)
        period = '30days'
        date_format = '%Y-%m-%d'
    
    

    # Get all feedbacks within the period
    feedbacks_in_period = Feedback.objects.filter(
        created_at__range=[start_date, end_date]
    )
    
    # Overall statistics
    total_feedbacks = Feedback.objects.count()
    total_in_period = feedbacks_in_period.count()
    
    stats = {
        'total': total_feedbacks,
        'period_total': total_in_period,
        'pending': Feedback.objects.filter(status='pending').count(),
        'investigation': Feedback.objects.filter(status='investigation').count(),
        'completed': Feedback.objects.filter(status='completed').count(),
        'today': Feedback.objects.filter(created_at__date=timezone.now().date()).count(),
        'yesterday': Feedback.objects.filter(
            created_at__date=timezone.now().date() - timedelta(days=1)
        ).count(),
        'this_week': Feedback.objects.filter(
            created_at__gte=timezone.now() - timedelta(days=7)
        ).count(),
        'this_month': Feedback.objects.filter(
            created_at__gte=timezone.now() - timedelta(days=30)
        ).count(),
    }
    
    # Add response rate
    responded = Feedback.objects.filter(admin_response__isnull=False).count()
    if total_feedbacks > 0:
        stats['response_rate'] = round((responded / total_feedbacks) * 100, 1)
    else:
        stats['response_rate'] = 0
    
    # Status distribution for the period
    status_data = {
        'pending': feedbacks_in_period.filter(status='pending').count(),
        'investigation': feedbacks_in_period.filter(status='investigation').count(),
        'completed': feedbacks_in_period.filter(status='completed').count(),
    }
    
    # Type distribution
    type_data_raw = feedbacks_in_period.values('feedback_type').annotate(
        count=Count('id')
    ).order_by('-count')
    
    type_data = {}
    type_labels = []
    type_counts = []
    type_colors = ['#6C757D', '#DC3545', '#17A2B8', '#FFC107', '#28A745']
    
    for i, item in enumerate(type_data_raw):
        type_name = dict(Feedback.FEEDBACK_TYPES).get(item['feedback_type'], item['feedback_type'])
        type_data[type_name] = item['count']
        type_labels.append(type_name)
        type_counts.append(item['count'])
    
    # If no data, add placeholder
    if not type_data:
        type_data = {'No Data': 0}
        type_labels = ['No Data']
        type_counts = [0]
        type_colors = ['#6C757D']
    
    # Monthly trend data
    monthly_data = {}
    monthly_labels = []
    monthly_counts = []
    
    # Generate last 6 months
    for i in range(5, -1, -1):
        month_date = end_date - timedelta(days=30*i)
        month_key = month_date.strftime('%b %Y')
        month_start = month_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        next_month = (month_start + timedelta(days=32)).replace(day=1)
        
        count = Feedback.objects.filter(
            created_at__gte=month_start,
            created_at__lt=next_month
        ).count()
        
        monthly_data[month_key] = count
        monthly_labels.append(month_key)
        monthly_counts.append(count)
        status_trend_data = {
    'pending': [],
    'investigation': [],
    'completed': []
}

    
    # Daily trend for selected period
    daily_data = {}
    daily_labels = []
    daily_counts = []
    
    current_date = start_date.date()
    end_date_date = end_date.date()
    
    while current_date <= end_date_date:
        date_key = current_date.strftime('%Y-%m-%d')
        display_key = current_date.strftime('%b %d')
        
        count = Feedback.objects.filter(
            created_at__date=current_date
        ).count()
        
        daily_data[date_key] = count
        daily_labels.append(display_key)
        daily_counts.append(count)
        current_date += timedelta(days=1)
    
    # If period is long, show weekly data instead
    if len(daily_labels) > 31:
        # Convert to weekly data
        daily_data = {}
        daily_labels = []
        daily_counts = []
        
        current_date = start_date.date()
        week_count = 0
        week_total = 0
        
        while current_date <= end_date_date:
            count = Feedback.objects.filter(created_at__date=current_date).count()
            week_total += count
            week_count += 1
            
            if week_count == 7 or current_date == end_date_date:
                week_label = f"Week {len(daily_labels) + 1}"
                daily_data[week_label] = week_total
                daily_labels.append(week_label)
                daily_counts.append(week_total)
                week_count = 0
                week_total = 0
            
            current_date += timedelta(days=1)
    
    # Hourly distribution
    hour_data = {}
    hour_labels = []
    hour_counts = []
    
    for hour in range(24):
        hour_str = f"{hour:02d}:00"
        count = feedbacks_in_period.filter(created_at__hour=hour).count()
        hour_data[hour_str] = count
        hour_labels.append(hour_str)
        hour_counts.append(count)

        am_total = sum(hour_counts[:12])   # 00–11
        pm_total = sum(hour_counts[12:])   # 12–23


        hour_data[hour_str] = count
        hour_labels.append(hour_str)
        hour_counts.append(count)
    
    # User statistics
    total_users = User.objects.filter(feedbacks__isnull=False).distinct().count()
    active_users = User.objects.filter(
        feedbacks__created_at__gte=start_date
    ).distinct().count()
    
    # Get most active user
    most_active = feedbacks_in_period.values('user__username', 'user__email').annotate(
        count=Count('id')
    ).order_by('-count').first()
    
    user_stats = {
        'total_users': total_users,
        'active_users': active_users,
        'most_active_user': most_active,
        'new_users': User.objects.filter(
            date_joined__gte=start_date,
            feedbacks__isnull=False
        ).distinct().count(),
    }
    
    # Response time analysis
    completed_with_response = feedbacks_in_period.filter(
        status='completed',
        admin_response__isnull=False
    )
    
    response_times = []
    for fb in completed_with_response:
        if fb.admin_response:
            response_days = (fb.updated_at - fb.created_at).days
            response_times.append(response_days)
    
    if response_times:
        avg_response_time = sum(response_times) / len(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
    else:
        avg_response_time = 0
        min_response_time = 0
        max_response_time = 0
    
    response_time_stats = {
        'average': round(avg_response_time, 1),
        'min': min_response_time,
        'max': max_response_time,
    }
    
    # Top feedback submitters
    top_submitters = User.objects.filter(feedbacks__isnull=False).annotate(
        feedback_count=Count('feedbacks'),
        last_feedback=models.Max('feedbacks__created_at'),
        pending_count=Count('feedbacks', filter=Q(feedbacks__status='pending')),
        investigation_count=Count('feedbacks', filter=Q(feedbacks__status='investigation')),
        completed_count=Count('feedbacks', filter=Q(feedbacks__status='completed')),
    ).order_by('-feedback_count')[:10]


    
    # Generate data for last 6 months for each status
    for i in range(5, -1, -1):
        month_date = end_date - timedelta(days=30*i)
        month_start = month_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        next_month = (month_start + timedelta(days=32)).replace(day=1)
        
        
        for status in ['pending', 'investigation', 'completed']:
            count = Feedback.objects.filter(
                status=status,
                created_at__gte=month_start,
                created_at__lt=next_month
            ).count()
            status_trend_data[status].append(count)
            status_trend_data = {
    'pending': [],
    'investigation': [],
    'completed': []
}


    # First 7 days total
        first_week_total = sum(daily_counts[:7])

        # Last 7 days total
        last_week_total = sum(daily_counts[-7:])

    
    # Prepare context for template
    context = {
        # Chart data
        'status_data': status_data,
        'status_labels': ['Pending', 'Investigation', 'Completed'],
        'status_counts': [status_data['pending'], status_data['investigation'], status_data['completed']],
        'status_colors': ['#FFC107', '#17A2B8', '#28A745'],
        'last_week_total' : last_week_total,
        'first_week_total' : first_week_total,
        'type_data': type_data,
        'type_labels': type_labels,
        'type_counts': type_counts,
        'type_colors': type_colors,
            'hour_labels': hour_labels,
    'hour_counts': hour_counts,
    'am_total': am_total,
    'pm_total': pm_total,
        'monthly_data': monthly_data,
        'monthly_labels': monthly_labels,
        'monthly_counts': monthly_counts,
        
        'daily_data': daily_data,
        'daily_labels': daily_labels,
        'daily_counts': daily_counts,
        
        'hour_data': hour_data,
        'hour_labels': hour_labels,
        'hour_counts': hour_counts,
        
        'status_trend_labels': monthly_labels,
        'status_trend_pending': status_trend_data['pending'],
        'status_trend_investigation': status_trend_data['investigation'],
        'status_trend_completed': status_trend_data['completed'],
        
        # Statistics
        'stats': stats,
        'user_stats': user_stats,
        'response_time_stats': response_time_stats,
        'top_submitters': top_submitters,
        
        # Period info
        'period': period,
        'start_date': start_date.date(),
        'end_date': end_date.date(),
        'active_page': 'feedback',
    }
    
    # If AJAX request for chart data, return JSON
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({
            'status_data': status_data,
            'type_data': type_data,
            'monthly_data': monthly_data,
            'daily_data': daily_data,
                'daily_counts': daily_counts,
    'first_week_total': first_week_total,
    'last_week_total': last_week_total,
            'hour_data': hour_data,
        })
    
    return render(request, 'admin_panel/feedback_analytics.html', context)

@staff_member_required
def get_chart_data(request):
    """API endpoint to get chart data (for AJAX updates)"""
    chart_type = request.GET.get('chart', 'status')
    period = request.GET.get('period', '30days')
    
    # Calculate date range
    end_date = timezone.now()
    if period == '7days':
        start_date = end_date - timedelta(days=7)
    elif period == '30days':
        start_date = end_date - timedelta(days=30)
    elif period == '90days':
        start_date = end_date - timedelta(days=90)
    elif period == '1year':
        start_date = end_date - timedelta(days=365)
    else:
        start_date = end_date - timedelta(days=30)
    
    feedbacks = Feedback.objects.filter(created_at__range=[start_date, end_date])
    
    if chart_type == 'status':
        data = {
            'pending': feedbacks.filter(status='pending').count(),
            'investigation': feedbacks.filter(status='investigation').count(),
            'completed': feedbacks.filter(status='completed').count(),
        }
        
    elif chart_type == 'type':
        type_data = {}
        for type_code, type_name in Feedback.FEEDBACK_TYPES:
            count = feedbacks.filter(feedback_type=type_code).count()
            type_data[type_name] = count
        data = type_data
        
    elif chart_type == 'monthly':
        monthly_data = {}
        for i in range(5, -1, -1):
            month_date = end_date - timedelta(days=30*i)
            month_key = month_date.strftime('%b %Y')
            month_start = month_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            next_month = (month_start + timedelta(days=32)).replace(day=1)
            
            count = Feedback.objects.filter(
                created_at__gte=month_start,
                created_at__lt=next_month
            ).count()
            monthly_data[month_key] = count
        data = monthly_data
        status_trend_data = {
    'pending': [],
    'investigation': [],
    'completed': []
}

        
    elif chart_type == 'hourly':
        hour_data = {}
        for hour in range(24):
            hour_str = f"{hour:02d}:00"
            count = feedbacks.filter(created_at__hour=hour).count()
            hour_data[hour_str] = count
        data = hour_data
        
    else:
        data = {}
    
    return JsonResponse(data)

@staff_member_required
def feedback_analytics_export(request):
    """Export analytics data"""
    format_type = request.GET.get('format', 'csv')
    period = request.GET.get('period', '30days')
    
    # Calculate date range
    end_date = timezone.now()
    if period == '7days':
        start_date = end_date - timedelta(days=7)
    elif period == '30days':
        start_date = end_date - timedelta(days=30)
    elif period == '90days':
        start_date = end_date - timedelta(days=90)
    elif period == '1year':
        start_date = end_date - timedelta(days=365)
    else:
        start_date = end_date - timedelta(days=30)
    
    feedbacks = Feedback.objects.filter(created_at__range=[start_date, end_date])
    
    if format_type == 'csv':
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="feedback_analytics_{period}_{timezone.now().date()}.csv"'
        
        writer = csv.writer(response)
        
        # Write header
        writer.writerow(['Analytics Report', f'Period: {period}', f'From: {start_date.date()}', f'To: {end_date.date()}'])
        writer.writerow([])
        
        # Statistics
        writer.writerow(['Overall Statistics'])
        writer.writerow(['Total Feedbacks', feedbacks.count()])
        writer.writerow(['Pending', feedbacks.filter(status='pending').count()])
        writer.writerow(['Under Investigation', feedbacks.filter(status='investigation').count()])
        writer.writerow(['Completed', feedbacks.filter(status='completed').count()])
        writer.writerow([])
        
        # Type distribution
        writer.writerow(['Feedback Type Distribution'])
        for type_code, type_name in Feedback.FEEDBACK_TYPES:
            count = feedbacks.filter(feedback_type=type_code).count()
            writer.writerow([type_name, count])
        writer.writerow([])
        
        # Monthly trend
        writer.writerow(['Monthly Trend (Last 6 months)'])
        for i in range(5, -1, -1):
            month_date = end_date - timedelta(days=30*i)
            month_key = month_date.strftime('%b %Y')
            month_start = month_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            next_month = (month_start + timedelta(days=32)).replace(day=1)
            
            count = Feedback.objects.filter(
                created_at__gte=month_start,
                created_at__lt=next_month
            ).count()
            writer.writerow([month_key, count])
            status_trend_data = {
    'pending': [],
    'investigation': [],
    'completed': []
}

        
        return response
    

    
    elif format_type == 'json':
        data = {
            'period': period,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'total_feedbacks': feedbacks.count(),
            'statistics': {
                'pending': feedbacks.filter(status='pending').count(),
                'investigation': feedbacks.filter(status='investigation').count(),
                'completed': feedbacks.filter(status='completed').count(),
            },
            'type_distribution': {
                dict(Feedback.FEEDBACK_TYPES).get(type_code, type_code): 
                feedbacks.filter(feedback_type=type_code).count()
                for type_code, _ in Feedback.FEEDBACK_TYPES
            }
        }
        
        response = JsonResponse(data)
        response['Content-Disposition'] = f'attachment; filename="feedback_analytics_{period}_{timezone.now().date()}.json"'
        return response
    

    
    return redirect('admin_feedback_analytics')

# 1. Mark as Pending
def admin_mark_pending(request, feedback_id):
    feedback = Feedback.objects.get(id=feedback_id)
    feedback.status = 'pending'
    feedback.save()
    messages.success(request, f'Feedback #{feedback_id} marked as Pending')
    return redirect('admin_feedback_management')

# 2. Mark as Investigation  
def admin_mark_investigation(request, feedback_id):
    feedback = Feedback.objects.get(id=feedback_id)
    feedback.status = 'investigation'
    feedback.save()
    messages.success(request, f'Feedback #{feedback_id} marked as Under Investigation')
    return redirect('admin_feedback_management')

# 3. Mark as Completed
def admin_mark_completed(request, feedback_id):
    feedback = Feedback.objects.get(id=feedback_id)
    feedback.status = 'completed'
    feedback.save()
    messages.success(request, f'Feedback #{feedback_id} marked as Completed')
    return redirect('admin_feedback_management')

# 4. Mark as Completed with Response
def admin_mark_completed_with_response(request, feedback_id):
    feedback = Feedback.objects.get(id=feedback_id)
    feedback.status = 'completed'
    feedback.admin_response = request.POST.get('admin_response', '')
    feedback.save()
    messages.success(request, f'Feedback #{feedback_id} marked as Completed with response')
    return redirect('admin_feedback_management')