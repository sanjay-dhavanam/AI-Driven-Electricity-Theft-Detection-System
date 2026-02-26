"""
User Portal Views for Electricity Theft Detection System
"""

# Django core
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.csrf import csrf_protect
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.utils import timezone
from django.db.models import Count, Avg, Sum
from datetime import datetime, timedelta
import json
from django.db import models
# Core app models
from core.models import (
    Meter,
    UserProfile,
    HalfHourReading,
    DailyConsumption,
    AnomalyResult,
    AnomalyInjection,
    InjectionManager
)

# ML logic
from core.ml_logic import AnomalyPredictor

DEFAULT_PASSWORD = "electricity123"


@csrf_protect
def custom_login(request):
    """
    Login using Meter ID + fixed password (academic demo)
    """
    if request.method == "POST":
        meter_id = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")

        # Validate meter
        try:
            meter = Meter.objects.get(meter_id=meter_id)
        except Meter.DoesNotExist:
            messages.error(request, "Invalid Meter ID")
            return render(request, "users/login.html")

        # Validate password
        if password != DEFAULT_PASSWORD:
            messages.error(request, "Invalid password")
            return render(request, "users/login.html")

        # Create or get user
        user, created = User.objects.get_or_create(
            username=meter_id,
            defaults={"email": f"{meter_id}@example.com"}
        )

        if created:
            user.set_password(DEFAULT_PASSWORD)
            user.save()

        # Create or update profile
        UserProfile.objects.get_or_create(
            user=user,
            defaults={"meter": meter}
        )

        # Authenticate
        user = authenticate(request, username=meter_id, password=DEFAULT_PASSWORD)
        if user:
            login(request, user)

            # 🔁 Simulated 30-minute update
            run_simulation_for_user(user)

            return redirect("users:dashboard")

        messages.error(request, "Login failed")

    return render(request, "users/login.html")


def custom_logout(request):
    logout(request)
    return redirect("users:login")


from datetime import datetime, time
# ... other imports ...

def run_simulation_for_user(user):
    """
    Run simulation for a user on page load
    Creates new half-hour reading and runs anomaly detection
    Uses historical dates (2010-2014)
    """
    try:
        profile = user.profile
        if not profile.is_simulation_active:
            return None
            
        meter = profile.meter
        
        # Get next reading from simulation
        next_reading = profile.get_next_reading()
        if not next_reading:
            return None
        
        # Create timestamp for historical reading
        reading_date = next_reading['date']
        hour = next_reading['hour']
        minute = next_reading['minute']
        
        # Create datetime object for historical date
        naive_datetime = datetime.combine(reading_date, time(hour, minute))
        timestamp = timezone.make_aware(naive_datetime)
        
        # Check if reading already exists
        existing = HalfHourReading.objects.filter(
            meter=meter,
            timestamp=timestamp
        ).first()
        
        if existing:
            return existing
        
        # Create HalfHourReading
        half_hour_reading = HalfHourReading.objects.create(
            meter=meter,
            user_profile=profile,
            timestamp=timestamp,
            consumption=next_reading['consumption'],
            hour=hour,
            minute=minute,
            is_peak=(hour >= 7 and hour < 23),
            is_off_peak=(hour < 7 or hour >= 23),
        )
        
        # Check if we have a full day's worth of readings (48)
        today_readings = HalfHourReading.objects.filter(
            meter=meter,
            timestamp__date=reading_date
        ).count()
        
        # If we have 48 readings, run anomaly detection
        if today_readings >= 48:
            # Check if anomaly result already exists
            anomaly_exists = AnomalyResult.objects.filter(
                meter=meter,
                date=reading_date
            ).exists()
            
            if not anomaly_exists:
                # Run anomaly prediction
                anomaly_result = AnomalyPredictor.predict_for_day(
                    meter.meter_id, 
                    reading_date
                )
        
        return half_hour_reading
        
    except Exception as e:
        print(f"Error in simulation: {e}")
        return None


@login_required
def dashboard(request):
    """
    User dashboard - main landing page
    Uses historical data dates (2010-2014)
    """
    # Run simulation
    # run_simulation_for_user(request.user)
    
    profile = request.user.profile
    meter = profile.meter
    
    # Get dataset date range from DailyConsumption
    date_stats = DailyConsumption.objects.filter(
        meter=meter
    ).aggregate(
        min_date=models.Min('date'),
        max_date=models.Max('date')
    )
    
    dataset_min_date = date_stats['min_date']
    dataset_max_date = date_stats['max_date']
    
    if not dataset_min_date or not dataset_max_date:
        dataset_min_date = datetime(2010, 1, 1).date()
        dataset_max_date = datetime(2014, 12, 31).date()
    
    total_days_in_dataset = (dataset_max_date - dataset_min_date).days + 1
    
    # Get all anomalies with injections
    all_anomalies_with_injections = InjectionManager.get_anomalies_for_meter_with_injections(
        meter.meter_id,
        dataset_min_date,
        dataset_max_date
    )
    
    # Sort anomalies by date descending
    all_anomalies_with_injections.sort(key=lambda x: x['date'], reverse=True)
    
    # Get current simulation position
    current_pointer = profile.current_pointer
    consumptions = DailyConsumption.objects.filter(meter=meter).order_by('date')
    
    # Calculate current simulated day
    if consumptions.exists():
        total_half_hours = consumptions.count() * 48
        if current_pointer >= total_half_hours:
            current_pointer = total_half_hours - 1
            profile.current_pointer = current_pointer
            profile.save()
        
        day_index = current_pointer // 48
        current_simulated_half_hour = current_pointer % 48
        
        if day_index < len(consumptions):
            current_simulated_day = consumptions[day_index].date
            today_readings = HalfHourReading.objects.filter(
                meter=meter,
                timestamp__date=current_simulated_day
            )
            today_total = sum(r.consumption for r in today_readings) if today_readings.exists() else 0
            current_day_anomaly = InjectionManager.get_anomaly_with_injection(
                meter.meter_id,
                current_simulated_day
            )
        else:
            current_simulated_day = dataset_max_date
            current_simulated_half_hour = 47
            today_readings = HalfHourReading.objects.none()
            today_total = 0
            current_day_anomaly = None
    else:
        current_simulated_day = dataset_max_date
        current_simulated_half_hour = 0
        today_readings = HalfHourReading.objects.none()
        today_total = 0
        current_day_anomaly = None
    
    # Calculate progress
    total_half_hours = consumptions.count() * 48 if consumptions.exists() else 1
    progress_percentage = (current_pointer / total_half_hours * 100) if total_half_hours > 0 else 0
    
    # Get latest readings
    latest_readings = HalfHourReading.objects.filter(
        meter=meter
    ).order_by('-timestamp')[:10]
    
    # Calculate statistics
    last_day_prediction = all_anomalies_with_injections[0] if all_anomalies_with_injections else None
    
    # Last 30 days from dataset_max_date
    last_30_days_start = dataset_max_date - timedelta(days=30)
    last_30_anomalies = [
        a for a in all_anomalies_with_injections 
        if a['date'] >= last_30_days_start and a['date'] <= dataset_max_date
    ]
    
    last_30_normal = sum(1 for a in last_30_anomalies if a['classification'] == 'normal')
    last_30_suspicious = sum(1 for a in last_30_anomalies if a['classification'] == 'suspicious')
    last_30_theft = sum(1 for a in last_30_anomalies if a['classification'] == 'theft')
    last_30_total = len(last_30_anomalies)
    
    # Total statistics
    total_normal = sum(1 for a in all_anomalies_with_injections if a['classification'] == 'normal')
    total_suspicious = sum(1 for a in all_anomalies_with_injections if a['classification'] == 'suspicious')
    total_theft = sum(1 for a in all_anomalies_with_injections if a['classification'] == 'theft')
    total_predicted_days = len(all_anomalies_with_injections)
    
    dataset_coverage_percentage = (total_predicted_days / total_days_in_dataset * 100) if total_days_in_dataset > 0 else 0
    
    context = {
        'profile': profile,
        'meter': meter,
        'latest_readings': latest_readings,
        'dataset_min_date': dataset_min_date,
        'dataset_max_date': dataset_max_date,
        'total_days_in_dataset': total_days_in_dataset,
        'current_simulated_day': current_simulated_day,
        'current_simulated_half_hour': current_simulated_half_hour,
        'last_day_prediction': last_day_prediction,
        'last_30_stats': {
            'normal': last_30_normal,
            'suspicious': last_30_suspicious,
            'theft': last_30_theft,
            'total': last_30_total,
            'normal_percentage': (last_30_normal / last_30_total * 100) if last_30_total > 0 else 0,
            'suspicious_percentage': (last_30_suspicious / last_30_total * 100) if last_30_total > 0 else 0,
            'theft_percentage': (last_30_theft / last_30_total * 100) if last_30_total > 0 else 0,
        },
        'total_stats': {
            'normal': total_normal,
            'suspicious': total_suspicious,
            'theft': total_theft,
            'total_predicted': total_predicted_days,
            'dataset_coverage': dataset_coverage_percentage,
            'normal_percentage': (total_normal / total_predicted_days * 100) if total_predicted_days > 0 else 0,
            'suspicious_percentage': (total_suspicious / total_predicted_days * 100) if total_predicted_days > 0 else 0,
            'theft_percentage': (total_theft / total_predicted_days * 100) if total_predicted_days > 0 else 0,
        },
        'today_total': today_total,
        'today_readings_count': today_readings.count(),
        'today_half_hour_progress': f"{current_simulated_half_hour}/48",
        'current_day_anomaly': current_day_anomaly,
        'progress_percentage': progress_percentage,
    }
    
    return render(request, 'users/dashboard.html', context)


@login_required 
def daily_view(request, date_str=None):
    """
    Show daily consumption with 48 half-hour bars
    Uses historical dates
    """
    run_simulation_for_user(request.user)
    
    profile = request.user.profile
    meter = profile.meter
    
    # Get max date from consumption data
    max_date = DailyConsumption.objects.filter(meter=meter).aggregate(models.Max('date'))['date__max']
    if not max_date:
        max_date = datetime(2014, 12, 31).date()
    
    # Parse date or use current simulated day
    if date_str:
        try:
            selected_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except:
            # Use max date if invalid
            selected_date = max_date
    else:
        # Use max date
        selected_date = max_date
    
    # Get readings for selected date
    daily_readings = HalfHourReading.objects.filter(
        meter=meter,
        timestamp__date=selected_date
    ).order_by('timestamp')
    
    # Get anomaly with injection
    anomaly_result = InjectionManager.get_anomaly_with_injection(
        meter.meter_id,
        selected_date
    )
    
    # Get previous and next dates with data
    prev_date = HalfHourReading.objects.filter(
        meter=meter,
        timestamp__date__lt=selected_date
    ).order_by('-timestamp').values_list('timestamp__date', flat=True).first()
    
    next_date = HalfHourReading.objects.filter(
        meter=meter,
        timestamp__date__gt=selected_date
    ).order_by('timestamp').values_list('timestamp__date', flat=True).first()
    
    # Prepare chart data
    chart_data = {
        'labels': [],
        'consumptions': [],
        'colors': [],
        'times': [],
    }
    
    for reading in daily_readings:
        time_label = reading.get_time_label()
        chart_data['labels'].append(time_label)
        chart_data['consumptions'].append(reading.consumption)
        chart_data['times'].append(time_label)
        
        # Color coding
        if anomaly_result:
            color_map = {
                'normal': '#4CAF50',
                'suspicious': '#FF9800',
                'theft': '#F44336',
            }
            color = color_map.get(anomaly_result['classification'], '#9E9E9E')
        else:
            color = '#9E9E9E'
        
        chart_data['colors'].append(color)
    
    context = {
        'profile': profile,
        'meter': meter,
        'selected_date': selected_date,
        'daily_readings': daily_readings,
        'anomaly_result': anomaly_result,
        'prev_date': prev_date,
        'next_date': next_date,
        'chart_data_json': json.dumps(chart_data),
        'total_consumption': sum(chart_data['consumptions']) if chart_data['consumptions'] else 0,
        'max_consumption': max(chart_data['consumptions']) if chart_data['consumptions'] else 0,
        'avg_consumption': (sum(chart_data['consumptions']) / len(chart_data['consumptions'])) if chart_data['consumptions'] else 0,
    }
    
    return render(request, 'users/daily_view.html', context)


@login_required
def anomaly_timeline(request):
    """
    Show historical anomaly results with color coding
    """
    # Run simulation
    # run_simulation_for_user(request.user)
    
    profile = request.user.profile
    meter = profile.meter
    
    # Get all anomaly results
    anomaly_results = AnomalyResult.objects.filter(
        meter=meter
    ).order_by('-date')
    
    # Get date range for filtering
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    
    if start_date:
        try:
            anomaly_results = anomaly_results.filter(date__gte=start_date)
        except:
            pass
    
    if end_date:
        try:
            anomaly_results = anomaly_results.filter(date__lte=end_date)
        except:
            pass
    
    # Prepare timeline data with injection consideration
    timeline_data = []
    current_month = None
    month_data = []
    
    for result in anomaly_results:
        # Check for injection
        injection = InjectionManager.get_injection_for_result(
            meter.meter_id,
            result.date
        )
        
        if injection:
            classification = injection.injection_type
            score = injection.injected_score
            confidence = injection.injected_confidence
            is_injected = True
            reason = injection.reason
        else:
            classification = result.classification
            score = result.anomaly_score
            confidence = result.confidence
            is_injected = result.is_injected
            reason = result.injection_reason
        
        result_month = result.date.strftime('%Y-%m')
        
        if result_month != current_month:
            if month_data:
                timeline_data.append({
                    'month': current_month,
                    'results': month_data.copy()
                })
            month_data = []
            current_month = result_month
        
        month_data.append({
            'date': result.date,
            'classification': classification,
            'score': score,
            'confidence': confidence,
            'is_injected': is_injected,
            'reason': reason,
            'color': get_color_code_for_classification(classification),
            'icon': get_icon_for_classification(classification),
        })
    
    # Add last month
    if month_data:
        timeline_data.append({
            'month': current_month,
            'results': month_data
        })
    
    # Statistics with injection consideration
    normal_count = 0
    suspicious_count = 0
    theft_count = 0
    
    for result in anomaly_results:
        injection = InjectionManager.get_injection_for_result(
            meter.meter_id,
            result.date
        )
        
        if injection:
            classification = injection.injection_type
        else:
            classification = result.classification
        
        if classification == 'normal':
            normal_count += 1
        elif classification == 'suspicious':
            suspicious_count += 1
        elif classification == 'theft':
            theft_count += 1
    
    total_results = normal_count + suspicious_count + theft_count
    
    context = {
        'profile': profile,
        'meter': meter,
        'timeline_data': timeline_data,
        'total_results': total_results,
        'normal_results': normal_count,
        'suspicious_results': suspicious_count,
        'theft_results': theft_count,
        'normal_percentage': (normal_count / total_results * 100) if total_results > 0 else 0,
        'suspicious_percentage': (suspicious_count / total_results * 100) if total_results > 0 else 0,
        'theft_percentage': (theft_count / total_results * 100) if total_results > 0 else 0,
    }
    
    return render(request, 'users/anomaly_timeline.html', context)


@login_required
def settings(request):
    """
    User settings page
    """
    profile = request.user.profile
    
    if request.method == 'POST':
        # Update simulation speed
        simulation_speed = request.POST.get('simulation_speed', '1')
        try:
            profile.simulation_speed = int(simulation_speed)
        except:
            profile.simulation_speed = 1
        
        # Update chart theme
        chart_theme = request.POST.get('chart_theme', 'light')
        profile.chart_theme = chart_theme
        
        # Update simulation status
        is_active = request.POST.get('is_simulation_active', 'off')
        profile.is_simulation_active = (is_active == 'on')
        
        # Reset simulation pointer if requested
        if 'reset_simulation' in request.POST:
            profile.current_pointer = 0
        
        profile.save()
        messages.success(request, "Settings updated successfully")
        return redirect('users:settings')
    
    context = {
        'profile': profile,
        'simulation_speeds': [
            {'value': 1, 'label': '1 half-hour per page load'},
            {'value': 2, 'label': '2 half-hours per page load'},
            {'value': 4, 'label': '4 half-hours per page load'},
            {'value': 8, 'label': '8 half-hours per page load'},
            {'value': 48, 'label': '1 day per page load'},
        ],
        'chart_themes': [
            {'value': 'light', 'label': 'Light Theme'},
            {'value': 'dark', 'label': 'Dark Theme'},
            {'value': 'colorblind', 'label': 'Colorblind Friendly'},
        ],
        'current_date': timezone.now().date(),
    }
    
    return render(request, 'users/settings.html', context)


@login_required
def api_latest_reading(request):
    """
    API endpoint for latest reading (AJAX)
    """
    profile = request.user.profile
    meter = profile.meter
    
    latest_reading = HalfHourReading.objects.filter(
        meter=meter
    ).order_by('-timestamp').first()
    
    # Get latest anomaly with injection
    latest_anomaly_result = AnomalyResult.objects.filter(
        meter=meter
    ).order_by('-date').first()
    
    latest_anomaly = None
    if latest_anomaly_result:
        injection = InjectionManager.get_injection_for_result(
            meter.meter_id,
            latest_anomaly_result.date
        )
        
        if injection:
            latest_anomaly = {
                'classification': injection.injection_type,
                'score': injection.injected_score,
                'confidence': injection.injected_confidence,
                'color': get_color_code_for_classification(injection.injection_type),
                'icon': get_icon_for_classification(injection.injection_type),
            }
        else:
            latest_anomaly = {
                'classification': latest_anomaly_result.classification,
                'score': latest_anomaly_result.anomaly_score,
                'confidence': latest_anomaly_result.confidence,
                'color': latest_anomaly_result.get_color_code(),
                'icon': latest_anomaly_result.get_icon(),
            }
    
    data = {
        'success': True,
        'meter_id': meter.meter_id,
        'latest_reading': {
            'timestamp': latest_reading.timestamp.isoformat() if latest_reading else None,
            'consumption': latest_reading.consumption if latest_reading else 0,
            'time': latest_reading.get_time_label() if latest_reading else 'N/A',
        } if latest_reading else None,
        'latest_anomaly': latest_anomaly,
        'simulation_progress': profile.current_pointer,
        'simulation_speed': profile.simulation_speed,
    }
    
    return JsonResponse(data)


@login_required
def api_daily_data(request, date_str):
    """
    API endpoint for daily data (AJAX)
    """
    profile = request.user.profile
    meter = profile.meter
    
    try:
        selected_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except:
        selected_date = timezone.now().date()
    
    daily_readings = HalfHourReading.objects.filter(
        meter=meter,
        timestamp__date=selected_date
    ).order_by('timestamp')
    
    anomaly_result = AnomalyResult.objects.filter(
        meter=meter,
        date=selected_date
    ).first()
    
    display_anomaly = None
    if anomaly_result:
        injection = InjectionManager.get_injection_for_result(
            meter.meter_id,
            selected_date
        )
        
        if injection:
            display_anomaly = {
                'classification': injection.injection_type,
                'score': injection.injected_score,
                'confidence': injection.injected_confidence,
                'is_injected': True,
                'color': get_color_code_for_classification(injection.injection_type),
            }
        else:
            display_anomaly = {
                'classification': anomaly_result.classification,
                'score': anomaly_result.anomaly_score,
                'confidence': anomaly_result.confidence,
                'is_injected': anomaly_result.is_injected,
                'color': anomaly_result.get_color_code(),
            }
    
    chart_data = {
        'labels': [],
        'consumptions': [],
        'times': [],
    }
    
    for reading in daily_readings:
        chart_data['labels'].append(reading.get_time_label())
        chart_data['consumptions'].append(reading.consumption)
        chart_data['times'].append(reading.get_time_label())
    
    data = {
        'success': True,
        'date': selected_date.isoformat(),
        'chart_data': chart_data,
        'anomaly_result': display_anomaly,
        'total_consumption': sum(chart_data['consumptions']) if chart_data['consumptions'] else 0,
        'readings_count': len(chart_data['consumptions']),
    }
    
    return JsonResponse(data)


# Helper functions
def get_icon_for_classification(classification):
    """Return icon for classification"""
    icons = {
        'normal': '🟢',
        'suspicious': '🟡',
        'theft': '🔴',
    }
    return icons.get(classification, '⚪')


def get_color_code_for_classification(classification):
    """Return color code for classification"""
    colors = {
        'normal': 'success',
        'suspicious': 'warning',
        'theft': 'danger',
    }
    return colors.get(classification, 'secondary')


from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q, Count
from django.utils import timezone
from .forms import FeedbackForm
from core.models import Feedback, Meter
import json

@login_required
def feedback_view(request):
    """User view to submit feedback"""
    
    # Get user's meters for suggestions
    user_meters = Meter.objects.filter(feedbacks__user=request.user).distinct()
    
    if request.method == 'POST':
        form = FeedbackForm(request.POST)
        if form.is_valid():
            feedback = form.save(commit=False)
            feedback.user = request.user
            
            # Try to find associated meter
            try:
                meter = Meter.objects.get(meter_id=form.cleaned_data['id_number'])
                feedback.meter = meter
            except Meter.DoesNotExist:
                feedback.meter = None
            
            feedback.save()
            
            # Add success message
            messages.success(
                request, 
                'Thank you for your feedback! We have received your submission.',
                extra_tags='success'
            )
            
            # Send notification to admin (optional)
            # You can add email or notification system here
            
            return redirect('users:feedback_history')
        else:
            messages.error(
                request,
                'Please correct the errors below.',
                extra_tags='danger'
            )
    else:
        form = FeedbackForm()
    
    # Get user's recent feedbacks
    recent_feedbacks = Feedback.objects.filter(
        user=request.user
    ).order_by('-created_at')[:5]
    
    # Get statistics for user
    user_feedback_stats = {
        'total': Feedback.objects.filter(user=request.user).count(),
        'pending': Feedback.objects.filter(user=request.user, status='pending').count(),
        'investigation': Feedback.objects.filter(user=request.user, status='investigation').count(),
        'completed': Feedback.objects.filter(user=request.user, status='completed').count(),
    }
    
    return render(request, 'users/feedback.html', {
        'form': form,
        'recent_feedbacks': recent_feedbacks,
        'user_feedback_stats': user_feedback_stats,
        'user_meters': user_meters,
        'active_tab': 'feedback'
    })

@login_required
def feedback_history(request):
    """User view to see feedback history"""
    
    # Get all feedbacks for current user
    feedbacks_list = Feedback.objects.filter(
        user=request.user
    ).select_related('meter').order_by('-created_at')
    
    # Apply filters if any
    status_filter = request.GET.get('status', '')
    search_query = request.GET.get('search', '')
    
    if status_filter:
        feedbacks_list = feedbacks_list.filter(status=status_filter)
    
    if search_query:
        feedbacks_list = feedbacks_list.filter(
            Q(id_number__icontains=search_query) |
            Q(feedback_text__icontains=search_query) |
            Q(admin_response__icontains=search_query)
        )
    
    # Pagination
    paginator = Paginator(feedbacks_list, 10)
    page_number = request.GET.get('page')
    feedbacks = paginator.get_page(page_number)
    
    # Statistics
    stats = {
        'total': feedbacks_list.count(),
        'pending': feedbacks_list.filter(status='pending').count(),
        'investigation': feedbacks_list.filter(status='investigation').count(),
        'completed': feedbacks_list.filter(status='completed').count(),
    }
    
    # Calculate response rate
    if stats['total'] > 0:
        stats['response_rate'] = round(
            (feedbacks_list.filter(admin_response__isnull=False).count() / stats['total']) * 100, 
            1
        )
    else:
        stats['response_rate'] = 0
    
    # Get feedback type distribution for user
    type_distribution = {}
    for type_code, type_name in Feedback.FEEDBACK_TYPES:
        count = feedbacks_list.filter(feedback_type=type_code).count()
        if count > 0:
            type_distribution[type_name] = count
    
    return render(request, 'users/feedback_history.html', {
        'feedbacks': feedbacks,
        'stats': stats,
        'type_distribution': type_distribution,
        'status_filter': status_filter,
        'search_query': search_query,
        'active_tab': 'feedback',
        'paginator': paginator,
        'status_choices': dict(Feedback.STATUS_CHOICES),
        'feedback_types': dict(Feedback.FEEDBACK_TYPES),
    })

@login_required
def feedback_detail(request, feedback_id):
    """User view to see specific feedback details"""
    
    feedback = get_object_or_404(
        Feedback.objects.select_related('meter', 'user'),
        id=feedback_id, 
        user=request.user
    )
    
    # Get related feedbacks
    related_feedbacks = Feedback.objects.filter(
        user=request.user,
        id_number=feedback.id_number
    ).exclude(id=feedback.id)[:3]
    
    # Get user's feedback stats
    user_stats = {
        'total_feedbacks': Feedback.objects.filter(user=request.user).count(),
        'avg_response_time': calculate_avg_response_time(request.user),
    }
    
    return render(request, 'users/feedback_detail.html', {
        'feedback': feedback,
        'related_feedbacks': related_feedbacks,
        'user_stats': user_stats,
        'active_tab': 'feedback',
    })

@login_required
def feedback_export(request, format='json'):
    """Export user's feedbacks"""
    feedbacks = Feedback.objects.filter(user=request.user).order_by('-created_at')
    
    if format == 'json':
        import json
        from django.http import JsonResponse
        
        data = []
        for fb in feedbacks:
            data.append({
                'id': fb.id,
                'id_number': fb.id_number,
                'type': fb.get_feedback_type_display(),
                'status': fb.get_status_display(),
                'feedback': fb.feedback_text,
                'admin_response': fb.admin_response,
                'created_at': fb.created_at.isoformat(),
                'updated_at': fb.updated_at.isoformat(),
                'days_open': fb.get_days_since_submission()
            })
        
        return JsonResponse({'feedbacks': data}, safe=False)
    
    elif format == 'csv':
        import csv
        from django.http import HttpResponse
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="my_feedbacks_{timezone.now().date()}.csv"'
        
        writer = csv.writer(response)
        writer.writerow(['ID', 'ID/Meter Number', 'Type', 'Status', 'Feedback', 'Admin Response', 
                        'Created Date', 'Updated Date', 'Days Open'])
        
        for fb in feedbacks:
            writer.writerow([
                fb.id,
                fb.id_number,
                fb.get_feedback_type_display(),
                fb.get_status_display(),
                fb.feedback_text[:500],  # Truncate for CSV
                fb.admin_response[:500] if fb.admin_response else '',
                fb.created_at.date(),
                fb.updated_at.date(),
                fb.get_days_since_submission()
            ])
        
        return response
    
    return redirect('users:feedback_history')

def calculate_avg_response_time(user):
    """Calculate average response time for user's feedbacks"""
    completed_feedbacks = Feedback.objects.filter(
        user=user,
        status='completed',
        admin_response__isnull=False
    )
    
    if not completed_feedbacks:
        return 0
    
    total_days = 0
    for fb in completed_feedbacks:
        if fb.admin_response:
            response_date = fb.updated_at
            days = (response_date - fb.created_at).days
            total_days += days
    
    return round(total_days / len(completed_feedbacks), 1)