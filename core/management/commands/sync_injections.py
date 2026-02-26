from django.core.management.base import BaseCommand
from core.models import Meter, AnomalyResult, AnomalyInjection
from django.utils import timezone

class Command(BaseCommand):
    help = 'Sync injections with anomaly results (create synthetic results for injection-only dates)'
    
    def handle(self, *args, **options):
        # Get all active injections
        injections = AnomalyInjection.objects.filter(is_active=True)
        
        created_count = 0
        updated_count = 0
        
        for injection in injections:
            try:
                # Check if anomaly result already exists for this date
                anomaly_result = AnomalyResult.objects.filter(
                    meter=injection.meter,
                    date=injection.date
                ).first()
                
                if anomaly_result:
                    # Update existing anomaly result with injection
                    if not anomaly_result.is_injected:
                        anomaly_result.classification = injection.injection_type
                        anomaly_result.anomaly_score = injection.injected_score
                        anomaly_result.confidence = injection.injected_confidence
                        anomaly_result.is_injected = True
                        anomaly_result.injection_reason = injection.reason
                        anomaly_result.save()
                        updated_count += 1
                        self.stdout.write(f"Updated: {injection.meter.meter_id} - {injection.date}")
                else:
                    # Create synthetic anomaly result
                    AnomalyResult.objects.create(
                        meter=injection.meter,
                        date=injection.date,
                        classification=injection.injection_type,
                        anomaly_score=injection.injected_score,
                        confidence=injection.injected_confidence,
                        is_injected=True,
                        injection_reason=injection.reason,
                        threshold_normal=0.0,
                        threshold_suspicious=-0.1,
                        threshold_theft=-0.2,
                        model_name='Injection',
                        model_version='1.0',
                        predicted_at=timezone.now()
                    )
                    created_count += 1
                    self.stdout.write(f"Created: {injection.meter.meter_id} - {injection.date}")
                    
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error processing injection {injection.id}: {e}"))
        
        self.stdout.write(self.style.SUCCESS(
            f"\nSync complete! Created: {created_count}, Updated: {updated_count}"
        ))