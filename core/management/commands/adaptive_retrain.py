"""
Management command for adaptive retraining
Usage: python manage.py adaptive_retrain --job "Monthly Retraining" --meters MAC000002,MAC000048
"""

import os
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from core.adaptive_learning import RetrainingCommands, AdaptiveLearningManager

class Command(BaseCommand):
    help = 'Run adaptive retraining for electricity theft detection models'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--job',
            type=str,
            default='Scheduled Retraining',
            help='Name of the retraining job'
        )
        parser.add_argument(
            '--meters',
            type=str,
            help='Comma-separated list of meter IDs to retrain (or "all" for all meters)'
        )
        parser.add_argument(
            '--user',
            type=int,
            default=1,
            help='User ID to associate with retraining job (default: 1)'
        )
        parser.add_argument(
            '--batch',
            type=int,
            default=5,
            help='Batch size for processing meters (default: 5)'
        )
        parser.add_argument(
            '--list-needing',
            action='store_true',
            help='List meters that need retraining based on criteria'
        )
        parser.add_argument(
            '--health',
            action='store_true',
            help='Show system health metrics'
        )
    
    def handle(self, *args, **options):
        if options['list_needing']:
            self.list_meters_needing_retraining()
        elif options['health']:
            self.show_system_health()
        else:
            self.run_retraining(options)
    
    def list_meters_needing_retraining(self):
        """List meters that might benefit from retraining"""
        candidates = AdaptiveLearningManager.get_meters_needing_retraining()
        
        self.stdout.write(self.style.SUCCESS(f"\nMeters needing retraining: {len(candidates)}"))
        self.stdout.write("=" * 80)
        
        for candidate in candidates[:20]:  # Show first 20
            self.stdout.write(
                f"\n{candidate['meter_id']} (Score: {candidate['score']}):"
            )
            for reason in candidate['reasons']:
                self.stdout.write(f"  • {reason}")
        
        if len(candidates) > 20:
            self.stdout.write(f"\n... and {len(candidates) - 20} more meters")
    
    def show_system_health(self):
        """Show adaptive learning system health"""
        health = AdaptiveLearningManager.get_system_health()
        
        self.stdout.write(self.style.SUCCESS("\nADAPTIVE LEARNING SYSTEM HEALTH"))
        self.stdout.write("=" * 80)
        
        self.stdout.write(f"\nModel Coverage:")
        self.stdout.write(f"  Total meters: {health['total_meters']}")
        self.stdout.write(f"  Meters with models: {health['meters_with_models']}")
        self.stdout.write(f"  Coverage: {health['coverage_percentage']:.1f}%")
        
        self.stdout.write(f"\nRetraining Activity (last 30 days):")
        self.stdout.write(f"  Total retraining jobs: {health['recent_retrains']}")
        self.stdout.write(f"  Successful jobs: {health['successful_retrains']}")
        self.stdout.write(f"  Success rate: {health['success_rate']:.1f}%")
        
        self.stdout.write(f"\nModel Age:")
        self.stdout.write(f"  Average model age: {health['average_model_age_days']:.1f} days")
        self.stdout.write(f"  Oldest model: {health['oldest_model_age']} days")
        
        self.stdout.write(f"\nMeters needing retraining: {health['meters_needing_retraining']}")
    
    def run_retraining(self, options):
        """Run retraining job"""
        job_name = options['job']
        user_id = options['user']
        batch_size = options['batch']
        
        # Get meter IDs
        if options['meters']:
            if options['meters'].lower() == 'all':
                meter_ids = None  # Will process all meters
            else:
                meter_ids = [m.strip() for m in options['meters'].split(',')]
        else:
            # If no meters specified, process those needing retraining
            candidates = AdaptiveLearningManager.get_meters_needing_retraining()
            meter_ids = [c['meter_id'] for c in candidates]
        
        if meter_ids is not None:
            self.stdout.write(self.style.SUCCESS(
                f"\nStarting retraining job: '{job_name}'"
            ))
            self.stdout.write(f"Meters to process: {len(meter_ids)}")
            if len(meter_ids) <= 20:
                self.stdout.write(f"Meter IDs: {', '.join(meter_ids)}")
            else:
                self.stdout.write(f"Meter IDs: {', '.join(meter_ids[:20])}...")
        else:
            self.stdout.write(self.style.SUCCESS(
                f"\nStarting retraining job: '{job_name}' for ALL meters"
            ))
        
        # Run retraining
        result = RetrainingCommands.run_retraining_job(
            job_name=job_name,
            meter_ids=meter_ids,
            user_id=user_id,
            batch_size=batch_size,
        )
        
        if result:
            self.stdout.write(self.style.SUCCESS("\nRetraining Job Complete!"))
            self.stdout.write("=" * 80)
            
            # Show summary
            self.stdout.write(f"\nSummary:")
            self.stdout.write(f"  Job ID: {result.id}")
            self.stdout.write(f"  Status: {result.get_status_display()}")
            self.stdout.write(f"  Duration: {result.get_duration_minutes():.1f} minutes")
            self.stdout.write(f"  Success rate: {result.get_success_rate():.1f}%")
            self.stdout.write(f"  Improvement rate: {result.get_improvement_rate():.1f}%")
            
            # Show sample results
            if result.detailed_logs:
                self.stdout.write(f"\nSample results:")
                successful_logs = [log for log in result.detailed_logs 
                                 if log.get('status') == 'success']
                
                for log in successful_logs[:5]:  # Show first 5
                    meter_id = log.get('meter_id', 'Unknown')
                    decision = log.get('decision', 'unknown')
                    improvement = log.get('improvement_percentage', 0)
                    
                    if decision == 'replaced':
                        self.stdout.write(self.style.SUCCESS(
                            f"  {meter_id}: REPLACED (+{improvement:.1f}%)"
                        ))
                    else:
                        self.stdout.write(self.style.WARNING(
                            f"  {meter_id}: KEPT ({improvement:.1f}%)"
                        ))
                
                if len(successful_logs) > 5:
                    self.stdout.write(f"  ... and {len(successful_logs) - 5} more meters")
            
            self.stdout.write(f"\nView details in admin: http://localhost:8000/admin/core/retraininglog/{result.id}/")