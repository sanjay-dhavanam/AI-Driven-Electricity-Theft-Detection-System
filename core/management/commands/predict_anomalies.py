"""
Management command to predict anomalies
Usage: python manage.py predict_anomalies --meter MAC000002 --date 2023-01-01
"""

import os
from django.core.management.base import BaseCommand
from django.utils.dateparse import parse_date
from core.ml_logic import AnomalyPredictor, predict_all_meters_latest

class Command(BaseCommand):
    help = 'Predict anomalies for electricity consumption data'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--meter',
            type=str,
            help='Meter ID to predict (if not specified, process all meters)'
        )
        parser.add_argument(
            '--date',
            type=str,
            help='Specific date to predict (YYYY-MM-DD format)'
        )
        parser.add_argument(
            '--days',
            type=int,
            default=7,
            help='Number of recent days to predict (default: 7)'
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Process all meters with models'
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=50,
            help='Limit number of meters to process (default: 50)'
        )
    
    def handle(self, *args, **options):
        if options['meter']:
            # Predict for specific meter
            self.predict_single_meter(
                options['meter'],
                options['date'],
                options['days']
            )
        elif options['all']:
            # Predict for all meters
            self.predict_all_meters(options['limit'])
        else:
            # Show usage
            self.stdout.write(self.style.WARNING(
                'Please specify --meter METER_ID or --all to process all meters'
            ))
    
    def predict_single_meter(self, meter_id, date_str, days):
        """Predict anomalies for a single meter"""
        self.stdout.write(self.style.SUCCESS(f'Predicting anomalies for meter: {meter_id}'))
        
        if date_str:
            # Predict for specific date
            date = parse_date(date_str)
            if not date:
                self.stdout.write(self.style.ERROR(f'Invalid date format: {date_str}'))
                return
            
            result = AnomalyPredictor.predict_for_day(meter_id, date)
            if result:
                self.print_result(result)
            else:
                self.stdout.write(self.style.WARNING(f'No prediction generated'))
        else:
            # Predict for recent days
            results = AnomalyPredictor.batch_predict_for_meter(
                meter_id,
                limit=days
            )
            
            self.stdout.write(f'Generated {len(results)} predictions:')
            for result in results:
                self.print_result(result, indent=True)
    
    def predict_all_meters(self, limit):
        """Predict anomalies for all meters with models"""
        self.stdout.write(self.style.SUCCESS(
            f'Predicting latest anomalies for up to {limit} meters...'
        ))
        
        results = predict_all_meters_latest(limit)
        
        self.stdout.write(f'\nGenerated {len(results)} predictions:')
        
        # Group by classification
        from collections import Counter
        counter = Counter([r.classification for r in results])
        
        self.stdout.write(self.style.SUCCESS('\nSummary:'))
        for classification, count in counter.items():
            self.stdout.write(f'  {classification.title()}: {count}')
    
    def print_result(self, result, indent=False):
        """Print prediction result"""
        prefix = '  ' if indent else ''
        
        self.stdout.write(
            f"{prefix}{result.date}: {result.classification.upper()} "
            f"(score: {result.anomaly_score:.4f}, "
            f"confidence: {result.confidence:.2f})"
        )