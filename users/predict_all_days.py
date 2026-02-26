"""
Management command to predict anomalies for all days in dataset
Usage: python manage.py predict_all_days --meter MAC000002 (or --all)
"""

import os
from django.core.management.base import BaseCommand
from django.db import models
from core.models import Meter, DailyConsumption, AnomalyResult
from core.ml_logic import AnomalyPredictor

class Command(BaseCommand):
    help = 'Predict anomalies for all days in dataset for all meters'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--meter',
            type=str,
            help='Specific meter ID to predict (or use --all for all meters)'
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Process all meters with consumption data'
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=0,
            help='Limit number of meters to process (0 for all)'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force re-prediction even if result already exists'
        )
    
    def handle(self, *args, **options):
        if options['meter']:
            # Process specific meter
            self.process_meter(options['meter'], options['force'])
        elif options['all']:
            # Process all meters
            self.process_all_meters(options['limit'], options['force'])
        else:
            self.stdout.write(self.style.WARNING(
                'Please specify --meter METER_ID or --all to process all meters'
            ))
    
    def process_meter(self, meter_id, force=False):
        """Process a single meter"""
        try:
            meter = Meter.objects.get(meter_id=meter_id)
        except Meter.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'Meter {meter_id} not found'))
            return
        
        self.stdout.write(self.style.SUCCESS(f'Processing meter: {meter_id}'))
        
        # Get all consumption days for this meter
        consumptions = DailyConsumption.objects.filter(meter=meter).order_by('date')
        
        if not consumptions.exists():
            self.stdout.write(self.style.WARNING(f'No consumption data for {meter_id}'))
            return
        
        total_days = consumptions.count()
        predicted_count = 0
        skipped_count = 0
        
        for consumption in consumptions:
            # Check if prediction already exists
            if not force and AnomalyResult.objects.filter(
                meter=meter,
                date=consumption.date
            ).exists():
                skipped_count += 1
                continue
            
            # Generate prediction
            result = AnomalyPredictor.predict_for_day(meter_id, consumption.date)
            if result:
                predicted_count += 1
            
            # Progress update
            if predicted_count % 10 == 0:
                self.stdout.write(f'  Processed {predicted_count}/{total_days} days...')
        
        self.stdout.write(self.style.SUCCESS(
            f'Completed: {predicted_count} predictions generated, {skipped_count} skipped'
        ))
    
    def process_all_meters(self, limit=0, force=False):
        """Process all meters with consumption data"""
        # Get all meters with consumption data
        meters_with_data = Meter.objects.filter(
            consumption__isnull=False
        ).distinct().order_by('meter_id')
        
        if limit > 0:
            meters_with_data = meters_with_data[:limit]
        
        total_meters = meters_with_data.count()
        
        self.stdout.write(self.style.SUCCESS(
            f'Processing {total_meters} meters with consumption data...'
        ))
        
        for i, meter in enumerate(meters_with_data, 1):
            self.stdout.write(f'\n[{i}/{total_meters}] Processing meter: {meter.meter_id}')
            
            # Get consumption days
            consumptions = DailyConsumption.objects.filter(meter=meter).order_by('date')
            total_days = consumptions.count()
            
            predicted_count = 0
            skipped_count = 0
            
            for consumption in consumptions:
                # Check if prediction already exists
                if not force and AnomalyResult.objects.filter(
                    meter=meter,
                    date=consumption.date
                ).exists():
                    skipped_count += 1
                    continue
                
                # Generate prediction
                result = AnomalyPredictor.predict_for_day(meter.meter_id, consumption.date)
                if result:
                    predicted_count += 1
            
            self.stdout.write(f'  {meter.meter_id}: {predicted_count} predictions, {skipped_count} skipped')
        
        self.stdout.write(self.style.SUCCESS('\nAll predictions completed!'))