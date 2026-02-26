"""
Management command to import CSV block files into the database.
Usage: python manage.py import_data --path /path/to/block_files
"""

import os
import pandas as pd
from django.core.management.base import BaseCommand
from django.db import transaction
from tqdm import tqdm
import time
from datetime import datetime
from core.models import Meter, DailyConsumption, ImportLog

class Command(BaseCommand):
    help = 'Import electricity consumption data from CSV block files'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--path',
            type=str,
            help='Path to directory containing block CSV files'
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=0,
            help='Limit number of files to process (0 for all)'
        )
        parser.add_argument(
            '--chunksize',
            type=int,
            default=10000,
            help='Chunk size for reading large CSV files'
        )
    
    def handle(self, *args, **options):
        start_time = time.time()
        
        # Get path to block files
        if options['path']:
            block_dir = options['path']
        else:
            from django.conf import settings
            block_dir = settings.BLOCK_FILES_DIR
        
        if not os.path.exists(block_dir):
            self.stdout.write(self.style.ERROR(f'Directory not found: {block_dir}'))
            return
        
        # Get list of CSV files
        csv_files = [f for f in os.listdir(block_dir) if f.endswith('.csv')]
        
        if options['limit'] > 0:
            csv_files = csv_files[:options['limit']]
        
        if not csv_files:
            self.stdout.write(self.style.WARNING('No CSV files found'))
            return
        
        # Create import log
        import_log = ImportLog.objects.create(
            file_name=f"Batch import: {len(csv_files)} files",
            status='in_progress'
        )
        
        total_meters_created = 0
        total_records_created = 0
        errors = []
        
        self.stdout.write(self.style.SUCCESS(f'Found {len(csv_files)} CSV files to process'))
        self.stdout.write(f'Reading from: {block_dir}')
        
        try:
            # Process each CSV file
            for csv_file in tqdm(csv_files, desc="Processing files"):
                file_path = os.path.join(block_dir, csv_file)
                
                try:
                    # Process file in chunks to handle large files
                    for chunk in pd.read_csv(file_path, chunksize=options['chunksize']):
                        self.process_chunk(chunk, import_log)
                        
                        # Update totals
                        total_meters_created = Meter.objects.count()
                        total_records_created = DailyConsumption.objects.count()
                        
                except Exception as e:
                    error_msg = f"Error processing {csv_file}: {str(e)}"
                    self.stdout.write(self.style.ERROR(error_msg))
                    errors.append(error_msg)
            
            # Update import log
            duration = time.time() - start_time
            import_log.status = 'completed' if not errors else 'partial'
            import_log.meters_imported = total_meters_created
            import_log.records_imported = total_records_created
            import_log.errors = errors
            import_log.completed_at = datetime.now()
            import_log.duration_seconds = duration
            import_log.save()
            
            # Print summary
            self.stdout.write(self.style.SUCCESS('\n' + '='*50))
            self.stdout.write(self.style.SUCCESS('IMPORT COMPLETE'))
            self.stdout.write(self.style.SUCCESS('='*50))
            self.stdout.write(f'Total meters in database: {total_meters_created}')
            self.stdout.write(f'Total consumption records: {total_records_created}')
            self.stdout.write(f'Time taken: {duration:.2f} seconds')
            self.stdout.write(f'Files processed: {len(csv_files)}')
            
            if errors:
                self.stdout.write(self.style.WARNING(f'\nErrors occurred: {len(errors)}'))
                for error in errors[:5]:  # Show first 5 errors
                    self.stdout.write(self.style.WARNING(f'  - {error}'))
                if len(errors) > 5:
                    self.stdout.write(self.style.WARNING(f'  ... and {len(errors)-5} more'))
            
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING('\nImport interrupted by user'))
            import_log.status = 'failed'
            import_log.errors = errors + ['Import interrupted by user']
            import_log.save()
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'\nFatal error: {str(e)}'))
            import_log.status = 'failed'
            import_log.errors = errors + [f'Fatal error: {str(e)}']
            import_log.save()
    
    def process_chunk(self, chunk, import_log):
        """Process a chunk of data from CSV"""
        with transaction.atomic():
            for _, row in chunk.iterrows():
                try:
                    # Extract meter ID
                    meter_id = row.get('LCLid')
                    if not meter_id or pd.isna(meter_id):
                        continue
                    
                    # Create or get meter
                    meter, created = Meter.objects.get_or_create(
                        meter_id=meter_id,
                        defaults={
                            'location': f'Meter {meter_id}',
                            'acorn_group': row.get('Acorn', ''),
                            'acorn_grouped': row.get('Acorn_grouped', ''),
                            'stdorToU': row.get('stdorToU', ''),
                            'file': row.get('file', ''),
                        }
                    )
                    
                    # Extract date
                    date_str = row.get('day')
                    if not date_str or pd.isna(date_str):
                        continue
                    
                    date = pd.to_datetime(date_str).date()
                    
                    # Extract half-hour readings (hh_0 to hh_47)
                    consumption_data = {}
                    for i in range(48):
                        col_name = f'hh_{i}'
                        if col_name in row and not pd.isna(row[col_name]):
                            consumption_data[col_name] = float(row[col_name])
                    ā
                    if not consumption_data:
                        continue
                    
                    # Create or update daily consumption
                    DailyConsumption.objects.update_or_create(
                        meter=meter,
                        date=date,
                        defaults={
                            'consumption_data': consumption_data,
                        }
                    )
                    
                except Exception as e:
                    import_log.errors.append(f"Error processing row: {str(e)}")