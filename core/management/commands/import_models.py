"""
Command to import existing trained models into ModelVersion system
"""
import os
import pickle
import shutil
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.conf import settings
from core.models import Meter, ModelVersion
from tqdm import tqdm

class Command(BaseCommand):
    help = 'Import existing trained models into ModelVersion system'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--source',
            type=str,
            default='all',
            help='Source directory: trained_models, models_batch, models_sample, or all'
        )
        parser.add_argument(
            '--activate',
            action='store_true',
            help='Activate imported models'
        )
    
    def handle(self, *args, **options):
        # Define source directories
        sources = []
        if options['source'] == 'all':
            sources = [
                'trained_models',
                'models_batch',
                'models_sample'
            ]
        else:
            sources = [options['source']]
        
        imported_count = 0
        skipped_count = 0
        error_count = 0
        
        for source_dir in sources:
            if not os.path.exists(source_dir):
                self.stdout.write(self.style.WARNING(f"Directory not found: {source_dir}"))
                continue
            
            self.stdout.write(self.style.SUCCESS(f"\nScanning {source_dir}..."))
            
            # Find all .pkl files
            pkl_files = [f for f in os.listdir(source_dir) if f.endswith('.pkl')]
            
            for filename in tqdm(pkl_files, desc=f"Importing from {source_dir}"):
                if not filename.startswith('model_'):
                    continue
                
                try:
                    # Extract meter ID
                    meter_id = filename.replace('model_', '').replace('.pkl', '')
                    
                    # Check if meter exists in database
                    try:
                        meter = Meter.objects.get(meter_id=meter_id)
                    except Meter.DoesNotExist:
                        self.stdout.write(f"Meter {meter_id} not found in database, skipping")
                        skipped_count += 1
                        continue
                    
                    # Check if model already imported
                    existing = ModelVersion.objects.filter(meter=meter).count()
                    if existing > 0:
                        # Skip if already has models
                        continue
                    
                    # Load model data to get metadata
                    model_path = os.path.join(source_dir, filename)
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    # Create media directories
                    media_models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
                    media_scalers_dir = os.path.join(settings.MEDIA_ROOT, 'scalers')
                    os.makedirs(media_models_dir, exist_ok=True)
                    os.makedirs(media_scalers_dir, exist_ok=True)
                    
                    # Copy model file to media directory
                    new_model_path = os.path.join(media_models_dir, filename)
                    shutil.copy2(model_path, new_model_path)
                    
                    # Extract metadata
                    threshold_n = model_data.get('threshold_n', 0.0)
                    threshold_s = model_data.get('threshold_s', -0.1)
                    training_samples = model_data.get('training_samples', 30)
                    
                    # Create ModelVersion
                    model_version = ModelVersion.objects.create(
                        meter=meter,
                        version_number=1,
                        model_name='IsolationForest',
                        model_file=f'models/{filename}',
                        window_size=30,
                        contamination=0.1,
                        threshold_normal=threshold_n,
                        threshold_theft=threshold_s,
                        training_samples=training_samples,
                        is_trained=True,
                        trained_on=timezone.now().date(),
                        training_logs={
                            'source': source_dir,
                            'original_file': filename,
                            'imported_at': timezone.now().isoformat(),
                        }
                    )
                    
                    # Activate if requested
                    if options['activate']:
                        model_version.activate()
                    
                    imported_count += 1
                    
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"Error importing {filename}: {str(e)}"))
                    error_count += 1
        
        # Print summary
        self.stdout.write(self.style.SUCCESS("\n" + "="*60))
        self.stdout.write(self.style.SUCCESS("IMPORT SUMMARY"))
        self.stdout.write(self.style.SUCCESS("="*60))
        self.stdout.write(f"Models imported: {imported_count}")
        self.stdout.write(f"Meters skipped: {skipped_count}")
        self.stdout.write(f"Errors: {error_count}")
        self.stdout.write(f"Total ModelVersion records: {ModelVersion.objects.count()}")
        self.stdout.write(f"Active models: {ModelVersion.objects.filter(is_active=True).count()}")