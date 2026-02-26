"""
One-time setup script to initialize predictions for all meters
Run: python manage.py shell < setup_predictions.py
"""

from core.models import Meter, DailyConsumption, AnomalyResult
from core.ml_logic import AnomalyPredictor

def setup_all_predictions():
    """Generate predictions for all meters and all days"""
    print("Starting prediction setup for all meters...")
    
    meters = Meter.objects.filter(is_active=True)
    total_meters = meters.count()
    
    for i, meter in enumerate(meters, 1):
        print(f"\n[{i}/{total_meters}] Processing {meter.meter_id}")
        
        # Get consumption data for this meter
        consumptions = DailyConsumption.objects.filter(meter=meter)
        
        if not consumptions.exists():
            print(f"  No consumption data for {meter.meter_id}")
            continue
        
        total_days = consumptions.count()
        predicted = 0
        
        for consumption in consumptions:
            # Check if prediction already exists
            if not AnomalyResult.objects.filter(
                meter=meter,
                date=consumption.date
            ).exists():
                result = AnomalyPredictor.predict_for_day(meter.meter_id, consumption.date)
                if result:
                    predicted += 1
            
            if predicted % 10 == 0:
                print(f"  Generated {predicted}/{total_days} predictions...")
        
        print(f"  Completed: {predicted} predictions generated")
    
    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)

# Run the setup
if __name__ == "__main__":
    setup_all_predictions()