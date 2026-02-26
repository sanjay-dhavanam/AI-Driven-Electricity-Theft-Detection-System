from django.core.management.base import BaseCommand
from core.models import Meter, DailyConsumption
from django.utils import timezone
from datetime import timedelta
import random

class Command(BaseCommand):
    def handle(self, *args, **kwargs):

        for i in range(1, 132):
            meter, _ = Meter.objects.get_or_create(meter_id=f"MTR{i:03}")

            for d in range(30):
                DailyConsumption.objects.get_or_create(
                    meter=meter,
                    date=timezone.now().date() - timedelta(days=d),
                    defaults={"units": random.uniform(5, 20)}
                )

        self.stdout.write("Demo data created")