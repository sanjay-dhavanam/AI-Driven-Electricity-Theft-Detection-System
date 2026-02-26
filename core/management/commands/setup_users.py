"""
Create one Django user per Meter (demo: first 100 meters)
Username = Meter ID
Password = electricity123
"""

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from core.models import Meter, UserProfile

class Command(BaseCommand):
    help = "Setup users for meters with default password"

    def handle(self, *args, **options):
        DEFAULT_PASSWORD = "electricity123"

        meters = Meter.objects.all()[:100]  # limit for college demo

        self.stdout.write(f"Creating users for {meters.count()} meters...\n")

        created = 0
        existing = 0

        for meter in meters:
            user, is_created = User.objects.get_or_create(
                username=meter.meter_id,
                defaults={
                    "email": f"{meter.meter_id}@example.com"
                }
            )

            if is_created:
                user.set_password(DEFAULT_PASSWORD)
                user.save()
                created += 1
            else:
                existing += 1

            UserProfile.objects.get_or_create(
                user=user,
                defaults={"meter": meter}
            )

        self.stdout.write(self.style.SUCCESS(
            f"""
User setup completed successfully!

Created users   : {created}
Existing users  : {existing}
Total users     : {created + existing}

LOGIN DETAILS:
Username : Meter ID (e.g. MAC000002)
Password : electricity123

Login URL:
http://localhost:8000/users/login/
"""
        ))
