# admin_panel/middleware.py
from django.shortcuts import redirect
from django.urls import reverse

class AdminAccessMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Check if path starts with /admin-panel/
        if request.path.startswith('/admin-panel/'):
            # Check if user is authenticated and is superuser
            if not request.user.is_authenticated:
                return redirect(f"{reverse('admin_login')}?next={request.path}")
            
            if not request.user.is_superuser:
                from django.contrib import messages
                from django.shortcuts import redirect
                
                messages.error(request, 'Access denied. Superuser privileges required.')
                return redirect('admin_login')
        
        response = self.get_response(request)
        return response