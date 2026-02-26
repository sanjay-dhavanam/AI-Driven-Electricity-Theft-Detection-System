from django.contrib import admin
from core.models import Feedback

@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'meter', 'id_number', 'feedback_type', 'status', 'created_at']
    list_filter = ['status', 'feedback_type', 'created_at']
    search_fields = ['id_number', 'feedback_text', 'user__username', 'user__email']
    list_editable = ['status']
    readonly_fields = ['created_at', 'updated_at']
    actions = ['mark_as_pending', 'mark_as_investigation', 'mark_as_completed']
    
    fieldsets = (
        ('Feedback Information', {
            'fields': ('user', 'meter', 'id_number', 'feedback_type', 'feedback_text')
        }),
        ('Status Management', {
            'fields': ('status', 'admin_response', 'is_resolved')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def mark_as_pending(self, request, queryset):
        updated = queryset.update(status='pending')
        self.message_user(request, f'{updated} feedback(s) marked as pending.')
    mark_as_pending.short_description = "Mark selected as pending"
    
    def mark_as_investigation(self, request, queryset):
        updated = queryset.update(status='investigation')
        self.message_user(request, f'{updated} feedback(s) marked as under investigation.')
    mark_as_investigation.short_description = "Mark selected as under investigation"
    
    def mark_as_completed(self, request, queryset):
        updated = queryset.update(status='completed')
        self.message_user(request, f'{updated} feedback(s) marked as completed.')
    mark_as_completed.short_description = "Mark selected as completed"