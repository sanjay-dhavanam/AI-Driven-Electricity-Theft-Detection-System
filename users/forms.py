from django import forms
from core.models import Feedback

class FeedbackForm(forms.ModelForm):
    class Meta:
        model = Feedback
        fields = ['id_number', 'feedback_text', 'feedback_type']
        widgets = {
            'id_number': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter your ID or Meter Number',
                'required': True,
                'id': 'feedback-id-number'
            }),
            'feedback_text': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 6,
                'placeholder': 'Please provide detailed feedback to help us improve our service...',
                'required': True,
                'id': 'feedback-text',
                'style': 'resize: vertical;'
            }),
            'feedback_type': forms.Select(attrs={
                'class': 'form-select',
                'required': True,
                'id': 'feedback-type'
            })
        }
        labels = {
            'id_number': 'ID/Meter Number',
            'feedback_text': 'Feedback Details',
            'feedback_type': 'Type of Feedback'
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add help text
        self.fields['id_number'].help_text = 'Enter your unique ID or meter number for reference'
        self.fields['feedback_text'].help_text = 'Be specific and provide as much detail as possible'
        self.fields['feedback_type'].help_text = 'Select the category that best describes your feedback'