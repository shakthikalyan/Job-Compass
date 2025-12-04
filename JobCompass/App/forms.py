from django import forms
from .models import Resume, JobDescription

class ResumeUploadForm(forms.ModelForm):
    class Meta:
        model = Resume
        fields = ['file']

class JobUploadForm(forms.ModelForm):
    class Meta:
        model = JobDescription
        fields = ['title', 'file', 'raw_text']
        widgets = {
            'raw_text': forms.Textarea(attrs={'rows':6, 'placeholder':'Paste job description text here (optional)'})
        }

class NLQueryForm(forms.Form):
    query = forms.CharField(widget=forms.Textarea(attrs={'rows':3}), label="Ask a question about this resume / job")
    resume_id = forms.CharField(widget=forms.HiddenInput(), required=False)
    job_id = forms.CharField(widget=forms.HiddenInput(), required=False)
