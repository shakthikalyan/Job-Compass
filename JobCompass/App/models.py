from django.db import models

# Create your models here.

class Resume(models.Model):
    file = models.FileField(upload_to='resumes/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    parsed_text = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"Resume uploaded at {self.uploaded_at}"  # Display upload time


class JobDescription(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title  # Display job title
