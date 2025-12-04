from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone
import uuid
import jsonfield

User = get_user_model()

class Skill(models.Model):
    name = models.CharField(max_length=120, unique=True)
    normalized = models.CharField(max_length=120, blank=True, help_text="Normalized form (e.g., React.js -> React)")
    def __str__(self):
        return self.name

class Resume(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    uploaded_at = models.DateTimeField(default=timezone.now)
    file = models.FileField(upload_to='resumes/')
    raw_text = models.TextField(blank=True)
    parsed = jsonfield.JSONField(blank=True, default=dict)  # store parsed structure
    def __str__(self):
        return f"Resume {self.id}"

class JobDescription(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    uploaded_at = models.DateTimeField(default=timezone.now)
    title = models.CharField(max_length=255, blank=True)
    file = models.FileField(upload_to='jobs/', null=True, blank=True)
    raw_text = models.TextField(blank=True)
    parsed = jsonfield.JSONField(blank=True, default=dict)
    def __str__(self):
        return self.title or str(self.id)

class MatchResult(models.Model):
    resume = models.ForeignKey(Resume, on_delete=models.CASCADE)
    job = models.ForeignKey(JobDescription, on_delete=models.CASCADE)
    score = models.FloatField()
    rating = models.CharField(max_length=20)
    breakdown = jsonfield.JSONField()
    computed_at = models.DateTimeField(default=timezone.now)

class Gap(models.Model):
    match = models.ForeignKey(MatchResult, on_delete=models.CASCADE, related_name='gaps')
    skill = models.CharField(max_length=255)
    type = models.CharField(max_length=50, choices=(('missing','missing'),('transferable','transferable'),('learnable','learnable')))
    importance = models.FloatField(default=0.0)
    suggestion = models.CharField(max_length=512, blank=True)

class Recommendation(models.Model):
    match = models.ForeignKey(MatchResult, on_delete=models.CASCADE, related_name='recommendations')
    kind = models.CharField(max_length=30)  # Type A/B/C/D
    text = models.TextField()
    created_at = models.DateTimeField(default=timezone.now)

class NLSession(models.Model):
    resume = models.ForeignKey(Resume, null=True, blank=True, on_delete=models.CASCADE)
    created_at = models.DateTimeField(default=timezone.now)
    # last 5 turns stored
    turns = jsonfield.JSONField(default=list)
