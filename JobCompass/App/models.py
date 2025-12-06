# App/models.py
from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone
import uuid
import jsonfield

User = get_user_model()

class Skill(models.Model):
    # Increased max_length to allow longer labels from ESCO
    name = models.CharField(max_length=512, unique=True)
    normalized = models.CharField(max_length=512, blank=True, db_index=True)
    esco_id = models.CharField(max_length=256, blank=True, null=True, db_index=True)
    synonyms = jsonfield.JSONField(blank=True, default=list)
    embedding = models.BinaryField(blank=True, null=True)  # optional: store bytes
    source = models.CharField(max_length=50, blank=True, default="local")

    def __str__(self):
        return self.name

class Occupation(models.Model):
    name = models.CharField(max_length=512, unique=True)
    esco_id = models.CharField(max_length=256, blank=True, null=True, db_index=True)
    description = models.TextField(blank=True, null=True)
    source = models.CharField(max_length=50, blank=True, default="local")

    def __str__(self):
        return self.name

class OccupationSkillRelation(models.Model):
    RELATION_CHOICES = (
        ('essential', 'essential'),
        ('optional', 'optional'),
        ('other', 'other'),
    )

    occupation = models.ForeignKey(Occupation, on_delete=models.CASCADE, related_name='skill_relations')
    skill = models.ForeignKey(Skill, on_delete=models.CASCADE, related_name='occupation_relations')
    relation_type = models.CharField(max_length=32, choices=RELATION_CHOICES, default='other')

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('occupation', 'skill', 'relation_type')
        indexes = [
            models.Index(fields=['occupation']),
            models.Index(fields=['skill']),
            models.Index(fields=['relation_type']),
        ]

    def __str__(self):
        return f"{self.occupation} - {self.skill} ({self.relation_type})"

class Resume(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    uploaded_at = models.DateTimeField(default=timezone.now)
    file = models.FileField(upload_to='resumes/', null=True, blank=True)
    raw_text = models.TextField(blank=True)
    parsed = jsonfield.JSONField(blank=True, default=dict)

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
    kind = models.CharField(max_length=30)
    text = models.TextField()
    created_at = models.DateTimeField(default=timezone.now)

class NLSession(models.Model):
    resume = models.ForeignKey(Resume, null=True, blank=True, on_delete=models.CASCADE)
    created_at = models.DateTimeField(default=timezone.now)
    turns = jsonfield.JSONField(default=list)
