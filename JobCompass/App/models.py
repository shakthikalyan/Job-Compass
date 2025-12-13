# App/models.py
from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone
import uuid
import jsonfield

User = get_user_model()


class Domain(models.Model):
    name = models.CharField(max_length=255, unique=True, db_index=True)
    description = models.TextField(blank=True, null=True)
    source = models.CharField(max_length=50, blank=True, default="onet")

    def __str__(self):
        return self.name


class Skill(models.Model):
    """
    Canonical skill/knowledge/ability token.
    'element_type' stores whether this is a Skill / Knowledge / Ability per O*NET.
    """
    name = models.CharField(max_length=512, unique=True)
    normalized = models.CharField(max_length=512, blank=True, db_index=True)
    element_type = models.CharField(max_length=64, blank=True, db_index=True)  # Skill, Knowledge, Ability
    external_id = models.CharField(max_length=256, blank=True, null=True, db_index=True)  # O*NET Element ID
    synonyms = jsonfield.JSONField(blank=True, default=list)
    embedding = models.BinaryField(blank=True, null=True)
    source = models.CharField(max_length=50, blank=True, default="onet")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['normalized']),
            models.Index(fields=['element_type']),
        ]

    def __str__(self):
        return self.name


class Occupation(models.Model):
    """
    Job role / occupation. external_id holds O*NET-SOC Code.
    alt_titles can hold related / alternate occupation names.
    """
    name = models.CharField(max_length=512, unique=True, db_index=True)
    external_id = models.CharField(max_length=256, blank=True, null=True, db_index=True)  # O*NET-SOC Code
    domain = models.ForeignKey(Domain, null=True, blank=True, on_delete=models.SET_NULL, related_name='occupations')
    description = models.TextField(blank=True, null=True)
    alt_titles = jsonfield.JSONField(blank=True, default=list)
    source = models.CharField(max_length=50, blank=True, default="onet")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['name']),
            models.Index(fields=['external_id']),
        ]

    def __str__(self):
        return self.name


class Technology(models.Model):
    """
    Tools/technologies (AWS, SAP, Figma...). Seeded from Technology Skills file.
    """
    name = models.CharField(max_length=256, unique=True)
    normalized = models.CharField(max_length=256, blank=True, db_index=True)
    external_id = models.CharField(max_length=256, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class OccupationSkillRelation(models.Model):
    """
    Through table mapping Occupation -> Skill (or Knowledge/Ability).
    Stores O*NET importance (Data Value) and level if provided.
    Unique on (occupation, skill) so we aggregate multiple rows into one authoritative relation.
    """
    RELATION_CHOICES = (
        ('essential', 'essential'),
        ('important', 'important'),
        ('optional', 'optional'),
        ('other', 'other'),
    )

    occupation = models.ForeignKey(Occupation, on_delete=models.CASCADE, related_name='skill_relations')
    skill = models.ForeignKey(Skill, on_delete=models.CASCADE, related_name='occupation_relations')
    element_type = models.CharField(max_length=64, blank=True)  # mirror skill.element_type for quick filtering
    relation_type = models.CharField(max_length=32, choices=RELATION_CHOICES, default='other')
    importance = models.FloatField(default=0.0, db_index=True)  # O*NET Data Value (0-100)
    level = models.FloatField(default=0.0, db_index=True)  # if O*NET provides level
    provenance = models.CharField(max_length=50, default='onet')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('occupation', 'skill')
        indexes = [
            models.Index(fields=['occupation', 'skill']),
            models.Index(fields=['relation_type']),
            models.Index(fields=['importance']),
        ]

    def __str__(self):
        return f"{self.occupation} ← {self.skill} ({self.relation_type})"


class OccupationTechnology(models.Model):
    occupation = models.ForeignKey(Occupation, on_delete=models.CASCADE, related_name='technology_relations')
    technology = models.ForeignKey(Technology, on_delete=models.CASCADE, related_name='occupation_relations')
    importance = models.FloatField(default=0.0)
    provenance = models.CharField(max_length=50, default='onet')

    class Meta:
        unique_together = ('occupation', 'technology')


class Resume(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    uploaded_at = models.DateTimeField(default=timezone.now)
    file = models.FileField(upload_to='resumes/', null=True, blank=True)
    raw_text = models.TextField(blank=True)
    parsed = jsonfield.JSONField(blank=True, default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Resume {self.id}"


class JobDescription(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    uploaded_at = models.DateTimeField(default=timezone.now)
    title = models.CharField(max_length=255, blank=True)
    occupation = models.ForeignKey(Occupation, null=True, blank=True, on_delete=models.SET_NULL)
    file = models.FileField(upload_to='jobs/', null=True, blank=True)
    raw_text = models.TextField(blank=True)
    parsed = jsonfield.JSONField(blank=True, default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title or str(self.id)


class MatchResult(models.Model):
    resume = models.ForeignKey(Resume, on_delete=models.CASCADE)
    job = models.ForeignKey(JobDescription, on_delete=models.CASCADE)
    score = models.FloatField(db_index=True)
    rating = models.CharField(max_length=20)
    breakdown = jsonfield.JSONField()  # skill match details, tech match, weighted totals
    computed_at = models.DateTimeField(default=timezone.now)


class Gap(models.Model):
    match = models.ForeignKey(MatchResult, on_delete=models.CASCADE, related_name='gaps')
    skill = models.CharField(max_length=255)
    type = models.CharField(max_length=50, choices=(('missing','missing'),('transferable','transferable'),('learnable','learnable')))
    importance = models.FloatField(default=0.0)
    suggestion = models.CharField(max_length=1024, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class Recommendation(models.Model):
    match = models.ForeignKey(MatchResult, on_delete=models.CASCADE, related_name='recommendations')
    kind = models.CharField(max_length=30)  # course, project, certification
    text = models.TextField()
    metadata = jsonfield.JSONField(blank=True, default=dict)
    created_at = models.DateTimeField(default=timezone.now)


class NLSession(models.Model):
    resume = models.ForeignKey(Resume, null=True, blank=True, on_delete=models.CASCADE)
    created_at = models.DateTimeField(default=timezone.now)
    turns = jsonfield.JSONField(default=list)

    def __str__(self):
        return f"NLSession {self.id}"
