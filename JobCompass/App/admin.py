from django.contrib import admin
from django.contrib.admin import ModelAdmin
from .models import Resume, JobDescription, MatchResult, Gap, Recommendation, NLSession

class ResumeAdmin(ModelAdmin):
    list_display = ('id', 'owner', 'uploaded_at', 'file')
    search_fields = ('owner__username',)
admin.site.register(Resume,ResumeAdmin)

class JobDescriptionAdmin(ModelAdmin):
    list_display = ('id', 'title', 'uploaded_at')
    search_fields = ('title',)
admin.site.register(JobDescription,JobDescriptionAdmin)

class MatchResultAdmin(ModelAdmin):
    list_display = ('id', 'resume', 'job', 'score', 'rating', 'computed_at')
    search_fields = ('resume__id', 'job__title')
admin.site.register(MatchResult,MatchResultAdmin)

class GapAdmin(ModelAdmin):
    list_display = ('id', 'match', 'skill', 'type', 'importance')
    search_fields = ('skill',)
admin.site.register(Gap,GapAdmin)

class RecommendationAdmin(ModelAdmin):
    list_display = ('id', 'match', 'kind', 'created_at')
    search_fields = ('kind',)
admin.site.register(Recommendation,RecommendationAdmin)

class NLSessionAdmin(ModelAdmin):
    list_display = ('id', 'resume', 'created_at')
    search_fields = ('resume__id',)
admin.site.register(NLSession,NLSessionAdmin)
