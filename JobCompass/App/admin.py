from django.contrib import admin
from django.contrib.admin import ModelAdmin
from App.models import Resume, JobDescription, MatchResult, Gap, Recommendation, NLSession
from App.models import Skill, Occupation, OccupationSkillRelation

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


@admin.register(Skill)
class SkillAdmin(admin.ModelAdmin):
    list_display = ("name", "esco_id", "source")
    search_fields = ("name", "esco_id", "synonyms")
    list_filter = ("source",)
    ordering = ("name",)
    readonly_fields = ("normalized", "esco_id")

    fieldsets = (
        ("Basic Info", {
            "fields": ("name", "normalized", "esco_id", "source")
        }),
        ("Synonyms", {
            "fields": ("synonyms",),
        }),
        ("Embedding (optional)", {
            "fields": ("embedding",),
            "classes": ("collapse",),
        })
    )


@admin.register(Occupation)
class OccupationAdmin(admin.ModelAdmin):
    list_display = ("name", "esco_id", "source")
    search_fields = ("name", "esco_id")
    list_filter = ("source",)
    ordering = ("name",)

    fieldsets = (
        ("Basic Info", {
            "fields": ("name", "esco_id", "source")
        }),
        ("Description", {
            "fields": ("description",),
        }),
    )


@admin.register(OccupationSkillRelation)
class OccupationSkillRelationAdmin(admin.ModelAdmin):
    list_display = ("occupation", "skill", "relation_type")
    list_filter = ("relation_type", "occupation")
    search_fields = (
        "occupation__name",
        "skill__name",
        "occupation__esco_id",
        "skill__esco_id",
    )
    autocomplete_fields = ("occupation", "skill")
    ordering = ("occupation", "skill")