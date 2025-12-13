# App/admin.py
from django.contrib import admin
from django.contrib.admin import ModelAdmin
from App.models import (
    Resume, JobDescription, MatchResult, Gap, Recommendation, NLSession,
    Skill, Occupation, OccupationSkillRelation, Technology, Domain, OccupationTechnology
)


class ResumeAdmin(ModelAdmin):
    list_display = ('id', 'owner', 'uploaded_at', 'file')
    search_fields = ('owner__username',)
admin.site.register(Resume, ResumeAdmin)


class JobDescriptionAdmin(ModelAdmin):
    list_display = ('id', 'title', 'uploaded_at')
    search_fields = ('title',)
admin.site.register(JobDescription, JobDescriptionAdmin)


class MatchResultAdmin(ModelAdmin):
    list_display = ('id', 'resume', 'job', 'score', 'rating', 'computed_at')
    search_fields = ('resume__id', 'job__title')
admin.site.register(MatchResult, MatchResultAdmin)


class GapAdmin(ModelAdmin):
    list_display = ('id', 'match', 'skill', 'type', 'importance')
    search_fields = ('skill',)
admin.site.register(Gap, GapAdmin)


class RecommendationAdmin(ModelAdmin):
    list_display = ('id', 'match', 'kind', 'created_at')
    search_fields = ('kind',)
admin.site.register(Recommendation, RecommendationAdmin)


class NLSessionAdmin(ModelAdmin):
    list_display = ('id', 'resume', 'created_at')
    search_fields = ('resume__id',)
admin.site.register(NLSession, NLSessionAdmin)


@admin.register(Skill)
class SkillAdmin(admin.ModelAdmin):
    list_display = ("name", "external_id", "source", "element_type")
    search_fields = ("name", "external_id", "synonyms")
    list_filter = ("source", "element_type")
    ordering = ("name",)
    readonly_fields = ("normalized", "external_id")

    fieldsets = (
        ("Basic Info", {
            "fields": ("name", "normalized", "external_id", "element_type", "source")
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
    list_display = ("name", "external_id", "source", "domain")
    search_fields = ("name", "external_id")
    list_filter = ("source", "domain")
    ordering = ("name",)

    fieldsets = (
        ("Basic Info", {
            "fields": ("name", "external_id", "domain", "source")
        }),
        ("Description & Alts", {
            "fields": ("description", "alt_titles"),
        }),
    )


@admin.register(OccupationSkillRelation)
class OccupationSkillRelationAdmin(admin.ModelAdmin):
    list_display = ("occupation", "skill", "relation_type", "importance")
    list_filter = ("relation_type", "occupation")
    search_fields = (
        "occupation__name",
        "skill__name",
        "occupation__external_id",
        "skill__external_id",
    )
    autocomplete_fields = ("occupation", "skill")
    ordering = ("occupation", "skill")


@admin.register(Technology)
class TechnologyAdmin(admin.ModelAdmin):
    list_display = ("name", "normalized")
    search_fields = ("name",)
    ordering = ("name",)


@admin.register(Domain)
class DomainAdmin(admin.ModelAdmin):
    list_display = ("name",)
    search_fields = ("name",)
