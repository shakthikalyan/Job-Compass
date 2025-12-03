from django.contrib import admin
from django.contrib.admin import ModelAdmin
from .models import Resume, JobDescription


# Register your models here.
class ResumeAdmin(ModelAdmin):
    list_display = ('uploaded_at', 'file')  # Display upload time and file name in admin list view
    search_fields = ('uploaded_at',)  # Allow searching by upload time
admin.site.register(Resume, ResumeAdmin)

class JobDescriptionAdmin(ModelAdmin):
    list_display = ('title', 'created_at')  # Display title and creation time in admin list view
    search_fields = ('title',)  # Allow searching by job title
admin.site.register(JobDescription, JobDescriptionAdmin)