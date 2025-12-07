# """
# URL configuration for JobCompass project.

# The `urlpatterns` list routes URLs to views. For more information please see:
#     https://docs.djangoproject.com/en/5.2/topics/http/urls/
# Examples:
# Function views
#     1. Add an import:  from my_app import views
#     2. Add a URL to urlpatterns:  path('', views.home, name='home')
# Class-based views
#     1. Add an import:  from other_app.views import Home
#     2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
# Including another URLconf
#     1. Import the include() function: from django.urls import include, path
#     2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
# """

from django.contrib import admin
from django.urls import path
from App import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('resume/<uuid:resume_id>/', views.resume_detail, name='resume_detail'),
    path('job/<uuid:job_id>/', views.job_detail, name='job_detail'),
    path('nl_query/', views.nl_query, name='nl_query'),
    path('analyze/', views.analyze_job_fit, name='analyze_job_fit'),

    #testing
    # path('analyze/quick_resume/', views.analyze_and_show_resume, name='analyze_and_show_resume'),
    path('analyze/quick_job/', views.analyze_and_show_job, name='analyze_and_show_job'),
    path('analyze/job/json/', views.analyze_job_json, name='analyze_job_json'),
    path('analyze/job/json/<uuid:job_id>/', views.analyze_job_json, name='analyze_job_json_by_id'),


    #need changes
    path('match/<uuid:resume_id>/<uuid:job_id>/', views.compute_match, name='compute_match'),
    path('gaps/<int:match_id>/', views.gaps_view, name='gaps'),
    path('recommendation/<int:match_id>/', views.recommendation_view, name='recommendation'),

]

