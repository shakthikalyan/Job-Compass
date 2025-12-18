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

# urlpatterns = [
#     path('admin/', admin.site.urls),
#     path('', views.index, name='index'),
#     path('resume/<uuid:resume_id>/', views.resume_detail, name='resume_detail'),
#     path('job/<uuid:job_id>/', views.job_detail, name='job_detail'),
#     path('nl_query/', views.nl_query, name='nl_query'),

    
#     path('analyze/', views.analyze_job_fit, name='analyze_job_fit'),
#     path('analyze/loading/', views.analyze_loading, name='analyze_loading'),
#     path('analyze/start/', views.analyze_start_background, name='analyze_start_background'),
#     path('analyze/status/', views.analyze_status, name='analyze_status'),
#     path('analyze/result-redirect/', views.analyze_result_redirect, name='analyze_result_redirect'),
#     path('match/<int:match_id>/', views.show_match_result, name='show_match_result'),
#     path('analyze/process/', views.analyze_process, name='analyze_process'),


    
#     path('analyze/quick_resume/', views.analyze_and_show_resume, name='analyze_and_show_resume'),
#     path('analyze/job/json/', views.analyze_job_json, name='analyze_job_json'),
#     path('analyze/job/json/<uuid:job_id>/', views.analyze_job_json, name='analyze_job_json_by_id'),
# ]


urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),

    # Dashboard
    path('', views.index, name='index'),

    # Resume & Job views
    path('resume/<uuid:resume_id>/', views.resume_detail, name='resume_detail'),
    path('job/<uuid:job_id>/', views.job_detail, name='job_detail'),

    # Analysis flow
    path('analyze/', views.analyze_job_fit, name='analyze_job_fit'),
    path('analyze/loading/', views.analyze_loading, name='analyze_loading'),
    path('analyze/start/', views.analyze_start_background, name='analyze_start_background'),
    path('analyze/status/', views.analyze_status, name='analyze_status'),
    path('analyze/result/', views.analyze_result_redirect, name='analyze_result_redirect'),

    # Match result
    path('match/<int:match_id>/', views.show_match_result, name='show_match_result'),

    # NL Chatbot
    path('nl_query/', views.nl_query, name='nl_query'),
]