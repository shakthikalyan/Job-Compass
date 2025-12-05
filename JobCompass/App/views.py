from django.shortcuts import render, redirect, get_object_or_404
from .forms import ResumeUploadForm, JobUploadForm, NLQueryForm
from .models import Resume, JobDescription, MatchResult, Gap, Recommendation, NLSession
from . import utils
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Avg

# def index(request):
#     resumes = Resume.objects.all().order_by('-uploaded_at')[:10]
#     jobs = JobDescription.objects.all().order_by('-uploaded_at')[:10]
#     return render(request, "base.html", {"resumes": resumes, "jobs": jobs})

def index(request):
    resumes = Resume.objects.all().order_by('-uploaded_at')[:10]
    jobs = JobDescription.objects.all().order_by('-uploaded_at')[:10]

    # Compute safe stats in Python
    total_resumes = Resume.objects.count()
    total_jobs = JobDescription.objects.count()
    avg_match = MatchResult.objects.aggregate(avg=Avg('score'))['avg'] or 0.0

    stats = {
        "total_resumes": total_resumes,
        "total_jobs": total_jobs,
        "avg_match_score": round(avg_match, 2)
    }

    # Example skill_counts & recent_matches (fill as you like)
    skill_counts = []  # build from parsed resumes if desired
    recent_matches = MatchResult.objects.order_by('-computed_at')[:5]

    return render(request, "index.html", {
        "resumes": resumes,
        "jobs": jobs,
        "stats": stats,
        "skill_counts": skill_counts,
        "recent_matches": recent_matches
    })

def upload_resume(request):
    if request.method == "POST":
        form = ResumeUploadForm(request.POST, request.FILES)
        if form.is_valid():
            r = form.save()
            utils.parse_and_store_resume(r)
            return redirect(f'/resume/{r.id}/')  # project-level URLs
    else:
        form = ResumeUploadForm()
    return render(request, "upload_resume.html", {"form": form})

def resume_detail(request, resume_id):
    r = get_object_or_404(Resume, pk=resume_id)
    return render(request, "resume_detail.html", {"resume": r})

def upload_job(request):
    if request.method == "POST":
        form = JobUploadForm(request.POST, request.FILES)
        if form.is_valid():
            j = form.save()
            utils.parse_and_store_job(j)
            return redirect(f'/job/{j.id}/')
    else:
        form = JobUploadForm()
    return render(request, "upload_job.html", {"form": form})

def job_detail(request, job_id):
    j = get_object_or_404(JobDescription, pk=job_id)
    return render(request, "job_detail.html", {"job": j})

def compute_match(request, resume_id, job_id):
    resume = get_object_or_404(Resume, pk=resume_id)
    job = get_object_or_404(JobDescription, pk=job_id)
    mm = utils.semantic_match_score(resume.parsed or {}, job.parsed or {})
    mr = MatchResult.objects.create(resume=resume, job=job, score=mm['score'], rating=mm['rating'], breakdown=mm['breakdown'])
    return render(request, "match_result.html", {"match": mr, "breakdown": mm['breakdown']})

def gaps_view(request, match_id):
    match = get_object_or_404(MatchResult, pk=match_id)
    gaps = utils.gap_analysis(match.resume.parsed or {}, match.job.parsed or {}, top_n=5)
    match.gaps.all().delete()
    gap_objs = []
    for g in gaps:
        gap_objs.append(Gap.objects.create(match=match, skill=g['skill'], type=g['type'], importance=g['importance'], suggestion=g['suggestion']))
    return render(request, "gaps.html", {"match": match, "gaps": gap_objs})

def recommendation_view(request, match_id):
    match = get_object_or_404(MatchResult, pk=match_id)
    gaps = [{"skill": g.skill, "type": g.type, "importance": g.importance, "suggestion": g.suggestion} for g in match.gaps.all()]
    rec = utils.generate_personalized_recommendation(match.resume.parsed or {}, match.job.parsed or {}, gaps)
    Recommendation.objects.create(match=match, kind=rec['kind'], text=rec['text'])
    return render(request, "recommendation.html", {"match": match, "recommendation": rec})

@csrf_exempt
def nl_query(request):
    if request.method == "POST":
        form = NLQueryForm(request.POST)
        if form.is_valid():
            q = form.cleaned_data['query']
            rid = form.cleaned_data.get('resume_id') or None
            jid = form.cleaned_data.get('job_id') or None
            resume = Resume.objects.filter(pk=rid).first() if rid else None
            job = JobDescription.objects.filter(pk=jid).first() if jid else None
            parsed_resume = resume.parsed if resume else {}
            parsed_job = job.parsed if job else {}
            session = NLSession.objects.create(resume=resume)
            result = utils.handle_nl_query(parsed_resume, parsed_job, [], q)
            session.turns = [{"user": q, "assistant": result.get('answer')}]
            session.save()
            return render(request, "nl_query.html", {"form": form, "answer": result, "session": session})
    else:
        form = NLQueryForm()
    return render(request, "nl_query.html", {"form": form})



def analyze_job_fit(request):
    """
    Unified workflow: Upload resume + job → Get match score, gaps, and recommendation all at once
    """
    if request.method == "POST":
        # Handle resume (file or text)
        if 'resume_file' in request.FILES:
            resume_file = request.FILES['resume_file']
            resume = Resume.objects.create(file=resume_file)
            utils.parse_and_store_resume(resume)
        elif 'resume_text' in request.POST and request.POST['resume_text'].strip():
            resume_text = request.POST['resume_text']
            resume = Resume.objects.create(raw_text=resume_text)
            prompt = utils.build_resume_prompt(resume_text)
            resume.parsed = utils.call_claude_parse(prompt)
            resume.save()
        else:
            return render(request, "analyze.html", {"error": "Please provide a resume"})
        
        # Handle job description (file or text)
        if 'job_file' in request.FILES:
            job_file = request.FILES['job_file']
            job = JobDescription.objects.create(file=job_file)
            utils.parse_and_store_job(job)
        elif 'job_text' in request.POST and request.POST['job_text'].strip():
            job_text = request.POST['job_text']
            job = JobDescription.objects.create(raw_text=job_text)
            job.parsed = utils.parse_job_description_text(job_text)
            job.title = job.parsed.get('title', 'Job')
            job.save()
        else:
            return render(request, "analyze.html", {"error": "Please provide a job description"})
        
        # STEP 3: Compute match score
        match_result = utils.semantic_match_score(resume.parsed or {}, job.parsed or {})
        
        # STEP 4: Perform gap analysis
        gaps = utils.gap_analysis(resume.parsed or {}, job.parsed or {}, top_n=5)
        
        # STEP 5: Generate personalized recommendation
        recommendation = utils.generate_personalized_recommendation(
            resume.parsed or {}, 
            job.parsed or {}, 
            gaps
        )
        
        # Save everything to database
        match = MatchResult.objects.create(
            resume=resume,
            job=job,
            score=match_result['score'],
            rating=match_result['rating'],
            breakdown=match_result['breakdown']
        )
        
        # Save gaps
        for g in gaps:
            Gap.objects.create(
                match=match,
                skill=g['skill'],
                type=g['type'],
                importance=g['importance'],
                suggestion=g['suggestion']
            )
        
        # Save recommendation
        Recommendation.objects.create(
            match=match,
            kind=recommendation['kind'],
            text=recommendation['text']
        )
        
        # Render unified results page
        context = {
            'match': match,
            'match_result': match_result,
            'gaps': gaps,
            'recommendation': recommendation,
            'resume': resume,
            'job': job
        }
        return render(request, "analysis_result.html", context)
    
    # GET request - show upload form
    return render(request, "analyze.html")