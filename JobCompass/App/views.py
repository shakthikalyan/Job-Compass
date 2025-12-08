# # App/views.py
# from django.shortcuts import render, redirect, get_object_or_404
# from .forms import ResumeUploadForm, JobUploadForm, NLQueryForm
# from .models import Resume, JobDescription, MatchResult, Gap, Recommendation, NLSession
# from . import utils
# from django.views.decorators.csrf import csrf_exempt
# from django.db.models import Avg
# from django.utils import timezone
# from django.http import JsonResponse, HttpResponseBadRequest
# from django.urls import reverse, NoReverseMatch
# import json

# def index(request):
#     resumes = Resume.objects.all().order_by('-uploaded_at')[:10]
#     jobs = JobDescription.objects.all().order_by('-uploaded_at')[:10]
#     total_resumes = Resume.objects.count()
#     total_jobs = JobDescription.objects.count()
#     avg_match = MatchResult.objects.aggregate(avg=Avg('score'))['avg'] or 0.0
#     stats = {
#         "total_resumes": total_resumes,
#         "total_jobs": total_jobs,
#         "avg_match_score": round(avg_match, 2)
#     }
#     recent_matches = MatchResult.objects.order_by('-computed_at')[:5]
#     return render(request, "index.html", {
#         "resumes": resumes,
#         "jobs": jobs,
#         "stats": stats,
#         "recent_matches": recent_matches
#     })



# def resume_detail(request, resume_id):
#     r = get_object_or_404(Resume, pk=resume_id)
#     return render(request, "resume_detail.html", {"resume": r})




# def compute_match(request, resume_id, job_id):
#     resume = get_object_or_404(Resume, pk=resume_id)
#     job = get_object_or_404(JobDescription, pk=job_id)
#     mm = utils.semantic_match_score(resume.parsed or {}, job.parsed or {})
#     mr = MatchResult.objects.create(resume=resume, job=job, score=mm['score'], rating=mm['rating'], breakdown=mm['breakdown'])
#     return render(request, "match_result.html", {"match": mr, "breakdown": mm['breakdown']})

# def gaps_view(request, match_id):
#     match = get_object_or_404(MatchResult, pk=match_id)
#     gaps = utils.gap_analysis(match.resume.parsed or {}, match.job.parsed or {}, top_n=5)
#     match.gaps.all().delete()
#     gap_objs = []
#     for g in gaps:
#         gap_objs.append(Gap.objects.create(match=match, skill=g['skill'], type=g['type'], importance=g['importance'], suggestion=g['suggestion']))
#     return render(request, "gaps.html", {"match": match, "gaps": gap_objs})

# def recommendation_view(request, match_id):
#     match = get_object_or_404(MatchResult, pk=match_id)
#     gaps = [{"skill": g.skill, "type": g.type, "importance": g.importance, "suggestion": g.suggestion} for g in match.gaps.all()]
#     rec = utils.generate_personalized_recommendation(match.resume.parsed or {}, match.job.parsed or {}, gaps)
#     Recommendation.objects.create(match=match, kind=rec['kind'], text=rec['text'])
#     return render(request, "recommendation.html", {"match": match, "recommendation": rec})

# @csrf_exempt
# def nl_query(request):
#     if request.method == "POST":
#         form = NLQueryForm(request.POST)
#         if form.is_valid():
#             q = form.cleaned_data['query']
#             rid = form.cleaned_data.get('resume_id') or None
#             jid = form.cleaned_data.get('job_id') or None
#             resume = Resume.objects.filter(pk=rid).first() if rid else None
#             job = JobDescription.objects.filter(pk=jid).first() if jid else None
#             parsed_resume = resume.parsed if resume else {}
#             parsed_job = job.parsed if job else {}
#             session = NLSession.objects.create(resume=resume)
#             result = utils.handle_nl_query(parsed_resume, parsed_job, [], q)
#             session.turns = [{"user": q, "assistant": result.get('answer')}]
#             session.save()
#             return render(request, "nl_query.html", {"form": form, "answer": result, "session": session})
#     else:
#         form = NLQueryForm()
#     return render(request, "nl_query.html", {"form": form})

# def analyze_job_fit(request):
#     """
#     Unified: upload resume + JD -> compute match, gaps, recommendation and save results.
#     """
#     if request.method == "POST":
#         # Resume
#         resume = None
#         if 'resume_file' in request.FILES:
#             resume_file = request.FILES['resume_file']
#             resume = Resume.objects.create(file=resume_file)
#             utils.parse_and_store_resume(resume)
#         elif request.POST.get('resume_text', '').strip():
#             resume_text = request.POST['resume_text']
#             resume = Resume.objects.create(raw_text=resume_text)
#             resume.parsed = utils.call_claude_parse(utils.build_resume_prompt(resume_text))
#             resume.save()
#         else:
#             return render(request, "analyze.html", {"error": "Please provide a resume"})

#         # Job
#         job = None
#         if 'job_file' in request.FILES:
#             job_file = request.FILES['job_file']
#             job = JobDescription.objects.create(file=job_file)
#             utils.parse_and_store_job(job)
#         elif request.POST.get('job_text', '').strip():
#             job_text = request.POST['job_text']
#             job = JobDescription.objects.create(raw_text=job_text)
#             job.parsed = utils.parse_job_description_text(job_text)
#             job.title = job.parsed.get('title', 'Job')
#             job.save()
#         else:
#             return render(request, "analyze.html", {"error": "Please provide a job description"})

#         # Compute and save results
#         match_result = utils.semantic_match_score(resume.parsed or {}, job.parsed or {})
#         gaps = utils.gap_analysis(resume.parsed or {}, job.parsed or {}, top_n=5)
#         recommendation = utils.generate_personalized_recommendation(resume.parsed or {}, job.parsed or {}, gaps)

#         match = MatchResult.objects.create(
#             resume=resume,
#             job=job,
#             score=match_result['score'],
#             rating=match_result['rating'],
#             breakdown=match_result['breakdown']
#         )
#         for g in gaps:
#             Gap.objects.create(match=match, skill=g['skill'], type=g['type'], importance=g['importance'], suggestion=g['suggestion'])
#         Recommendation.objects.create(match=match, kind=recommendation['kind'], text=recommendation['text'])

#         context = {
#             'match': match,
#             'match_result': match_result,
#             'gaps': gaps,
#             'recommendation': recommendation,
#             'resume': resume,
#             'job': job
#         }
#         return render(request, "analysis_result.html", context)

#     return render(request, "analyze.html")


# @csrf_exempt
# def analyze_and_show_resume(request):
#     """
#     Accepts a POST from the Analyze form (resume_file or resume_text).
#     Parses and saves the resume using utils.parse_and_store_resume (same pipeline),
#     then redirects to the resume_detail page to show resume_detail.html.

#     Usage: POST same form fields as analyze.html:
#       - resume_file (file) OR resume_text (text)
#     """
#     if request.method == "POST":
#         # Resume by file
#         if 'resume_file' in request.FILES:
#             resume_file = request.FILES['resume_file']
#             # create Resume record with uploaded file (reusing app's model)
#             r = Resume.objects.create(file=resume_file)
#             # parse and persist parsed JSON into r.parsed
#             utils.parse_and_store_resume(r)

#             # redirect to resume detail page to view parsed results
#             return redirect('resume_detail', resume_id=r.id)

#         # Resume by pasted text
#         if request.POST.get('resume_text', '').strip():
#             resume_text = request.POST['resume_text']
#             r = Resume.objects.create(raw_text=resume_text)
#             # use the same lightweight parser used elsewhere in the app
#             r.parsed = utils.call_claude_parse(utils.build_resume_prompt(resume_text))
#             r.save()
#             return redirect('resume_detail', resume_id=r.id)

#         # If no resume provided, re-render analyze form with an error (keeps original UI)
#         return render(request, "analyze.html", {"error": "Please provide a resume (file or text)."})
#     # If GET, simply show the analyze page (same as existing view)
#     return render(request, "analyze.html")


# @csrf_exempt
# def analyze_and_show_job(request):
#     """
#     Accepts POST with job_file OR job_text, parse+persist using utils,
#     then redirect to job_detail.
#     """
#     if request.method == "POST":
#         # file upload
#         if 'job_file' in request.FILES:
#             job_file = request.FILES['job_file']
#             j = JobDescription.objects.create(file=job_file, uploaded_at=timezone.now())
#             # try robust parse_and_store_job (reads file if needed)
#             try:
#                 utils.parse_and_store_job(j)
#             except Exception as e:
#                 # fallback: simple parse from extracted text
#                 try:
#                     txt = utils.simple_text_from_file(job_file) or ""
#                     parsed = utils.parse_job_description_text(txt)
#                     j.parsed = parsed
#                     j.title = parsed.get('title', j.title or 'Job')
#                     j.save()
#                 except Exception:
#                     j.parsed = {"error": f"parsing failed: {str(e)}"}
#                     j.save()
#             return redirect('job_detail', job_id=j.id)

#         # pasted text
#         job_text = request.POST.get('job_text', '').strip()
#         if job_text:
#             j = JobDescription.objects.create(raw_text=job_text, uploaded_at=timezone.now())
#             try:
#                 parsed = utils.parse_job_description_text(job_text)
#                 j.parsed = parsed
#                 j.title = parsed.get('title', j.title or (job_text.splitlines()[0] if job_text.splitlines() else "Job"))
#                 j.save()
#             except Exception as e:
#                 j.parsed = {"error": f"parsing failed: {str(e)}"}
#                 j.save()
#             return redirect('job_detail', job_id=j.id)

#         # nothing provided
#         return render(request, "analyze.html", {"error": "Please provide a job description (file or text)."})
#     return render(request, "analyze.html")


# def job_detail(request, job_id):
#     """
#     Show job detail page. Ensure job.parsed is a dict and compute admin_change_url safely.
#     """
#     job = get_object_or_404(JobDescription, pk=job_id)

#     # Ensure parsed is a Python dict (sometimes saved as JSON string)
#     if isinstance(job.parsed, str):
#         try:
#             job.parsed = json.loads(job.parsed)
#         except Exception:
#             # not JSON, wrap as text
#             job.parsed = {"raw": job.parsed}

#     # ensure parsed has required keys (avoid template errors)
#     if not isinstance(job.parsed, dict):
#         job.parsed = {}

#     # compute admin change url safely
#     admin_change_url = None
#     try:
#         name = f'admin:{job._meta.app_label}_{job._meta.model_name}_change'
#         admin_change_url = reverse(name, args=[job.pk])
#     except NoReverseMatch:
#         admin_change_url = None

#     context = {
#         "job": job,
#         "admin_change_url": admin_change_url
#     }
#     return render(request, "job_detail.html", context)


# def analyze_job_json(request, job_id=None):
#     """
#     Useful debug endpoint:
#       - If job_id provided: return parsed JSON for that job
#       - If POST with job_text or job_file: parse and return JSON directly (no DB save)
#     """
#     if request.method == "GET":
#         if not job_id:
#             return HttpResponseBadRequest("Provide job_id as URL parameter or POST job_text/job_file.")
#         job = get_object_or_404(JobDescription, pk=job_id)
#         parsed = job.parsed
#         # normalize if string
#         if isinstance(parsed, str):
#             try:
#                 parsed = json.loads(parsed)
#             except Exception:
#                 parsed = {"raw": parsed}
#         return JsonResponse({"job_id": str(job.pk), "parsed": parsed}, json_dumps_params={"indent": 2})

#     # POST: parse provided job_text or job_file but do not save
#     if request.method == "POST":
#         if 'job_file' in request.FILES:
#             txt = utils.simple_text_from_file(request.FILES['job_file']) or ""
#             parsed = utils.parse_job_description_text(txt)
#             return JsonResponse({"parsed": parsed}, json_dumps_params={"indent": 2})
#         job_text = request.POST.get('job_text', '').strip()
#         if job_text:
#             parsed = utils.parse_job_description_text(job_text)
#             return JsonResponse({"parsed": parsed}, json_dumps_params={"indent": 2})
#         return HttpResponseBadRequest("Provide job_text or job_file in POST.")


# App/views.py
from django.shortcuts import render, redirect, get_object_or_404
from .forms import ResumeUploadForm, JobUploadForm, NLQueryForm
from .models import Resume, JobDescription, MatchResult, Gap, Recommendation, NLSession
from . import utils
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Avg
from django.utils import timezone
from django.http import JsonResponse, HttpResponseBadRequest
from django.urls import reverse, NoReverseMatch
import json
import logging

log = logging.getLogger(__name__)

def index(request):
    resumes = Resume.objects.all().order_by('-uploaded_at')[:10]
    jobs = JobDescription.objects.all().order_by('-uploaded_at')[:10]
    total_resumes = Resume.objects.count()
    total_jobs = JobDescription.objects.count()
    avg_match = MatchResult.objects.aggregate(avg=Avg('score'))['avg'] or 0.0
    stats = {
        "total_resumes": total_resumes,
        "total_jobs": total_jobs,
        "avg_match_score": round(avg_match, 2)
    }
    recent_matches = MatchResult.objects.order_by('-computed_at')[:5]
    return render(request, "index.html", {
        "resumes": resumes,
        "jobs": jobs,
        "stats": stats,
        "recent_matches": recent_matches
    })

def resume_detail(request, resume_id):
    r = get_object_or_404(Resume, pk=resume_id)
    return render(request, "resume_detail.html", {"resume": r})

def compute_match(request, resume_id, job_id):
    resume = get_object_or_404(Resume, pk=resume_id)
    job = get_object_or_404(JobDescription, pk=job_id)
    # ensure parsed exist
    resume_parsed = resume.parsed or {}
    job_parsed = job.parsed or {}
    # ensure embedding model has opportunity to load (if not loaded)
    try:
        utils.get_embedding_model()
    except Exception:
        log.debug("embedding model preload attempt in compute_match failed.", exc_info=True)
    mm = utils.semantic_match_score(resume_parsed, job_parsed)
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
    # ensure gaps are present on match (compute if missing)
    if not match.gaps.exists():
        gaps = utils.gap_analysis(match.resume.parsed or {}, match.job.parsed or {}, top_n=5)
        for g in gaps:
            Gap.objects.create(match=match, skill=g['skill'], type=g['type'], importance=g['importance'], suggestion=g['suggestion'])
    gaps = [{"skill": g.skill, "type": g.type, "importance": g.importance, "suggestion": g.suggestion} for g in match.gaps.all()]
    rec = utils.generate_personalized_recommendation(match.resume.parsed or {}, match.job.parsed or {}, gaps)
    Recommendation.objects.create(match=match, kind=rec.get('kind',''), text=rec.get('text',''))
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
    Unified: upload resume + JD -> compute match, gaps, recommendation and save results.
    """
    if request.method == "POST":
        # Resume
        resume = None
        if 'resume_file' in request.FILES:
            resume_file = request.FILES['resume_file']
            resume = Resume.objects.create(file=resume_file)
            utils.parse_and_store_resume(resume)
        elif request.POST.get('resume_text', '').strip():
            resume_text = request.POST['resume_text']
            resume = Resume.objects.create(raw_text=resume_text)
            resume.parsed = utils.call_claude_parse(utils.build_resume_prompt(resume_text))
            resume.save()
        else:
            return render(request, "analyze.html", {"error": "Please provide a resume"})

        # Job
        job = None
        if 'job_file' in request.FILES:
            job_file = request.FILES['job_file']
            job = JobDescription.objects.create(file=job_file)
            utils.parse_and_store_job(job)
        elif request.POST.get('job_text', '').strip():
            job_text = request.POST['job_text']
            job = JobDescription.objects.create(raw_text=job_text)
            job.parsed = utils.parse_job_description_text(job_text)
            job.title = job.parsed.get('title', 'Job')
            job.save()
        else:
            return render(request, "analyze.html", {"error": "Please provide a job description"})

        # ensure embedding model warm (best-effort)
        try:
            utils.get_embedding_model()
        except Exception:
            log.debug("embedding model preload attempt in analyze_job_fit failed.", exc_info=True)

        # Compute and save results
        match_result = utils.semantic_match_score(resume.parsed or {}, job.parsed or {})
        gaps = utils.gap_analysis(resume.parsed or {}, job.parsed or {}, top_n=5)
        recommendation = utils.generate_personalized_recommendation(resume.parsed or {}, job.parsed or {}, gaps)

        match = MatchResult.objects.create(
            resume=resume,
            job=job,
            score=match_result['score'],
            rating=match_result['rating'],
            breakdown=match_result['breakdown']
        )
        for g in gaps:
            Gap.objects.create(match=match, skill=g['skill'], type=g['type'], importance=g['importance'], suggestion=g['suggestion'])
        Recommendation.objects.create(match=match, kind=recommendation.get('kind',''), text=recommendation.get('text',''))

        context = {
            'match': match,
            'match_result': match_result,
            'gaps': gaps,
            'recommendation': recommendation,
            'resume': resume,
            'job': job
        }
        return render(request, "analysis_result.html", context)

    return render(request, "analyze.html")

@csrf_exempt
def analyze_and_show_resume(request):
    if request.method == "POST":
        if 'resume_file' in request.FILES:
            resume_file = request.FILES['resume_file']
            r = Resume.objects.create(file=resume_file)
            utils.parse_and_store_resume(r)
            return redirect('resume_detail', resume_id=r.id)
        if request.POST.get('resume_text', '').strip():
            resume_text = request.POST['resume_text']
            r = Resume.objects.create(raw_text=resume_text)
            r.parsed = utils.call_claude_parse(utils.build_resume_prompt(resume_text))
            r.save()
            return redirect('resume_detail', resume_id=r.id)
        return render(request, "analyze.html", {"error": "Please provide a resume (file or text)."})
    return render(request, "analyze.html")

@csrf_exempt
def analyze_and_show_job(request):
    if request.method == "POST":
        if 'job_file' in request.FILES:
            job_file = request.FILES['job_file']
            j = JobDescription.objects.create(file=job_file, uploaded_at=timezone.now())
            try:
                utils.parse_and_store_job(j)
            except Exception as e:
                try:
                    txt = utils.simple_text_from_file(job_file) or ""
                    parsed = utils.parse_job_description_text(txt)
                    j.parsed = parsed
                    j.title = parsed.get('title', j.title or 'Job')
                    j.save()
                except Exception:
                    j.parsed = {"error": f"parsing failed: {str(e)}"}
                    j.save()
            return redirect('job_detail', job_id=j.id)
        job_text = request.POST.get('job_text', '').strip()
        if job_text:
            j = JobDescription.objects.create(raw_text=job_text, uploaded_at=timezone.now())
            try:
                parsed = utils.parse_job_description_text(job_text)
                j.parsed = parsed
                j.title = parsed.get('title', j.title or (job_text.splitlines()[0] if job_text.splitlines() else "Job"))
                j.save()
            except Exception as e:
                j.parsed = {"error": f"parsing failed: {str(e)}"}
                j.save()
            return redirect('job_detail', job_id=j.id)
        return render(request, "analyze.html", {"error": "Please provide a job description (file or text)."})
    return render(request, "analyze.html")

def job_detail(request, job_id):
    job = get_object_or_404(JobDescription, pk=job_id)
    if isinstance(job.parsed, str):
        try:
            job.parsed = json.loads(job.parsed)
        except Exception:
            job.parsed = {"raw": job.parsed}
    if not isinstance(job.parsed, dict):
        job.parsed = {}
    admin_change_url = None
    try:
        name = f'admin:{job._meta.app_label}_{job._meta.model_name}_change'
        admin_change_url = reverse(name, args=[job.pk])
    except NoReverseMatch:
        admin_change_url = None
    context = {
        "job": job,
        "admin_change_url": admin_change_url
    }
    return render(request, "job_detail.html", context)

def analyze_job_json(request, job_id=None):
    if request.method == "GET":
        if not job_id:
            return HttpResponseBadRequest("Provide job_id as URL parameter or POST job_text/job_file.")
        job = get_object_or_404(JobDescription, pk=job_id)
        parsed = job.parsed
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except Exception:
                parsed = {"raw": parsed}
        return JsonResponse({"job_id": str(job.pk), "parsed": parsed}, json_dumps_params={"indent": 2})
    if request.method == "POST":
        if 'job_file' in request.FILES:
            txt = utils.simple_text_from_file(request.FILES['job_file']) or ""
            parsed = utils.parse_job_description_text(txt)
            return JsonResponse({"parsed": parsed}, json_dumps_params={"indent": 2})
        job_text = request.POST.get('job_text', '').strip()
        if job_text:
            parsed = utils.parse_job_description_text(job_text)
            return JsonResponse({"parsed": parsed}, json_dumps_params={"indent": 2})
        return HttpResponseBadRequest("Provide job_text or job_file in POST.")
