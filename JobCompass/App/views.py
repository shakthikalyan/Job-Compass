from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from django.db.models import Avg
from django.utils import timezone
from django.http import JsonResponse, HttpResponseBadRequest
from django.urls import reverse, NoReverseMatch
from django.core.cache import cache

from .forms import NLQueryForm
from .models import (
    Resume, JobDescription, MatchResult,
    Gap, Recommendation, NLSession
)
from App.resume_parser import parse_and_store_resume, build_resume_prompt, call_parse
from .utils import (
    parse_job_llm,
    semantic_match_score,
    gap_analysis,
    generate_recommendations,
    nl_query,
    sanitize_text
)

import json
import logging
import uuid
import threading
import time

log = logging.getLogger(__name__)

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------

def index(request):
    stats = {
        "total_resumes": Resume.objects.count(),
        "total_jobs": JobDescription.objects.count(),
        "avg_match_score": round(
            MatchResult.objects.aggregate(avg=Avg('score'))['avg'] or 0.0, 2
        )
    }
    return render(request, "index.html", {
        "resumes": Resume.objects.order_by('-uploaded_at')[:10],
        "jobs": JobDescription.objects.order_by('-uploaded_at')[:10],
        "recent_matches": MatchResult.objects.order_by('-computed_at')[:5],
        "stats": stats
    })


def resume_detail(request, resume_id):
    return render(request, "resume_detail.html", {
        "resume": get_object_or_404(Resume, pk=resume_id)
    })


def job_detail(request, job_id):
    job = get_object_or_404(JobDescription, pk=job_id)
    admin_change_url = None
    try:
        admin_change_url = reverse(
            f'admin:{job._meta.app_label}_{job._meta.model_name}_change',
            args=[job.pk]
        )
    except NoReverseMatch:
        pass
    return render(request, "job_detail.html", {
        "job": job,
        "admin_change_url": admin_change_url
    })

# -------------------------------------------------
# ANALYZE ENTRY
# -------------------------------------------------

def analyze_job_fit(request):
    if request.method != "POST":
        return render(request, "analyze.html")

    # -------- Resume --------

    if 'resume_file' in request.FILES:
        resume = Resume.objects.create(file=request.FILES['resume_file'])
        raw = resume.file.read()
        text = raw.decode("utf-8", errors="ignore")
        parse_and_store_resume(resume)
    elif request.POST.get('resume_text', '').strip():
        resume_text = request.POST.get("resume_text", "").strip()
        if not resume_text:
            return render(request, "analyze.html", {"error": "Resume required"})
        resume = Resume.objects.create(raw_text=resume_text)
        resume.parsed = call_parse(build_resume_prompt(resume_text))
    else:
        return render(request, "analyze.html", {"error": "Please provide a resume"})

    # -------- Job --------
    if 'job_file' in request.FILES:
        job = JobDescription.objects.create(file=request.FILES['job_file'])
        raw = job.file.read()
        text = raw.decode("utf-8", errors="ignore")
        job.raw_text = sanitize_text(text)
        job.save()
    else:
        job_text = request.POST.get("job_text", "").strip()
        if not job_text:
            return render(request, "analyze.html", {"error": "Job description required"})
        job = JobDescription.objects.create(raw_text=job_text)

    job.parsed = parse_job_llm(job.raw_text)
    job.title = job.parsed.get("job_title", "Job")
    job.save()

    # -------- Session --------
    session_id = str(uuid.uuid4())
    request.session.update({
        "analysis_session_id": session_id,
        "resume_id": str(resume.id),
        "job_id": str(job.id)
    })

    return redirect("analyze_loading")

# -------------------------------------------------
# BACKGROUND WORKER
# -------------------------------------------------

def _background_analysis_worker(session_id, resume_id, job_id):
    cache_key = f"analysis_progress_{session_id}"
    cache.set(cache_key, {"status": "starting", "progress": 0}, timeout=1800)

    resume = Resume.objects.get(pk=resume_id)
    job = JobDescription.objects.get(pk=job_id)

    try:
        match = semantic_match_score(
        resume.parsed or {},
        job.parsed or {},
        occupation=job.occupation
    )
    except Exception as e:
        log.exception("Semantic match failed")
        match = {"score": 0, "rating": "Weak", "breakdown": {}}

    try:
        gaps = gap_analysis(resume.parsed or {}, job.parsed or {})
    except Exception as e:
        log.exception("Gap analysis failed")
        gaps = []
    
    try:
        recs = generate_recommendations(resume.parsed or {}, job.parsed or {}, gaps)
    except Exception as e:
        log.exception("Recommendation generation failed")
        recs = []

    match_obj = MatchResult.objects.create(
        resume=resume,
        job=job,
        score=match["score"],
        rating=match["rating"],
        breakdown=match["breakdown"]
    )

    for g in gaps:
        Gap.objects.create(
            match=match_obj,
            skill=g["skill"],
            type=g["type"],
            importance=g["importance"],
            suggestion=g["suggestion"]
        )

    for r in recs:
        Recommendation.objects.create(
            match=match_obj,
            kind="resume_tip",
            text=r["tip"],
            metadata={"skill": r["skill"]}
        )

    cache.set(cache_key, {
        "status": "done",
        "progress": 100,
        "match_id": str(match_obj.id)
    }, timeout=1800)


@require_POST
def analyze_start_background(request):
    sid = request.session.get("analysis_session_id")
    rid = request.session.get("resume_id")
    jid = request.session.get("job_id")

    if not all([sid, rid, jid]):
        return JsonResponse({"error": "Invalid session"}, status=400)

    threading.Thread(
        target=_background_analysis_worker,
        args=(sid, rid, jid),
        daemon=True
    ).start()

    return JsonResponse({"status": "started"})


@require_GET
def analyze_status(request):
    sid = request.session.get("analysis_session_id")
    if not sid:
        return JsonResponse({"status": "error"}, status=400)
    return JsonResponse(cache.get(f"analysis_progress_{sid}", {}))

# -------------------------------------------------
# RESULTS
# -------------------------------------------------

def analyze_loading(request):
    return render(request, "loading.html", {
        "session_id": request.session.get("analysis_session_id")
    })


def analyze_result_redirect(request):
    sid = request.session.get("analysis_session_id")
    data = cache.get(f"analysis_progress_{sid}", {})
    if data.get("status") != "done":
        return redirect("analyze_loading")
    return redirect("show_match_result", match_id=data["match_id"])


def show_match_result(request, match_id):
    match = get_object_or_404(MatchResult, pk=match_id)
    return render(request, "analysis_result.html", {
        "match": match,
        "gaps": list(match.gaps.values()),
        "recommendation": match.recommendations.first(),
        "resume": match.resume,
        "job": match.job,
        "match_score_display": int(round(match.score))
    })

# -------------------------------------------------
# NL CHATBOT
# -------------------------------------------------

@csrf_exempt
def nl_query(request):
    form = NLQueryForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        resume = Resume.objects.filter(pk=form.cleaned_data.get("resume_id")).first()
        job = JobDescription.objects.filter(pk=form.cleaned_data.get("job_id")).first()

        answer = nl_query(
            form.cleaned_data["query"],
            resume.parsed if resume else {},
            job.parsed if job else {},
            {},
            []
        )

        session = NLSession.objects.create(
            resume=resume,
            turns=[{"user": form.cleaned_data["query"], "assistant": answer}]
        )

        return render(request, "nl_query.html", {
            "form": form,
            "answer": answer,
            "session": session
        })

    return render(request, "nl_query.html", {"form": form})
