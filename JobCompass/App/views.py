# App/views.py
from django.shortcuts import render, redirect, get_object_or_404
from .forms import NLQueryForm
from .models import Resume, JobDescription, MatchResult, Gap, Recommendation, NLSession
from . import utils
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Avg
from django.utils import timezone
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse
from django.urls import reverse, NoReverseMatch
import json
import logging
import uuid
import threading
import time
from django.core.cache import cache
from django.views.decorators.http import require_POST, require_GET

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


def analyze_job_fit(request):
    # identical to original code — unchanged
    # (parsing and storing resume/job objects)
    # Ensure utils.parse_and_store_resume and parse_job_description_text are called as before.
    if request.method == "POST":
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

        try:
            utils.get_embedding_model()
        except Exception:
            log.debug("embedding model preload attempt in analyze_job_fit failed.", exc_info=True)

        session_id = str(uuid.uuid4())
        request.session['analysis_session_id'] = session_id
        request.session['resume_id'] = str(resume.id)
        request.session['job_id'] = str(job.id)

        return redirect('analyze_loading')

    return render(request, "analyze.html")


def _background_analysis_worker(session_id, resume_pk, job_pk):
    cache_key = f"analysis_progress_{session_id}"
    cache.set(cache_key, {"status": "starting", "progress": 0, "steps": {
        "match": {"done": False, "progress": 0},
        "gaps": {"done": False, "progress": 0},
        "recommendation": {"done": False, "progress": 0}
    }}, timeout=60*30)

    try:
        resume = Resume.objects.get(pk=resume_pk)
        job = JobDescription.objects.get(pk=job_pk)
    except Exception as e:
        cache.set(cache_key, {"status": "error", "error": str(e), "progress": 100}, timeout=60*30)
        return

    try:
        utils.get_embedding_model()
    except Exception:
        log.debug("embedding model preload attempt in background worker failed.", exc_info=True)

    results = {}
    exceptions = {}

    def run_match():
        try:
            cache_data = cache.get(cache_key) or {}
            cache_data["status"] = "computing_match"
            cache.set(cache_key, cache_data, timeout=60*30)
            r = utils.semantic_match_score(resume.parsed or {}, job.parsed or {})
            results['match_result'] = r
            cache_data = cache.get(cache_key) or {}
            cache_data["steps"]["match"]["done"] = True
            cache_data["steps"]["match"]["progress"] = 100
            cache_data["progress"] = (
                (cache_data["steps"]["match"]["progress"] * 0.4) +
                (cache_data["steps"]["gaps"]["progress"] * 0.35) +
                (cache_data["steps"]["recommendation"]["progress"] * 0.25)
            )
            cache.set(cache_key, cache_data, timeout=60*30)
        except Exception as e:
            exceptions['match'] = str(e)
            log.exception("match computation failed in background worker")

    def run_gaps():
        try:
            cache_data = cache.get(cache_key) or {}
            cache_data["status"] = "computing_gaps"
            cache.set(cache_key, cache_data, timeout=60*30)
            g = utils.gap_analysis(resume.parsed or {}, job.parsed or {}, top_n=5)
            results['gaps'] = g
            cache_data = cache.get(cache_key) or {}
            cache_data["steps"]["gaps"]["done"] = True
            cache_data["steps"]["gaps"]["progress"] = 100
            cache_data["progress"] = (
                (cache_data["steps"]["match"]["progress"] * 0.4) +
                (cache_data["steps"]["gaps"]["progress"] * 0.35) +
                (cache_data["steps"]["recommendation"]["progress"] * 0.25)
            )
            cache.set(cache_key, cache_data, timeout=60*30)
        except Exception as e:
            exceptions['gaps'] = str(e)
            log.exception("gaps computation failed in background worker")

    def run_recommendation():
        try:
            cache_data = cache.get(cache_key) or {}
            cache_data["status"] = "computing_recommendation"
            cache.set(cache_key, cache_data, timeout=60*30)
            wait_cycles = 0
            while 'gaps' not in results and wait_cycles < 10:
                time.sleep(0.3)
                wait_cycles += 1
            rec = utils.generate_personalized_recommendation(resume.parsed or {}, job.parsed or {}, results.get('gaps', []))
            results['recommendation'] = rec
            cache_data = cache.get(cache_key) or {}
            cache_data["steps"]["recommendation"]["done"] = True
            cache_data["steps"]["recommendation"]["progress"] = 100
            cache_data["progress"] = (
                (cache_data["steps"]["match"]["progress"] * 0.4) +
                (cache_data["steps"]["gaps"]["progress"] * 0.35) +
                (cache_data["steps"]["recommendation"]["progress"] * 0.25)
            )
            cache.set(cache_key, cache_data, timeout=60*30)
        except Exception as e:
            exceptions['recommendation'] = str(e)
            log.exception("recommendation computation failed in background worker")

    t1 = threading.Thread(target=run_match, name=f"analysis-{session_id}-match")
    t2 = threading.Thread(target=run_gaps, name=f"analysis-{session_id}-gaps")
    t3 = threading.Thread(target=run_recommendation, name=f"analysis-{session_id}-rec")

    cache_data = cache.get(cache_key) or {}
    cache_data["steps"]["match"]["progress"] = 0
    cache_data["steps"]["gaps"]["progress"] = 0
    cache_data["steps"]["recommendation"]["progress"] = 0
    cache.set(cache_key, cache_data, timeout=60*30)

    t1.start(); t2.start(); t3.start()

    while t1.is_alive() or t2.is_alive() or t3.is_alive():
        cache_data = cache.get(cache_key) or {}
        for step in cache_data["steps"]:
            if not cache_data["steps"][step]["done"]:
                cache_data["steps"][step]["progress"] = min(95, cache_data["steps"][step].get("progress", 0) + 5)
        cache_data["progress"] = (
            (cache_data["steps"]["match"]["progress"] * 0.4) +
            (cache_data["steps"]["gaps"]["progress"] * 0.35) +
            (cache_data["steps"]["recommendation"]["progress"] * 0.25)
        )
        cache.set(cache_key, cache_data, timeout=60*30)
        time.sleep(0.25)

    t1.join(); t2.join(); t3.join()

    if 'match_result' in results:
        mm = results['match_result']
    else:
        mm = {"score": 0, "rating": "Unknown", "breakdown": {}}
    gaps = results.get('gaps', [])
    recommendation = results.get('recommendation', {"kind": "", "text": ""})

    try:
        match = MatchResult.objects.create(
            resume=resume,
            job=job,
            score=mm.get('score', 0),
            rating=mm.get('rating', ''),
            breakdown=mm.get('breakdown', {})
        )
        # Use related_name 'gaps' and 'recommendations' to create rows
        for g in gaps:
            Gap.objects.create(match=match, skill=g['skill'], type=g['type'], importance=g['importance'], suggestion=g['suggestion'])
        Recommendation.objects.create(match=match, kind=recommendation.get('kind',''), text=recommendation.get('text',''))

        score_pct = max(0.0, min(10.0, float(match.score or 0.0))) * 10.0
        stroke_dashoffset = 703.72 - (703.72 * (score_pct / 100.0))

        cache.set(cache_key, {
            "status": "done",
            "progress": 100,
            "match_id": str(match.pk),
            "stroke_dashoffset": stroke_dashoffset,
            "score_pct": round(score_pct),
        }, timeout=60*30)

    except Exception as e:
        cache.set(cache_key, {"status": "error", "error": str(e), "progress": 100}, timeout=60*30)
        log.exception("failed to persist analysis results in background worker")


@require_POST
def analyze_start_background(request):
    session_id = request.session.get('analysis_session_id')
    resume_id = request.session.get('resume_id')
    job_id = request.session.get('job_id')

    if not all([session_id, resume_id, job_id]):
        return JsonResponse({"status": "error", "error": "Missing session/resume/job ids"}, status=400)

    cache_key = f"analysis_progress_{session_id}"
    existing = cache.get(cache_key)
    if existing and existing.get("status") in ("starting", "computing_match", "computing_gaps", "computing_recommendation"):
        return JsonResponse({"status": "started"})

    worker_thread = threading.Thread(
        target=_background_analysis_worker,
        args=(session_id, resume_id, job_id),
        name=f"analysis-worker-{session_id}",
        daemon=True
    )
    worker_thread.start()
    return JsonResponse({"status": "started"})


@require_GET
def analyze_status(request):
    session_id = request.session.get('analysis_session_id')
    if not session_id:
        return JsonResponse({"status": "error", "error": "no session id"}, status=400)
    cache_key = f"analysis_progress_{session_id}"
    data = cache.get(cache_key) or {"status": "unknown", "progress": 0}
    return JsonResponse(data)


def analyze_loading(request):
    session_id = request.session.get('analysis_session_id')
    if not session_id:
        return redirect('analyze_job_fit')

    return render(request, "loading.html", {
        "session_id": session_id
    })

def analyze_process(request):
    session_id = request.session.get('analysis_session_id')
    resume_id = request.session.get('resume_id')
    job_id = request.session.get('job_id')

    if not all([session_id, resume_id, job_id]):
        return redirect('analyze_job_fit')

    resume = get_object_or_404(Resume, pk=resume_id)
    job = get_object_or_404(JobDescription, pk=job_id)

    try:
        utils.get_embedding_model()
    except Exception:
        log.debug("embedding model preload attempt in analyze_process failed.", exc_info=True)

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

    if 'analysis_session_id' in request.session:
        del request.session['analysis_session_id']
    if 'resume_id' in request.session:
        del request.session['resume_id']
    if 'job_id' in request.session:
        del request.session['job_id']

    context = {
        'match': match,
        'match_result': match_result,
        'gaps': gaps,
        'recommendation': recommendation,
        'resume': resume,
        'job': job
    }
    return render(request, "analysis_result.html", context)

def analyze_result_redirect(request):
    session_id = request.session.get('analysis_session_id')
    if not session_id:
        return redirect('analyze_job_fit')
    cache_key = f"analysis_progress_{session_id}"
    data = cache.get(cache_key) or {}
    if data.get("status") != "done":
        return redirect('analyze_loading')
    match_id = data.get("match_id")
    if not match_id:
        return redirect('analyze_job_fit')
    return redirect('show_match_result', match_id=match_id)


def show_match_result(request, match_id):
    match = get_object_or_404(MatchResult, pk=match_id)
    # round score to whole number for display (your score stored 0..10)
    score_val = float(match.score or 0.0)
    score_pct = max(0.0, min(10.0, score_val)) * 10.0  # convert 0-10 -> 0-100
    stroke_dashoffset = 703.72 - (703.72 * (score_pct / 100.0))
    match_score_display = int(round(score_val))

    # Use related_name 'gaps' and 'recommendations'
    gaps_qs = match.gaps.all().order_by('id')
    gaps = [{"skill": g.skill, "type": g.type, "importance": g.importance, "suggestion": g.suggestion} for g in gaps_qs]

    rec_obj = match.recommendations.first()
    recommendation = {"kind": rec_obj.kind if rec_obj else "", "text": rec_obj.text if rec_obj else ""}

    context = {
        "match": match,
        "match_score_display": match_score_display,
        "match_result": {"score": match.score, "rating": match.rating, "breakdown": match.breakdown},
        "gaps": gaps,
        "recommendation": recommendation,
        "resume": match.resume,
        "job": match.job,
        "stroke_dashoffset": stroke_dashoffset
    }
    return render(request, "analysis_result.html", context)


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

def show_match_result(request, match_id):
    match = get_object_or_404(MatchResult, pk=match_id)
    # round score to whole number for display
    score_val = float(match.score or 0.0)
    score_pct = max(0.0, min(10.0, score_val)) * 10.0  # convert 0-10 -> 0-100
    stroke_dashoffset = 703.72 - (703.72 * (score_pct / 100.0))
    # provide integer display
    match_score_display = int(round(score_val))

    gaps_qs = match.gaps.all().order_by('id')
    gaps = [{"skill": g.skill, "type": g.type, "importance": g.importance, "suggestion": g.suggestion} for g in gaps_qs]

    rec_obj = match.recommendations.first()
    recommendation = {"kind": rec_obj.kind if rec_obj else "", "text": rec_obj.text if rec_obj else ""}

    context = {
        "match": match,
        "match_score_display": match_score_display,
        "match_result": {"score": match.score, "rating": match.rating, "breakdown": match.breakdown},
        "gaps": gaps,
        "recommendation": recommendation,
        "resume": match.resume,
        "job": match.job,
        "stroke_dashoffset": stroke_dashoffset
    }
    return render(request, "analysis_result.html", context)


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
