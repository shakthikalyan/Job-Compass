from django.test import TestCase
from .models import Resume, JobDescription
from .views import semantic_match_scoring, gap_analysis, actionable_recommendations, career_queries

class ResumeModelTest(TestCase):
    def test_resume_creation(self):
        resume = Resume.objects.create(file="test_resume.pdf", parsed_text="Sample text")
        self.assertEqual(resume.parsed_text, "Sample text")

class JobDescriptionModelTest(TestCase):
    def test_job_description_creation(self):
        job = JobDescription.objects.create(title="Software Engineer", description="Develop software.")
        self.assertEqual(job.title, "Software Engineer")

class APITest(TestCase):
    def test_semantic_match_scoring(self):
        result = semantic_match_scoring("Sample resume text", "Sample job description text")
        self.assertIsInstance(result, str)

    def test_gap_analysis(self):
        result = gap_analysis("Sample resume text", "Sample job description text")
        self.assertIsInstance(result, str)

    def test_actionable_recommendations(self):
        result = actionable_recommendations("Sample resume text", "Sample job description text")
        self.assertIsInstance(result, str)

    def test_career_queries(self):
        result = career_queries("What skills are needed for a data scientist?")
        self.assertIsInstance(result, str)
