import os
import json
import re
import logging
import time
from pathlib import Path
from typing import TypedDict, List, Dict, Set, Optional, Any
from datetime import datetime, timezone
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import spacy
from groq import Groq
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz  # Replaced rapidfuzz with fuzzywuzzy

# Custom types
class EnhancedAnalysis(TypedDict):
    name: str
    current_role: str
    experience: float
    skills: Set[str]
    education: List[Dict[str, str]]
    projects: List[str]
    certifications: Set[str]

class MatchResult(TypedDict):
    role_score: float
    skills_score: float
    experience_score: float
    total_score: float
    skills_matched: List[str]
    skills_missing: List[str]

class SkillTaxonomyManager:
    """Manages dynamic skill taxonomy using LLM."""
    
    def __init__(self, client: Groq):
        self.client = client
        self.cache = {}
        self.logger = logging.getLogger(__name__)

    def get_skill_categories(self, skill: str) -> List[str]:
        """Dynamically determine categories for a given skill."""
        cache_key = f"categories_{skill}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"""
        Analyze the technical skill "{skill}" and determine all relevant categories it belongs to.
        Consider categories like:
        - Programming Languages
        - Frameworks
        - Libraries
        - Databases
        - Cloud Services
        - DevOps Tools
        - Operating Systems
        - and any other relevant technical categories
        
        Return only the category names as a comma-separated list.
        """

        try:
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are a technical skill categorization expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            categories = [
                cat.strip().lower()
                for cat in response.choices[0].message.content.split(',')
            ]
            self.cache[cache_key] = categories
            return categories
            
        except Exception as e:
            self.logger.error(f"Error getting skill categories for {skill}: {e}")
            return []

    def get_skill_aliases(self, skill: str) -> List[str]:
        """Dynamically generate aliases for a given skill."""
        cache_key = f"aliases_{skill}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"""
        List all common aliases, abbreviations, and variations for the technical skill "{skill}".
        Include:
        - Common abbreviations
        - Alternative names
        - Version-specific names
        - Industry-specific terms
        - Common misspellings
        
        Return only the aliases as a comma-separated list.
        """

        try:
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are a technical terminology expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            aliases = [
                alias.strip().lower()
                for alias in response.choices[0].message.content.split(',')
            ]
            self.cache[cache_key] = aliases
            return aliases
            
        except Exception as e:
            self.logger.error(f"Error getting skill aliases for {skill}: {e}")
            return [skill]

    def get_related_skills(self, skill: str) -> List[str]:
        """Dynamically determine related skills."""
        cache_key = f"related_{skill}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"""
        List technical skills commonly used together with "{skill}" in professional settings.
        Consider:
        - Complementary technologies
        - Common tech stack combinations
        - Tools frequently used together
        - Related frameworks or libraries
        - Typical skill combinations in job postings
        
        Return only the skill names as a comma-separated list, ordered by relevance.
        """

        try:
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are a technical stack analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            related_skills = [
                skill.strip().lower()
                for skill in response.choices[0].message.content.split(',')
            ]
            self.cache[cache_key] = related_skills
            return related_skills
            
        except Exception as e:
            self.logger.error(f"Error getting related skills for {skill}: {e}")
            return []

    def get_skill_prerequisites(self, skill: str) -> List[str]:
        """Dynamically determine prerequisite skills."""
        cache_key = f"prereq_{skill}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"""
        List the prerequisite technical skills typically needed to learn and use "{skill}" effectively.
        Consider:
        - Fundamental technologies required
        - Base languages or concepts needed
        - Common learning path prerequisites
        - Essential underlying skills
        
        Return only the prerequisite skill names as a comma-separated list, ordered by importance.
        """

        try:
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are a technical learning path expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            prerequisites = [
                skill.strip().lower()
                for skill in response.choices[0].message.content.split(',')
            ]
            self.cache[cache_key] = prerequisites
            return prerequisites
            
        except Exception as e:
            self.logger.error(f"Error getting prerequisites for {skill}: {e}")
            return []

class ResumeScreeningSystem:
    def __init__(self, api_key: str, rate_limit: int = 10):
        self.logger = self._configure_logging()
        self.client = Groq(api_key=api_key)
        self.rate_limit = rate_limit  # Requests per minute
        self.last_request_time = 0
        self.nlp = self._load_spacy_model()
        self.skill_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.skill_taxonomy = SkillTaxonomyManager(self.client)
        self._load_dynamic_patterns()

    def _configure_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler("screening_debug.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def _load_spacy_model(self) -> spacy.Language:
        try:
            nlp = spacy.load("en_core_web_lg")
            nlp.add_pipe("merge_entities")
            nlp.add_pipe("merge_noun_chunks")
            return nlp
        except IOError:
            raise RuntimeError("SpaCy model not found. Install with: python -m spacy download en_core_web_lg")

    def _load_dynamic_patterns(self) -> None:
        """Dynamically generate skill patterns using LLM."""
        prompt = """
        Generate a list of common technical skills for software development, 
        data science, and engineering roles. Return the skills as a JSON list.
        """
        response = self._call_llm(prompt)
        skills = json.loads(response)
        self.skill_matcher.add("SKILLS", [self.nlp.make_doc(skill) for skill in skills])

    def _call_llm(self, prompt: str, model: str = "mixtral-8x7b-32768") -> str:
        """Call Groq API with rate limiting."""
        current_time = time.time()
        if current_time - self.last_request_time < 60 / self.rate_limit:
            time.sleep(60 / self.rate_limit - (current_time - self.last_request_time))
        self.last_request_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return "[]"

    def process_resumes(self, resume_dir: str, job_desc: str) -> List[Dict]:
        """Process all resumes in a directory."""
        resume_paths = [p for p in Path(resume_dir).glob("**/*.json") if p.is_file()]
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self._process_single_resume, p, job_desc) for p in resume_paths]
            return [f.result() for f in futures if f.result() is not None]

    def _process_single_resume(self, resume_path: Path, job_desc: str) -> Optional[Dict]:
        """Process a single resume and calculate match scores."""
        try:
            resume_data = json.loads(resume_path.read_text(encoding='utf-8'))
            analysis = self.analyze_resume(resume_data)
            job_analysis = self.analyze_job_description(job_desc)
            scores = self.calculate_scores(analysis, job_analysis)
            return {
                "name": analysis.get("name", "Unknown"),
                "current_role": analysis.get("current_role", "Unknown"),
                "experience": analysis.get("experience", 0),
                **scores,
                "file_name": resume_path.name
            }
        except Exception as e:
            self.logger.error(f"Error processing {resume_path}: {e}")
            return None

    def analyze_resume(self, resume_data: Dict) -> EnhancedAnalysis:
        """Analyze a resume and extract key information."""
        text = self._extract_text(resume_data)
        doc = self.nlp(text)
        return {
            "name": self._extract_name(resume_data),
            "current_role": self._extract_current_role(doc),
            "experience": self._calculate_experience(doc),
            "skills": self._extract_skills(doc),
            "education": self._extract_education(doc),
            "projects": self._extract_projects(doc),
            "certifications": self._extract_certifications(doc)
        }

    def analyze_job_description(self, job_desc: str) -> EnhancedAnalysis:
        """Analyze a job description and extract requirements."""
        doc = self.nlp(job_desc)
        return {
            "name": "Job Description",
            "current_role": self._extract_current_role(doc),
            "experience": self._calculate_experience(doc),
            "skills": self._extract_skills(doc),
            "education": self._extract_education(doc),
            "projects": self._extract_projects(doc),
            "certifications": self._extract_certifications(doc)
        }

    def calculate_scores(self, resume: EnhancedAnalysis, job: EnhancedAnalysis) -> MatchResult:
        """Calculate match scores between resume and job description."""
        skills_match = self._calculate_skill_match(resume["skills"], job["skills"])
        experience_match = min(resume["experience"] / job["experience"], 1) * 100 if job["experience"] else 0
        role_match = fuzz.ratio(resume["current_role"], job["current_role"])

        total_score = (
            role_match * 0.3 +
            skills_match["score"] * 0.4 +
            experience_match * 0.3
        )

        return {
            "role_score": role_match,
            "skills_score": skills_match["score"],
            "experience_score": experience_match,
            "total_score": total_score,
            "skills_matched": skills_match["matched"],
            "skills_missing": skills_match["missing"]
        }

    def _calculate_skill_match(self, resume_skills: Set[str], job_skills: Set[str]) -> Dict:
        matched = [skill for skill in job_skills if skill in resume_skills]
        score = len(matched) / len(job_skills) * 100 if job_skills else 0
        return {"score": score, "matched": matched, "missing": list(job_skills - resume_skills)}

    def _extract_name(self, resume_data: Dict) -> str:
        return resume_data.get("basics", {}).get("- full name", "Unknown")

    def _extract_current_role(self, doc: Doc) -> str:
        for ent in doc.ents:
            if ent.label_ == "ROLE":
                return ent.text
        return "Unknown"

    def _calculate_experience(self, doc: Doc) -> float:
        experience = 0
        for ent in doc.ents:
            if ent.label_ == "DATE":
                years = re.findall(r'\d+', ent.text)
                if years:
                    experience += sum(int(year) for year in years if int(year) < 100)
        return experience

    def _extract_skills(self, doc: Doc) -> Set[str]:
        matches = self.skill_matcher(doc)
        return {doc[start:end].text.lower() for _, start, end in matches}

    def _extract_education(self, doc: Doc) -> List[Dict[str, str]]:
        return [{"degree": ent.text, "institution": "Unknown"} for ent in doc.ents if ent.label_ == "EDU"]

    def _extract_projects(self, doc: Doc) -> List[str]:
        return [sent.text for sent in doc.sents if "project" in sent.text.lower()]

    def _extract_certifications(self, doc: Doc) -> Set[str]:
        return {ent.text for ent in doc.ents if ent.label_ == "CERT"}

    def _extract_text(self, data: Dict) -> str:
        sections = data.get("sections", {})
        text = []
        for section in sections.values():
            if isinstance(section, list):
                for item in section:
                    if isinstance(item, dict):
                        text.extend(f"{k}: {v}" for k, v in item.items())
                    else:
                        text.append(item)
            elif isinstance(section, str):
                text.append(section)
        return "\n".join(text)

def main():
    """Main function to run the resume screening process."""
    try:
        api_key = os.getenv("GROQ_API_KEY", "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")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")

        screener = ResumeScreeningSystem(api_key=api_key, rate_limit=10)
        resume_dir = input("Enter the path to the resume directory: ").strip()
        job_desc = input("Paste the job description: ").strip()

        if not resume_dir or not job_desc:
            raise ValueError("Resume directory and job description are required")

        results = screener.process_resumes(resume_dir, job_desc)
        sorted_results = sorted(results, key=lambda x: x["total_score"], reverse=True)

        print("\n=== Top 3 Matching Candidates ===")
        for i, result in enumerate(sorted_results[:3], 1):
            print(f"\n{i}{'🏆' if i == 1 else ''}) {result['name']}")
            print(f"   Current Role: {result['current_role']}")
            print(f"   Experience: {result['experience']} years")
            print(f"   Role Match Score: {result['role_score']}%")
            print(f"   Skills Match Score: {result['skills_score']}%")
            print(f"   Experience Score: {result['experience_score']}%")
            print(f"   Matched Skills: {', '.join(result['skills_matched'])}")
            print(f"   Total Score: {result['total_score']}%")

        # Highlight best match
        if sorted_results:
            best_match = sorted_results[0]
            print("\n🌟 BEST MATCH 🌟")
            print(f"Candidate: {best_match['name']}")
            print(f"Total Score: {best_match['total_score']}%")
            print(f"Key Matched Skills: {', '.join(best_match['skills_matched'][:5])}")

    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()