import os
import json
import re
import logging
from pathlib import Path
from typing import List, Dict

import spacy
from groq import Groq


class ResumeScreeningSystem:
    def __init__(self, api_key: str):
        """Initialize Resume Screening System."""
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

        # Validate API key
        if not api_key:
            raise ValueError("Groq API key is required")

        # Initialize Groq client
        self.client = Groq(api_key=api_key)

        # Load SpaCy model
        try:
            self.nlp = spacy.load('en_core_web_lg')
        except OSError:
            self.logger.error("SpaCy model not found. Install with: python -m spacy download en_core_web_lg")
            raise

    def extract_skills(self, data: Dict) -> List[str]:
        """Dynamically extract skills from any JSON structure."""
        skills = set()

        def traverse_json(item):
            if isinstance(item, dict):
                for key, value in item.items():
                    # Check for skill-related keys
                    if "skill" in key.lower() or "technology" in key.lower():
                        if isinstance(value, str):
                            skills.update(re.split(r'[,\n]', value))
                        elif isinstance(value, list):
                            skills.update(value)
                    else:
                        traverse_json(value)
            elif isinstance(item, list):
                for subitem in item:
                    traverse_json(subitem)

        traverse_json(data)
        return [skill.strip() for skill in skills if skill.strip()]

    def estimate_experience(self, data: Dict) -> float:
        """Estimate years of experience from the JSON data."""
        experience_years = []

        def traverse_json(item):
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, str):
                        matches = re.findall(r'(\d+)\s*(?:year|yr)', value, re.IGNORECASE)
                        experience_years.extend(int(match) for match in matches)
                    else:
                        traverse_json(value)
            elif isinstance(item, list):
                for subitem in item:
                    traverse_json(subitem)

        traverse_json(data)
        return max(experience_years, default=0)

    def extract_text_fields(self, data: Dict) -> str:
        """Extract all text content from the JSON structure."""
        text_content = []

        def traverse_json(item):
            if isinstance(item, dict):
                for value in item.values():
                    traverse_json(value)
            elif isinstance(item, list):
                for subitem in item:
                    traverse_json(subitem)
            elif isinstance(item, str):
                text_content.append(item)

        traverse_json(data)
        return ' '.join(text_content)

    def calculate_match_score(self, resume_data: Dict, job_details: Dict) -> Dict:
        """Calculate detailed match scores."""
        # Convert data to text for processing
        resume_text = self.extract_text_fields(resume_data)
        job_text = self.extract_text_fields(job_details)

        # SpaCy document creation
        resume_doc = self.nlp(resume_text)
        job_doc = self.nlp(job_text)

        # Semantic similarity
        semantic_score = resume_doc.similarity(job_doc) * 100

        # Skill matching
        resume_skills = self.extract_skills(resume_data)
        job_skills = self.extract_skills(job_details)

        skills_matched = list(set(resume_skills).intersection(set(job_skills)))
        skills_score = len(skills_matched) / len(job_skills) * 100 if job_skills else 0

        # Experience scoring
        resume_exp = self.estimate_experience(resume_data)
        job_min_exp = job_details.get('min_experience', 0)
        exp_score = min(resume_exp / job_min_exp * 100, 100) if job_min_exp > 0 else 50

        # Weighted total score
        total_score = (
                0.35 * semantic_score +
                0.35 * skills_score +
                0.30 * exp_score
        )

        return {
            'semantic_score': round(semantic_score, 2),
            'skills_score': round(skills_score, 2),
            'experience_score': round(exp_score, 2),
            'total_score': round(total_score, 2),
            'skills_matched': skills_matched
        }

    def process_resume(self, resume: Dict, job_details: Dict) -> Dict:
        """Process a single resume and calculate match scores."""
        try:
            match_scores = self.calculate_match_score(resume, job_details)
            result = {
                'name': resume.get('basics', {}).get('- full name', 'Unknown'),
                'current_role': resume.get('basics', {}).get('- current position', ''),
                'experience': self.estimate_experience(resume),
                **match_scores
            }
        except Exception as e:
            self.logger.error(f"Error processing resume: {e}")
            result = {
                'name': 'Unknown',
                'current_role': 'Unknown',
                'experience': 0,
                'semantic_score': 0.0,
                'skills_score': 0.0,
                'experience_score': 0.0,
                'total_score': 0.0,
                'skills_matched': []
            }
        return result

    def process_all_resumes(self, resume_dir: str, job_details: Dict):
        """Process all resumes in the given directory."""
        results = []
        for file in Path(resume_dir).glob('*.json'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    resume_data = json.load(f)
                    result = self.process_resume(resume_data, job_details)
                    results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing file {file}: {e}")
                continue

        # Sort results by total_score
        sorted_results = sorted(results, key=lambda x: x['total_score'], reverse=True)
        return sorted_results


def main():
    """Main function to run the resume screening process."""
    API_KEY = os.environ.get('GROQ_API_KEY', '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')
    RESUME_DIR = r"D:\Lusak.tech\real"

    # Initialize screening system
    screener = ResumeScreeningSystem(API_KEY)

    # Get job description
    print("\n===== Job Description Input =====")
    job_description = screener.extract_text_fields({
        "content": input("Paste the job description: ").strip()
    })

    # Process resumes
    print("\nProcessing resumes...")
    results = screener.process_all_resumes(RESUME_DIR, {"job_description": job_description})

    # Display top results
    print("\n=== Top 3 Matching Candidates ===")
    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}{'🏆' if i == 1 else ''}) {result['name']}")
        print(f"   Current Role: {result['current_role']}")
        print(f"   Experience: {result['experience']} years")
        print(f"   Skills Matched: {', '.join(result['skills_matched'])}")
        print(f"   Total Score: {result['total_score']}%")

    # Highlight the best match
    if results:
        best_match = results[0]
        print("\n🌟 BEST MATCH 🌟")
        print(f"Candidate: {best_match['name']}")
        print(f"Total Score: {best_match['total_score']}%")


if __name__ == "__main__":
    main()
