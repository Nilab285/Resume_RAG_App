
import os
import uuid

from config import DB_PATH

from db_ingest import process_resume
from certifications import extract_certifications
from personal_info_extractor import extract_personal_info_agentic
from skills_extractor import extract_skills
from work_experience import extract_experience
from education_extractor import extract_education
from resume_profile import transfer_to_profile


def generate_resume_id(pdf_path: str) -> str:
    """
    Generate a unique resume ID using the filename
    and a short UUID suffix.
    """

    filename = os.path.splitext(
        os.path.basename(pdf_path)
    )[0]

    filename = (
        filename.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    return f"{filename}_{uuid.uuid4().hex[:8]}"


def ingest_resume(pdf_path: str) -> dict:
    """
    Complete Resume Ingestion Pipeline.

    Returns
    -------
    {
        "status": "...",
        "resume_id": "...",
        "step": "...",
        "message": "..."
    }
    """

    if not os.path.exists(pdf_path):

        return {
            "status": "failed",
            "resume_id": None,
            "step": "File Validation",
            "message": f"File not found : {pdf_path}",
        }

    resume_id = generate_resume_id(pdf_path)

    try:

        print("=" * 70)
        print("STARTING RESUME INGESTION")
        print("=" * 70)

        print(f"Resume ID : {resume_id}")

        # --------------------------------------------------
        # INGEST PDF
        # --------------------------------------------------

        current_step = "PDF Ingestion"

        process_resume(
            resume_id=resume_id,
            pdf_path=pdf_path,
            db_path=DB_PATH,
        )

        # --------------------------------------------------
        # PERSONAL INFORMATION
        # --------------------------------------------------

        current_step = "Personal Information"

        extract_personal_info_agentic(
            resume_id=resume_id,
            db_path=DB_PATH,
        )

        # --------------------------------------------------
        # SKILLS
        # --------------------------------------------------

        current_step = "Skills Extraction"

        extract_skills(
            resume_id=resume_id,
            db_path=DB_PATH,
        )

        # --------------------------------------------------
        # WORK EXPERIENCE
        # --------------------------------------------------

        current_step = "Work Experience"

        extract_experience(
            resume_id=resume_id,
            db_path=DB_PATH,
        )

        # --------------------------------------------------
        # EDUCATION
        # --------------------------------------------------

        current_step = "Education"

        extract_education(
            resume_id=resume_id,
            db_path=DB_PATH,
        )

        # --------------------------------------------------
        # CERTIFICATIONS
        # --------------------------------------------------

        current_step = "Certifications"

        extract_certifications(
            resume_id=resume_id,
            db_path=DB_PATH,
        )

        # --------------------------------------------------
        # BUILD RESUME PROFILE
        # --------------------------------------------------

        current_step = "Resume Profile"

        transfer_to_profile(
            resume_id=resume_id,
            db_path=DB_PATH,
        )

        print("=" * 70)
        print("RESUME INGESTION COMPLETED")
        print("=" * 70)

        return {
            "status": "success",
            "resume_id": resume_id,
            "step": "Completed",
            "message": "Resume ingested successfully.",
        }

    except Exception as e:

        print(f"[ERROR] {current_step} failed")
        print(str(e))

        return {
            "status": "failed",
            "resume_id": resume_id,
            "step": current_step,
            "message": str(e),
        }