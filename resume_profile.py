import json
from typing import Optional

from config import DB_PATH
from db import get_connection, execute


# ─────────────────────────────────────────────
# TEXT BUILDERS (Pure Python)
# ─────────────────────────────────────────────

def build_skills_text(
    skills_json_str: Optional[str],
) -> str:
    """
    Flatten skills JSON into a comma-separated string.
    """

    if not skills_json_str:
        return ""

    try:

        skills_json = json.loads(skills_json_str)

        ordered_keys = [
            "programming_languages",
            "frameworks_and_libraries",
            "tools_and_platforms",
            "technical_skills",
            "soft_skills",
        ]

        all_items = []

        for key in ordered_keys:

            if key in skills_json:

                value = skills_json[key]

                if isinstance(value, list):
                    all_items.extend(value)

        for key, value in skills_json.items():

            if key in ordered_keys:
                continue

            if isinstance(value, list):
                all_items.extend(value)

        seen = set()
        unique = []

        for item in all_items:

            if not item:
                continue

            value = item.strip()

            if not value:
                continue

            lower = value.lower()

            if lower not in seen:

                seen.add(lower)
                unique.append(value)

        return ", ".join(unique)

    except Exception as e:

        print(f"[WARN] Could not build skills_text: {e}")

        return ""


def build_work_details_text(
    work_details_json_str: Optional[str],
) -> str:
    """
    Flatten work experience JSON into searchable text.
    """

    if not work_details_json_str:
        return ""

    try:

        work_json = json.loads(work_details_json_str)

        if isinstance(work_json, dict) and "experience" in work_json:

            jobs = work_json["experience"]

        elif isinstance(work_json, list):

            jobs = work_json

        elif isinstance(work_json, dict):

            jobs = [work_json]

        else:

            return ""

        lines = []

        for job in jobs:

            tokens = []

            if job.get("job_title"):
                tokens.append(job["job_title"])

            if job.get("company"):
                tokens.append(job["company"])

            if job.get("location"):
                tokens.append(job["location"])

            start = job.get("start_date", "")
            end = job.get("end_date", "")

            if start or end:
                tokens.append(f"{start} - {end}".strip(" -"))

            if isinstance(job.get("responsibilities"), list):
                tokens.extend(job["responsibilities"])

            if isinstance(job.get("technologies_used"), list):
                tokens.extend(job["technologies_used"])

            if tokens:
                lines.append(" | ".join(tokens))

        return "\n".join(lines)

    except Exception as e:

        print(f"[WARN] Could not build work_details_text: {e}")

        return ""


def build_certifications_text(
    certifications_json_str: Optional[str],
) -> str:
    """
    Flatten certifications JSON into searchable text.
    """

    if not certifications_json_str:
        return ""

    try:

        cert_json = json.loads(certifications_json_str)

        parts = []

        for cert in cert_json.get("certifications", []):

            name = (cert.get("name") or "").strip()

            if not name:
                continue

            issuer = (cert.get("issuer") or "").strip()
            date = (cert.get("date") or "").strip()

            token = name

            if issuer:
                token += f" ({issuer})"

            if date:
                token += f" — {date}"

            parts.append(token)

        for training in cert_json.get("trainings", []):

            name = (training.get("name") or "").strip()

            if not name:
                continue

            if name.lower() in {
                "internal training",
                "training",
                "bootcamp",
                "online course",
                "workshop",
            }:
                continue

            provider = (training.get("provider") or "").strip()
            ttype = (training.get("type") or "").strip()
            date = (training.get("date") or "").strip()

            token = f"[Training] {name}"

            if provider:
                token += f" ({provider})"

            elif ttype:
                token += f" ({ttype})"

            if date and date.lower() != "completed":
                token += f" — {date}"

            parts.append(token)

        return " | ".join(parts)

    except Exception as e:

        print(f"[WARN] Could not build certifications_text: {e}")

        return ""
    
# ─────────────────────────────────────────────
# MAIN TRANSFER FUNCTION
# ─────────────────────────────────────────────

def transfer_to_profile(
    resume_id: str,
    db_path: str = DB_PATH,
) -> None:
    """
    Consolidate extracted information from resume_chunks
    and upsert one row into resume_profiles.
    """

    print(f"\n{'=' * 60}")
    print(f"[TRANSFER] resume_id={resume_id}")
    print(f"{'=' * 60}")

    # --------------------------------------------------
    # STEP 1 : Fetch all chunks
    # --------------------------------------------------

    with get_connection(db_path) as conn:

        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                candidate_name,
                personal_info,
                personal_info_feedback,
                skills,
                education,
                education_exp,
                work_details,
                work_exp,
                certifications
            FROM resume_chunks
            WHERE resume_id = ?
            ORDER BY id
            """,
            (resume_id,),
        )

        rows = [dict(r) for r in cur.fetchall()]

    if not rows:

        print(f"[ERROR] No chunks found for {resume_id}")

        return

    print(f"[INFO] Total chunks : {len(rows)}")

    # --------------------------------------------------
    # STEP 2 : Consolidate values
    # --------------------------------------------------

    consolidated = {
        "candidate_name": None,
        "personal_info": None,
        "personal_info_feedback": None,
        "skills": None,
        "education": None,
        "education_exp": None,
        "work_details": None,
        "work_exp": None,
        "certifications": None,
    }

    for row in rows:

        for column in consolidated:

            if (
                consolidated[column] is None
                and row.get(column) is not None
            ):
                consolidated[column] = row[column]

    # --------------------------------------------------
    # STEP 3 : Build searchable text
    # --------------------------------------------------

    skills_text = build_skills_text(
        consolidated["skills"]
    )

    work_details_text = build_work_details_text(
        consolidated["work_details"]
    )

    certifications_text = build_certifications_text(
        consolidated["certifications"]
    )

    print(f"\nCandidate Name : {consolidated['candidate_name']}")
    print(f"Education Exp  : {consolidated['education_exp']}")
    print(f"Work Exp       : {consolidated['work_exp']}")

    # --------------------------------------------------
    # STEP 4 : Upsert resume_profiles
    # --------------------------------------------------

    execute(
        """
        INSERT INTO resume_profiles (

            resume_id,
            candidate_name,
            personal_info,
            personal_info_feedback,

            skills,
            skills_text,

            education,
            education_exp,

            work_details,
            work_details_text,
            work_exp,

            certifications,
            certifications_text

        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        ON CONFLICT(resume_id)
        DO UPDATE SET

            candidate_name         = excluded.candidate_name,
            personal_info          = excluded.personal_info,
            personal_info_feedback = excluded.personal_info_feedback,

            skills                 = excluded.skills,
            skills_text            = excluded.skills_text,

            education              = excluded.education,
            education_exp          = excluded.education_exp,

            work_details           = excluded.work_details,
            work_details_text      = excluded.work_details_text,
            work_exp               = excluded.work_exp,

            certifications         = excluded.certifications,
            certifications_text    = excluded.certifications_text
        """,
        (
            resume_id,

            consolidated["candidate_name"],
            consolidated["personal_info"],
            consolidated["personal_info_feedback"],

            consolidated["skills"],
            skills_text,

            consolidated["education"],
            consolidated["education_exp"],

            consolidated["work_details"],
            work_details_text,
            consolidated["work_exp"],

            consolidated["certifications"],
            certifications_text,
        ),
        db_path,
    )

    print("\n[OK] resume_profiles updated successfully.")
    print(f"{'=' * 60}\n")


