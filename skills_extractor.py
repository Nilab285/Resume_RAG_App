# =============================================================
# skills_extractor.py
#
# Flow:
#   1. Hybrid search → top 2 chunks for "skills"
#   2. LLM judges which chunk(s) are relevant
#   3. For each relevant chunk → fetch window (id-1, id, id+1)
#   4. LLM extracts skills from window context
#   5. Save skills as JSON into SQLite
# =============================================================

import json
import re

from typing import List, Dict, Any, Optional

from hybrid_search import hybrid_search

from config import (
    DB_PATH,
    LLM_MODEL,
)

from db import (
    get_connection,
    execute,
)

from llm_manager import llm_chat


TOP_K = 2


# ---------------------------------------------------------
# SQLITE HELPERS
# ---------------------------------------------------------

def ensure_skills_column(
    db_path: str = DB_PATH,
) -> None:
    """
    Add skills column if it doesn't exist.
    """

    with get_connection(db_path) as conn:

        cur = conn.cursor()

        cur.execute("PRAGMA table_info(resume_chunks)")

        columns = [
            row["name"]
            for row in cur.fetchall()
        ]

        if "skills" not in columns:

            cur.execute(
                """
                ALTER TABLE resume_chunks
                ADD COLUMN skills TEXT
                """
            )

            conn.commit()

            print("[INFO] 'skills' column added.")

        else:

            print("[INFO] 'skills' column already exists.")


def fetch_chunk_by_id(
    chunk_id: int,
    db_path: str = DB_PATH,
) -> Optional[Dict[str, Any]]:

    with get_connection(db_path) as conn:

        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                id,
                resume_id,
                candidate_name,
                chunk_type,
                chunk_content
            FROM resume_chunks
            WHERE id=?
            """,
            (chunk_id,),
        )

        row = cur.fetchone()

        return dict(row) if row else None


def fetch_window_chunks(
    target_id: int,
    resume_id: str,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:

    """
    Fetch previous, current and next chunk
    for the same resume.
    """

    with get_connection(db_path) as conn:

        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                id,
                resume_id,
                candidate_name,
                chunk_type,
                chunk_content
            FROM resume_chunks
            WHERE resume_id=?
              AND id IN (?, ?, ?)
            ORDER BY id
            """,
            (
                resume_id,
                target_id - 1,
                target_id,
                target_id + 1,
            ),
        )

        return [
            dict(row)
            for row in cur.fetchall()
        ]


def save_skills(
    target_id: int,
    skills_json: Dict[str, Any],
    db_path: str = DB_PATH,
) -> None:
    """
    Save extracted skills.
    """

    execute(
        """
        UPDATE resume_chunks
        SET skills=?
        WHERE id=?
        """,
        (
            json.dumps(
                skills_json,
                ensure_ascii=False,
            ),
            target_id,
        ),
        db_path,
    )

    print(
        f"[OK] Skills saved → chunk id={target_id}"
    )

    # ---------------------------------------------------------
# STEP 2 — LLM JUDGE
# ---------------------------------------------------------

def llm_judge_chunks(
    chunk1: Dict[str, Any],
    chunk2: Dict[str, Any],
) -> List[int]:
    """
    Decide which retrieved chunks are actually
    relevant for extracting skills.
    """

    prompt = f"""
You are a resume parsing assistant.

Below are two text chunks extracted from a resume.

Your task is to decide which chunk(s) contain
skills-related information.

Skills include:

- Programming languages
- Frameworks
- Libraries
- Tools
- Platforms
- Technical skills
- Soft skills
- Certifications

------------- CHUNK 1 -------------

Section:
{chunk1.get("chunk_type","")}

Content:
{chunk1.get("chunk_content","")}

------------- CHUNK 2 -------------

Section:
{chunk2.get("chunk_type","")}

Content:
{chunk2.get("chunk_content","")}

Reply ONLY in JSON.

Example:

{{
    "relevant_chunks":[1]
}}

Valid outputs are:

[1]

[2]

[1,2]

Do not explain anything.
""".strip()

    response = llm_chat(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()

    print(f"\n[JUDGE RAW]\n{raw}\n")

    try:

        clean = re.sub(
            r"```(?:json)?|```",
            "",
            raw,
        ).strip()

        parsed = json.loads(clean)

        relevant = parsed.get(
            "relevant_chunks",
            [1],
        )

    except Exception:

        print(
            "[WARN] Could not parse judge output."
        )

        relevant = [1]

    id_map = {
        1: chunk1["id"],
        2: chunk2["id"],
    }

    return [
        id_map[i]
        for i in relevant
        if i in id_map
    ]


# ---------------------------------------------------------
# STEP 4 — LLM SKILLS EXTRACTOR
# ---------------------------------------------------------

def llm_extract_skills(
    window_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:

    """
    Extract skills from the contextual
    window around the target chunk.
    """

    context = "\n\n".join(
        [
            f"[Section: {chunk.get('chunk_type','')}]\n"
            f"{chunk.get('chunk_content','').strip()}"
            for chunk in window_chunks
        ]
    )

    prompt = f"""
You are an expert resume parser.

Below is a contextual excerpt from a resume.

{context}

Extract every skill.

Return ONLY JSON.

{{
  "technical_skills": [],
  "programming_languages": [],
  "frameworks_and_libraries": [],
  "tools_and_platforms": [],
  "soft_skills": []
}}

Rules

- Do not invent skills.
- Preserve spelling.
- Return empty list if none.
- No markdown.
- No explanation.
""".strip()

    response = llm_chat(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()

    print(f"\n[SKILLS RAW]\n{raw}\n")

    try:

        clean = re.sub(
            r"```(?:json)?|```",
            "",
            raw,
        ).strip()

        skills = json.loads(clean)

    except Exception:

        print(
            "[WARN] Invalid JSON from extractor."
        )

        skills = {
            "raw_extraction": raw
        }

    return skills

    # ---------------------------------------------------------
# MAIN ORCHESTRATOR
# ---------------------------------------------------------

def extract_skills(
    resume_id: str,
    db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """
    Full Skills Extraction Pipeline

    1. Hybrid Search
    2. LLM Judge
    3. Fetch Context Window
    4. LLM Extract
    5. Save Skills
    """

    ensure_skills_column(db_path)

    print(f"\n{'=' * 60}")
    print(f"[STEP 1] Hybrid Search")
    print(f"{'=' * 60}")

    results = hybrid_search(
        query="technical skills programming languages frameworks tools software technologies",
        resume_id=resume_id,
        top_k=TOP_K,
        db_path=db_path,
    )

    if not results:
        print("[ERROR] No chunks found.")
        return {}

    chunk1 = results[0]
    chunk2 = results[1] if len(results) > 1 else results[0]

    print(
        f"Chunk 1 -> id={chunk1['id']} "
        f"section={chunk1['chunk_type']}"
    )

    print(
        f"Chunk 2 -> id={chunk2['id']} "
        f"section={chunk2['chunk_type']}"
    )

    print(f"\n{'=' * 60}")
    print("[STEP 2] LLM Judge")
    print(f"{'=' * 60}")

    relevant_ids = llm_judge_chunks(
        chunk1=chunk1,
        chunk2=chunk2,
    )

    print(
        f"Relevant chunk ids : {relevant_ids}"
    )

    all_skills = {}

    for target_id in relevant_ids:

        print(f"\n{'=' * 60}")
        print(
            f"[STEP 3] Fetch Window "
            f"(chunk={target_id})"
        )
        print(f"{'=' * 60}")

        window = fetch_window_chunks(
            target_id=target_id,
            resume_id=resume_id,
            db_path=db_path,
        )

        print(
            "Window ids:",
            [c["id"] for c in window],
        )

        print(f"\n[STEP 4] Extract Skills")

        skills = llm_extract_skills(
            window_chunks=window,
        )

        print(
            f"\n[STEP 5] Saving Skills "
            f"(chunk={target_id})"
        )

        save_skills(
            target_id=target_id,
            skills_json=skills,
            db_path=db_path,
        )

        for category, values in skills.items():

            if category not in all_skills:
                all_skills[category] = []

            if isinstance(values, list):

                merged = (
                    all_skills[category]
                    + values
                )

                all_skills[category] = list(
                    dict.fromkeys(merged)
                )

    print(f"\n{'=' * 60}")
    print("[DONE] Skills Extraction Complete")
    print(f"{'=' * 60}")

    print(
        json.dumps(
            all_skills,
            indent=2,
            ensure_ascii=False,
        )
    )

    return all_skills


