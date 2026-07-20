import json
import re
from typing import List, Dict, Any

from config import DB_PATH, LLM_MODEL
from db import get_connection, execute
from llm_manager import llm_chat
from hybrid_search import hybrid_search

TOP_K = 2


# ─────────────────────────────────────────────
# SQLITE HELPERS
# ─────────────────────────────────────────────

def ensure_education_column(
    db_path: str = DB_PATH,
) -> None:
    """
    Add education column if it doesn't already exist.
    """

    with get_connection(db_path) as conn:

        cur = conn.cursor()

        cur.execute("PRAGMA table_info(resume_chunks)")

        cols = [r["name"] for r in cur.fetchall()]

        if "education" not in cols:

            cur.execute(
                "ALTER TABLE resume_chunks ADD COLUMN education TEXT"
            )

            conn.commit()

            print("[INFO] 'education' column added.")

        else:

            print("[INFO] 'education' column already exists.")


def fetch_window_chunks(
    target_id: int,
    resume_id: str,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Fetch previous, current and next chunk.
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
            WHERE resume_id = ?
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

        rows = [dict(r) for r in cur.fetchall()]

    return rows


def save_education(
    target_id: int,
    education_json: Dict[str, Any],
    db_path: str = DB_PATH,
) -> None:

    execute(
        """
        UPDATE resume_chunks
        SET education = ?
        WHERE id = ?
        """,
        (
            json.dumps(
                education_json,
                ensure_ascii=False,
            ),
            target_id,
        ),
        db_path,
    )

    print(f"[OK] Education saved → chunk id={target_id}")


# ─────────────────────────────────────────────
# STEP 2 — LLM JUDGE
# ─────────────────────────────────────────────

def llm_judge_chunks_education(
    chunk1: Dict[str, Any],
    chunk2: Dict[str, Any],
) -> List[int]:

    prompt = f"""
You are a resume parsing assistant.

Below are two resume chunks.

Determine which chunk(s) contain EDUCATION information.

Education includes

- Degrees
- Diplomas
- Universities
- Colleges
- Schools
- GPA
- Graduation years
- Academic achievements
- Coursework

Chunk 1

Section:
{chunk1.get("chunk_type","")}

Content:
{chunk1.get("chunk_content","")}

Chunk 2

Section:
{chunk2.get("chunk_type","")}

Content:
{chunk2.get("chunk_content","")}

Return ONLY JSON.

{{
    "relevant_chunks":[1]
}}

Valid answers:

[1]

[2]

[1,2]
"""

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

        print("[WARN] Could not parse judge response.")

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


# ─────────────────────────────────────────────
# STEP 4 — LLM EDUCATION EXTRACTOR
# ─────────────────────────────────────────────

def llm_extract_education(
    window_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:

    context = []

    for chunk in window_chunks:

        context.append(
            f"[Section: {chunk.get('chunk_type','unknown')}]\n"
            f"{chunk.get('chunk_content','').strip()}"
        )

    prompt = f"""
You are an expert resume parser.

Resume Context

{chr(10).join(context)}

Extract ALL education entries.

Return ONLY JSON.

{{
  "education":[
    {{
      "degree":null,
      "field_of_study":null,
      "institution":null,
      "location":null,
      "start_year":null,
      "end_year":null,
      "gpa":null,
      "achievements":[],
      "relevant_courses":[]
    }}
  ]
}}

Rules

- Do not invent information.
- Use null if unavailable.
- achievements and relevant_courses should be [] if missing.
- Return [] if no education exists.
"""

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

    print(f"\n[EXTRACT RAW]\n{raw}\n")

    try:

        clean = re.sub(
            r"```(?:json)?|```",
            "",
            raw,
        ).strip()

        return json.loads(clean)

    except Exception:

        print("[WARN] Could not parse education JSON.")

        return {
            "raw_extraction": raw
        }
    
# ─────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────

def extract_education(
    resume_id: str,
    db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """
    Full pipeline

    1. Hybrid search
    2. LLM judge
    3. Fetch window
    4. LLM extract
    5. Save to DB
    """

    ensure_education_column(db_path)

    print(f"\n{'=' * 60}")
    print("[STEP 1] Hybrid Search")
    print(f"{'=' * 60}")

    results = hybrid_search(
        query="education degree university college graduation bachelor master phd",
        resume_id=resume_id,
        top_k=TOP_K,
        db_path=db_path,
    )

    if not results:

        print("[ERROR] No education chunks found.")

        return {}

    chunk1 = results[0]
    chunk2 = results[1] if len(results) > 1 else results[0]

    print(
        f"Chunk1 -> id={chunk1['id']} section={chunk1['chunk_type']}"
    )

    print(
        f"Chunk2 -> id={chunk2['id']} section={chunk2['chunk_type']}"
    )

    print(f"\n{'=' * 60}")
    print("[STEP 2] LLM Judge")
    print(f"{'=' * 60}")

    relevant_ids = llm_judge_chunks_education(
        chunk1,
        chunk2,
    )

    print(f"Relevant IDs : {relevant_ids}")

    if not relevant_ids:

        print("[WARN] No relevant education chunks.")

        return {}

    all_education_entries = []

    for target_id in relevant_ids:

        print(f"\n{'=' * 60}")
        print(f"[STEP 3] Fetch Window : chunk={target_id}")
        print(f"{'=' * 60}")

        window = fetch_window_chunks(
            target_id=target_id,
            resume_id=resume_id,
            db_path=db_path,
        )

        print(
            f"Window IDs : {[chunk['id'] for chunk in window]}"
        )

        print("\n[STEP 4] LLM Extraction")

        education = llm_extract_education(
            window
        )

        print("\n[STEP 5] Saving to DB")

        save_education(
            target_id=target_id,
            education_json=education,
            db_path=db_path,
        )

        entries = education.get(
            "education",
            [],
        )

        if not isinstance(entries, list):
            continue

        existing = {
            (
                item.get("degree"),
                item.get("institution"),
            )
            for item in all_education_entries
        }

        for item in entries:

            key = (
                item.get("degree"),
                item.get("institution"),
            )

            if key not in existing:

                all_education_entries.append(item)

                existing.add(key)

    final_output = {
        "education": all_education_entries
    }

    print(f"\n{'=' * 60}")
    print("[DONE] Education Extraction")
    print(f"{'=' * 60}")

    print(
        json.dumps(
            final_output,
            indent=2,
            ensure_ascii=False,
        )
    )

    return final_output

