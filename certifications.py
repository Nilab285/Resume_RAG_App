
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

def ensure_certifications_column(
    db_path: str = DB_PATH,
) -> None:
    """
    Add certifications column to resume_chunks if it doesn't exist.
    """

    with get_connection(db_path) as conn:

        cur = conn.cursor()

        cur.execute("PRAGMA table_info(resume_chunks)")
        cols = [r["name"] for r in cur.fetchall()]

        if "certifications" not in cols:

            cur.execute(
                "ALTER TABLE resume_chunks ADD COLUMN certifications TEXT"
            )

            conn.commit()

            print("[INFO] 'certifications' column added.")

        else:

            print("[INFO] 'certifications' column already exists.")


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


def save_certifications(
    target_id: int,
    certifications_json: Dict[str, Any],
    db_path: str = DB_PATH,
) -> None:

    execute(
        """
        UPDATE resume_chunks
        SET certifications = ?
        WHERE id = ?
        """,
        (
            json.dumps(
                certifications_json,
                ensure_ascii=False,
            ),
            target_id,
        ),
        db_path,
    )

    print(f"[OK] Certifications saved → chunk id={target_id}")


# ─────────────────────────────────────────────
# STEP 2 — LLM JUDGE
# ─────────────────────────────────────────────

def llm_judge_chunks(
    chunk1: Dict[str, Any],
    chunk2: Dict[str, Any],
) -> List[int]:

    prompt = f"""
You are a resume parsing assistant.

Below are two resume chunks.

Determine which chunk(s) contain Certifications or Training information.

Chunk 1
Section : {chunk1.get("chunk_type", "")}
Content :
{chunk1.get("chunk_content", "")}

Chunk 2
Section : {chunk2.get("chunk_type", "")}
Content :
{chunk2.get("chunk_content", "")}

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

        print("[WARN] Judge parsing failed.")

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
# STEP 4 — LLM EXTRACTOR
# ─────────────────────────────────────────────

def llm_extract_certifications(
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

Resume:

{chr(10).join(context)}

Extract certifications and trainings.

Return ONLY JSON.

{{
  "certifications":[
    {{
      "name":null,
      "issuer":null,
      "date":null,
      "expiry_date":null,
      "credential_id":null
    }}
  ],
  "trainings":[
    {{
      "name":null,
      "provider":null,
      "type":null,
      "date":null
    }}
  ]
}}

Rules

- Don't invent information.
- Use null where missing.
- Return [] if nothing found.
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

        print("[WARN] Could not parse JSON.")

        return {
            "raw_extraction": raw
        }
    
# ─────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────

def extract_certifications(
    resume_id: str,
    db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """
    Full pipeline:

        1. Hybrid search
        2. LLM judge
        3. Fetch neighbouring chunks
        4. LLM extraction
        5. Save to DB
    """

    ensure_certifications_column(db_path)

    print(f"\n{'=' * 60}")
    print("[STEP 1] Hybrid Search")
    print(f"{'=' * 60}")

    results = hybrid_search(
        query="certifications training certified oracle aws professional",
        resume_id=resume_id,
        top_k=TOP_K,
        db_path=db_path,
    )

    if not results:
        print("[ERROR] No certification-related chunks found.")
        return {}

    chunk1 = results[0]
    chunk2 = results[1] if len(results) > 1 else results[0]

    print(
        f"Chunk1 -> id={chunk1['id']}  section={chunk1['chunk_type']}"
    )
    print(
        f"Chunk2 -> id={chunk2['id']}  section={chunk2['chunk_type']}"
    )

    print(f"\n{'=' * 60}")
    print("[STEP 2] LLM Judge")
    print(f"{'=' * 60}")

    relevant_ids = llm_judge_chunks(
        chunk1,
        chunk2,
    )

    print(f"Relevant Chunk IDs : {relevant_ids}")

    if not relevant_ids:
        print("[WARN] No relevant certification chunks.")
        return {}

    all_certifications = {
        "certifications": [],
        "trainings": [],
    }

    for target_id in relevant_ids:

        print(f"\n{'=' * 60}")
        print(f"[STEP 3] Window Fetch : chunk={target_id}")
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

        cert_data = llm_extract_certifications(
            window
        )

        print("\n[STEP 5] Saving to DB")

        save_certifications(
            target_id=target_id,
            certifications_json=cert_data,
            db_path=db_path,
        )

        # ----------------------------------------------------
        # Merge + Remove duplicates
        # ----------------------------------------------------

        for key in ("certifications", "trainings"):

            incoming = cert_data.get(key, [])

            if not isinstance(incoming, list):
                continue

            existing_names = {
                (item.get("name") or "").strip().lower()
                for item in all_certifications[key]
            }



            for item in incoming:

                name = (item.get("name") or "").strip().lower()

                if not name:
                    print("[SKIP] Empty certification name")
                    continue

                if name in existing_names:
                    print(f"[SKIP] Duplicate : {name}")
                    continue

                print(f"[ADD] {name}")

                all_certifications[key].append(item)
                existing_names.add(name)

    print(f"\n{'=' * 60}")
    print("[DONE] Certification Extraction")
    print(f"{'=' * 60}")

    print(
        json.dumps(
            all_certifications,
            indent=2,
            ensure_ascii=False,
        )
    )

    return all_certifications


