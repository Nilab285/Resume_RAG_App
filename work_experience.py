# =============================================================
# experience_extractor.py
#
# Flow:
#   1. Hybrid search  → top 5 chunks for work experience
#      + enrich       → re-fetch full DB row using id
#   2. LLM judge      → identify main section header
#   3. SQL fetch      → fetch complete work experience section
#   4. LLM extract    → structured JSON
#   5. Save           → work_details + total experience
# =============================================================

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from dateutil.relativedelta import relativedelta

from config import (
    DB_PATH,
    LLM_MODEL,
)

from db import (
    get_connection,
    execute,
)

from hybrid_search import hybrid_search
from llm_manager import llm_chat


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

TOP_K = 5

INITIAL_LIMIT = 5
EXTEND_BATCH = 5
MAX_EXTENSIONS = 3


# ---------------------------------------------------------
# SQLITE HELPERS
# ---------------------------------------------------------

def ensure_work_exp_column(
    db_path: str = DB_PATH,
) -> None:
    """
    Add work_exp column if it doesn't exist.
    """

    with get_connection(db_path) as conn:

        cur = conn.cursor()

        cur.execute(
            "PRAGMA table_info(resume_chunks)"
        )

        columns = [
            row["name"]
            for row in cur.fetchall()
        ]

        if "work_exp" not in columns:

            cur.execute(
                """
                ALTER TABLE resume_chunks
                ADD COLUMN work_exp TEXT
                """
            )

            conn.commit()

            print(
                "[INFO] 'work_exp' column added."
            )

        else:

            print(
                "[INFO] 'work_exp' column already exists."
            )


def save_experience(
    target_id: int,
    experience_json: Dict[str, Any],
    db_path: str = DB_PATH,
) -> None:
    """
    Save structured work experience JSON.
    """

    execute(
        """
        UPDATE resume_chunks
        SET work_details=?
        WHERE id=?
        """,
        (
            json.dumps(
                experience_json,
                ensure_ascii=False,
            ),
            target_id,
        ),
        db_path,
    )

    print(
        f"[OK] Experience saved → chunk id={target_id}"
    )


# ---------------------------------------------------------
# DB ENRICHMENT
# ---------------------------------------------------------

def enrich_chunks_with_db_metadata(
    chunks: List[Dict[str, Any]],
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Re-fetch complete DB rows using ids returned
    from hybrid search.

    Guarantees fields like:

    - chunk_order
    - chunk_section
    - chunk_type

    are always present.
    """

    if not chunks:
        return []

    ids = [
        chunk["id"]
        for chunk in chunks
        if chunk.get("id") is not None
    ]

    if not ids:

        print(
            "[WARN] Search results have no ids."
        )

        return chunks

    placeholders = ",".join(
        "?"
        for _ in ids
    )

    with get_connection(db_path) as conn:

        cur = conn.cursor()

        cur.execute(
            f"""
            SELECT
                id,
                resume_id,
                candidate_name,
                chunk_type,
                chunk_content,
                chunk_section,
                chunk_order
            FROM resume_chunks
            WHERE id IN ({placeholders})
            ORDER BY chunk_order
            """,
            ids,
        )

        db_rows = {
            row["id"]: dict(row)
            for row in cur.fetchall()
        }

    enriched = []

    for chunk in chunks:

        cid = chunk.get("id")

        if cid in db_rows:

            enriched.append(
                db_rows[cid]
            )

        else:

            enriched.append(chunk)

    print(
        f"[ENRICH] {len(enriched)} chunk(s) enriched."
    )

    return enriched

    # ---------------------------------------------------------
# TRUNCATION CHECKER
# ---------------------------------------------------------

def _llm_looks_truncated(
    chunks: List[Dict[str, Any]],
) -> bool:
    """
    Ask the LLM whether the currently fetched
    work experience section appears truncated.
    """

    if not chunks:
        return False

    context = "\n\n".join(
        [
            f"[chunk_order={chunk.get('chunk_order')} | "
            f"{chunk.get('chunk_type')}]\n"
            f"{chunk.get('chunk_content','').strip()}"
            for chunk in chunks
        ]
    )

    prompt = f"""
You are a resume parsing assistant.

Below are all chunks currently fetched for a Work Experience section.

{context}

Determine whether the section looks COMPLETE or TRUNCATED.

Return ONLY JSON.

{{
    "truncated": true
}}

or

{{
    "truncated": false
}}

No explanation.
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

    try:

        clean = re.sub(
            r"```(?:json)?|```",
            "",
            raw,
        ).strip()

        parsed = json.loads(clean)

        result = parsed.get(
            "truncated",
            False,
        )

        print(
            f"[TRUNCATION CHECK] {result}"
        )

        return result

    except Exception:

        print(
            "[WARN] Could not parse truncation response."
        )

        return False


# ---------------------------------------------------------
# HEADER CHECK
# ---------------------------------------------------------

def _llm_is_work_exp_header(
    header_content: str,
) -> bool:
    """
    Decide whether a SectionHeaderItem
    is still part of work experience.
    """

    prompt = f"""
You are a resume parser.

Header:

{header_content}

Reply ONLY JSON.

{{
    "is_work_experience": true
}}

or

{{
    "is_work_experience": false
}}
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

    try:

        clean = re.sub(
            r"```(?:json)?|```",
            "",
            raw,
        ).strip()

        parsed = json.loads(clean)

        return parsed.get(
            "is_work_experience",
            False,
        )

    except Exception:

        print(
            "[WARN] Header parse failed."
        )

        return False


# ---------------------------------------------------------
# SECTION CHECK
# ---------------------------------------------------------

def _llm_section_belongs_to_work_exp(
    chunk_section: str,
    work_exp_header: str,
) -> bool:

    prompt = f"""
We are extracting Work Experience.

Main section:

{work_exp_header}

Current chunk_section:

{chunk_section}

Reply ONLY JSON.

{{
    "belongs_to_work_experience": true
}}

or

{{
    "belongs_to_work_experience": false
}}
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

    try:

        clean = re.sub(
            r"```(?:json)?|```",
            "",
            raw,
        ).strip()

        parsed = json.loads(clean)

        return parsed.get(
            "belongs_to_work_experience",
            False,
        )

    except Exception:

        print(
            "[WARN] Section check parse failed."
        )

        return False
    


def llm_judge_main_section_header(
    chunks: List[Dict[str, Any]],
) -> Optional[str]:
    """
    From the enriched search results, ask the LLM to identify
    the ONE chunk that is the main work experience section header.
    Uses chunk_order to reject summary chunks.
    """

    chunk_lines = []

    for i, chunk in enumerate(chunks, start=1):

        chunk_lines.append(
            f"""Chunk {i}:
id            : {chunk.get('id')}
chunk_order   : {chunk.get('chunk_order')}
chunk_type    : {chunk.get('chunk_type')}
chunk_section : {chunk.get('chunk_section')}
chunk_content : {chunk.get('chunk_content','')[:200]}"""
        )

    chunks_text = "\n\n".join(chunk_lines)

    prompt = f"""
You are a resume parsing assistant.

Below are chunks retrieved from a resume database.

Each chunk contains:

- id
- chunk_order
- chunk_type
- chunk_section
- chunk_content

{chunks_text}

Your task:

Identify the ONE chunk that represents the MAIN Work Experience section.

Rules:

1. Must be SectionHeaderItem.
2. chunk_section should be NULL/None.
3. Valid examples:
- Work Experience
- Professional Experience
- Employment History
- Career History
4. Do NOT choose company names.
5. Do NOT choose job titles.
6. Ignore Summary/Objective sections.
7. Summary sections usually have very LOW chunk_order.
8. The correct header usually appears later.

Reply ONLY JSON.

{{
    "main_header":"Professional Experience"
}}

If none exists:

{{
    "main_header":null
}}
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

    print(f"\n[JUDGE RAW RESPONSE]\n{raw}\n")

    try:

        clean = re.sub(
            r"```(?:json)?|```",
            "",
            raw,
        ).strip()

        parsed = json.loads(clean)

        return parsed.get("main_header")

    except Exception:

        print("[WARN] Could not parse judge response.")

        return None
    
def fetch_all_chunks_under_section(
    section_header: str,
    resume_id: str,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:


    # ── Step A: find chunk_order of the section header
    with get_connection(db_path) as conn:       # ← auto-closes no matter what
        cur = conn.cursor()
        cur.execute("""
            SELECT chunk_order
            FROM   resume_chunks
            WHERE  resume_id     = ?
              AND  chunk_content = ?
              AND  chunk_type    = 'SectionHeaderItem'
            LIMIT 1
        """, (resume_id, section_header))
        row = cur.fetchone()

    if not row:
        print(f"[WARN] Could not find main header '{section_header}' in DB.")
        return []

    main_header_order = row["chunk_order"]
    print(f"  [STEP A] '{section_header}' found at chunk_order={main_header_order}")

    # ── Step B: fetch initial batch
    with get_connection(db_path) as conn:       # ← opens, reads, closes
        cur = conn.cursor()
        cur.execute("""
            SELECT id, resume_id, candidate_name,
                   chunk_type, chunk_content,
                   chunk_section, chunk_order
            FROM   resume_chunks
            WHERE  resume_id    = ?
              AND  chunk_order  > ?
            ORDER  BY chunk_order
            LIMIT  ?
        """, (resume_id, main_header_order, INITIAL_LIMIT))
        rows = [dict(r) for r in cur.fetchall()]

    print(f"  [STEP B] Fetched {len(rows)} chunks "
          f"(chunk_order {main_header_order} → "
          f"{rows[-1]['chunk_order'] if rows else '?'})")

    # ── Step C: truncation loop — each DB call gets its own short-lived connection
    extensions = 0

    while extensions < MAX_EXTENSIONS:

        if not _llm_looks_truncated(rows):
            print(f"  [TRUNCATION LOOP] LLM says complete ✓ "
                  f"— stopping at {len(rows)} total chunks.")
            break

        extensions += 1
        last_order = rows[-1]["chunk_order"]

        print(f"  [TRUNCATION LOOP] Extension {extensions}/{MAX_EXTENSIONS} "
              f"→ fetching {EXTEND_BATCH} more chunks after chunk_order={last_order}...")

        # ── each extension fetch opens and closes its own connection
        with get_connection(db_path) as conn:   # ← opens, reads, closes
            cur = conn.cursor()
            cur.execute("""
                SELECT id, resume_id, candidate_name,
                       chunk_type, chunk_content,
                       chunk_section, chunk_order
                FROM   resume_chunks
                WHERE  resume_id   = ?
                  AND  chunk_order > ?
                ORDER  BY chunk_order
                LIMIT  ?
            """, (resume_id, last_order, EXTEND_BATCH))
            candidate_chunks = [dict(r) for r in cur.fetchall()]

        # connection is already closed here ↑ — safe to save later

        if not candidate_chunks:
            print("  [TRUNCATION LOOP] No more chunks in DB — section complete.")
            break

        safe_chunks      = []
        crossed_boundary = False

        for chunk in candidate_chunks:
            cs       = chunk.get("chunk_section")
            ctype    = chunk.get("chunk_type")
            ccontent = chunk.get("chunk_content", "")

            if cs is None and ctype == "SectionHeaderItem":
                still_work_exp = _llm_is_work_exp_header(
                        header_content=ccontent,
                    )
                if not still_work_exp:
                    print(f"  [BOUNDARY] chunk_order={chunk['chunk_order']} "
                          f"chunk_section=NULL, type=SectionHeaderItem "
                          f"→ LLM says '{ccontent[:50]}' is NOT work exp — STOP.")
                    crossed_boundary = True
                    break

            elif cs is not None:
                still_work_exp = _llm_section_belongs_to_work_exp(
                    chunk_section=cs,
                    work_exp_header=section_header,
                )
                if not still_work_exp:
                    print(f"  [BOUNDARY] chunk_order={chunk['chunk_order']} "
                          f"chunk_section='{cs}' "
                          f"→ LLM says NOT work exp — STOP.")
                    crossed_boundary = True
                    break

            safe_chunks.append(chunk)
            print(f"  [KEEP] order={chunk['chunk_order']} "
                  f"section='{cs or 'NULL'}' "
                  f"| {ccontent[:60]}...")

        if safe_chunks:
            rows.extend(safe_chunks)
            print(f"  [TRUNCATION LOOP] Added {len(safe_chunks)} safe chunk(s) "
                  f"→ total now {len(rows)} chunks.")

        if crossed_boundary:
            print(f"  [TRUNCATION LOOP] Boundary crossed — exiting loop.")
            break

    else:
        print(f"  [TRUNCATION LOOP] Hard cap reached — "
              f"proceeding with {len(rows)} chunks.")

    # ── at this point ZERO connections are open → save_experience() is safe
    return rows

def llm_extract_experience(
    all_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Single LLM call over all fetched chunks to extract
    structured work experience JSON.
    """

    context_blocks = []

    for chunk in all_chunks:

        context_blocks.append(
            f"[chunk_order={chunk.get('chunk_order')} | "
            f"{chunk.get('chunk_type')}]\n"
            f"{chunk.get('chunk_content', '').strip()}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are an expert resume parser.

Below is the complete Work Experience section of a resume,
presented chunk by chunk in document order.

{context}

Extract ALL work experience entries.

Return ONLY valid JSON.

{{
  "experience": [
    {{
      "job_title": "Senior Java Developer",
      "company": "Orion Payments",
      "location": "San Francisco, CA",
      "start_date": "June 2020",
      "end_date": "Present",
      "responsibilities": [
        "Led migration...",
        "Designed APIs..."
      ],
      "technologies_used": [
        "Java",
        "Spring Boot",
        "Kafka"
      ]
    }}
  ]
}}

Rules:

- Return ONE object per job.
- If unavailable use null.
- responsibilities must be a list.
- technologies_used only if explicitly mentioned.
- Never invent information.
- Never hallucinate dates or companies.
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

    print(f"\n[EXTRACTOR RAW RESPONSE]\n{raw}\n")

    try:

        clean = re.sub(
            r"```(?:json)?|```",
            "",
            raw,
        ).strip()

        experience = json.loads(clean)

        print("[OK] JSON parsed successfully.")

        return experience

    except Exception as e:

        print(
            f"[WARN] Could not parse JSON ({e})"
        )

        return {
            "raw_extraction": raw
        }
    
def save_total_experience(
    target_id: int,
    total_exp: str,
    db_path: str = DB_PATH,
) -> None:
    """
    Save the calculated total work experience
    into the work_exp column.
    """

    execute(
        """
        UPDATE resume_chunks
        SET work_exp = ?
        WHERE id = ?
        """,
        (
            total_exp,
            target_id,
        ),
        db_path,
    )

    print(
        f"[OK] Total experience saved → '{total_exp}' into chunk id={target_id}"
    )

def _parse_date(date_str: str) -> datetime:
    """
    Parse a resume date string into a datetime object.
    """

    date_str = (date_str or "").strip()

    if not date_str:
        raise ValueError("Empty date")

    if date_str.lower() == "present":
        return datetime.today().replace(day=1)

    formats = (
        "%B %Y",   # June 2020
        "%b %Y",   # Jun 2020
        "%m/%Y",   # 06/2020
        "%Y-%m",   # 2020-06
    )

    for fmt in formats:

        try:
            return datetime.strptime(date_str, fmt)

        except ValueError:
            continue

    raise ValueError(f"Cannot parse date: {date_str}")

def _calculate_total_experience_programmatically(
    experience_json: Dict[str, Any],
) -> str:
    """
    Calculate total work experience while merging
    overlapping employment periods.
    """

    jobs = experience_json.get("experience", [])

    ranges = []

    for job in jobs:

        try:

            start = _parse_date(
                job.get("start_date", "")
            )

            end = _parse_date(
                job.get("end_date", "Present")
            )

            ranges.append((start, end))

        except ValueError as e:

            print(f"[WARN] {e}")

    if not ranges:
        return "Unknown"

    ranges.sort(key=lambda r: r[0])

    merged = [ranges[0]]

    for current_start, current_end in ranges[1:]:

        last_start, last_end = merged[-1]

        if current_start <= last_end:

            merged[-1] = (
                last_start,
                max(last_end, current_end),
            )

        else:

            merged.append(
                (
                    current_start,
                    current_end,
                )
            )

    total_months = 0

    for start, end in merged:

        diff = relativedelta(end, start)

        total_months += (
            diff.years * 12
            + diff.months
        )

    years = total_months // 12
    months = total_months % 12

    if years and months:
        return f"{years} years {months} months"

    if years:
        return f"{years} years"

    return f"{months} months"

def calculate_total_experience(
    experience_json: Dict[str, Any],
) -> str:
    """
    Calculate total experience using Python only.
    """

    print(
        "\n[EXPERIENCE CALC] Calculating total experience..."
    )

    try:

        return _calculate_total_experience_programmatically(
            experience_json
        )

    except Exception as e:

        print(
            f"[ERROR] Failed to calculate total experience: {e}"
        )

        return "Unknown"
    
# ---------------------------------------------------------
# MAIN ORCHESTRATOR
# ---------------------------------------------------------

def extract_experience(
    resume_id: str,
    db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """
    Complete work experience extraction pipeline.

    Flow:
        1. Hybrid Search
        2. DB Enrichment
        3. LLM identifies main work experience header
        4. Fetch complete section
        5. Extract structured experience
        6. Save work_details
        7. Calculate total experience
        8. Save total experience
    """

    ensure_work_exp_column(db_path)

    print(f"\n{'=' * 60}")
    print("[STEP 1] Hybrid Search")
    print(f"{'=' * 60}")

    results = hybrid_search(
        query=(
            "work experience job title "
            "responsibilities company developer engineer"
        ),
        resume_id=resume_id,
        top_k=TOP_K,
        db_path=db_path,
    )

    if not results:

        print("[ERROR] No work experience chunks found.")

        return {}

    results = enrich_chunks_with_db_metadata(
        results,
        db_path,
    )

    print(f"\nRetrieved {len(results)} enriched chunk(s).\n")

    for i, chunk in enumerate(results, start=1):

        print(
            f"{i}. "
            f"id={chunk.get('id')} | "
            f"order={chunk.get('chunk_order')} | "
            f"type={chunk.get('chunk_type')} | "
            f"section={chunk.get('chunk_section')}"
        )

    print(f"\n{'=' * 60}")
    print("[STEP 2] Identifying Work Experience Header")
    print(f"{'=' * 60}")

    main_header = llm_judge_main_section_header(
        results
    )

    if not main_header:

        print("[ERROR] Unable to identify work experience header.")

        return {}

    print(f"\nMain Header : {main_header}")

    print(f"\n{'=' * 60}")
    print("[STEP 3] Fetching Complete Section")
    print(f"{'=' * 60}")

    section_chunks = fetch_all_chunks_under_section(
        section_header=main_header,
        resume_id=resume_id,
        db_path=db_path,
    )

    if not section_chunks:

        print("[ERROR] Section fetch failed.")

        return {}

    print(f"\nFetched {len(section_chunks)} chunk(s).")

    print(f"\n{'=' * 60}")
    print("[STEP 4] Extracting Experience")
    print(f"{'=' * 60}")

    experience = llm_extract_experience(
        section_chunks
    )

    if not experience:

        print("[ERROR] Experience extraction failed.")

        return {}

    header_chunk_id = section_chunks[0]["id"]

    print(f"\n{'=' * 60}")
    print("[STEP 5] Saving Experience")
    print(f"{'=' * 60}")

    save_experience(
        target_id=header_chunk_id,
        experience_json=experience,
        db_path=db_path,
    )

    print(f"\n{'=' * 60}")
    print("[STEP 6] Calculating Total Experience")
    print(f"{'=' * 60}")

    total_exp = calculate_total_experience(
        experience
    )

    print(f"Total Experience : {total_exp}")

    print(f"\n{'=' * 60}")
    print("[STEP 7] Saving Total Experience")
    print(f"{'=' * 60}")

    save_total_experience(
        target_id=header_chunk_id,
        total_exp=total_exp,
        db_path=db_path,
    )

    print(f"\n{'=' * 60}")
    print("[DONE]")
    print(f"{'=' * 60}")

    print(
        json.dumps(
            experience,
            indent=2,
            ensure_ascii=False,
        )
    )

    return experience


