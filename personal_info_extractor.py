import json
import re
from typing import Optional

from config import DB_PATH
from db import get_connection, execute
from llm_manager import llm_chat

MAX_EXTRA_ATTEMPTS = 2
INITIAL_WINDOW = 3


def extract_personal_info_agentic(
    resume_id: str,
    db_path: str = DB_PATH,
) -> Optional[dict]:
    """
    Agentic personal info extractor.

    Round 0 : chunks 1-3
    Round 1 : chunk 4
    Round 2 : chunk 5

    Stops early once all fields are found.
    """

    accumulated_context = []
    last_result = None

    # Initial window (chunks 1-3)
    initial_chunks = _fetch_chunks(
        resume_id=resume_id,
        db_path=db_path,
        from_order=1,
        limit=INITIAL_WINDOW,
    )

    accumulated_context.extend(initial_chunks)

    last_result = _call_llm(accumulated_context)

    if last_result and _all_fields_found(last_result):
        _write_to_db(
            resume_id=resume_id,
            info=last_result,
            feedback="personal info extracted successfully",
            db_path=db_path,
        )
        print("[INFO] All fields found in initial chunks 1-3 ✅")
        return last_result

    # Incrementally fetch one chunk at a time
    for attempt in range(1, MAX_EXTRA_ATTEMPTS + 1):

        next_order = INITIAL_WINDOW + attempt

        next_chunk = _fetch_chunks(
            resume_id=resume_id,
            db_path=db_path,
            from_order=next_order,
            limit=1,
        )

        if not next_chunk:
            print(f"[INFO] No chunk found at order {next_order}.")
            break

        accumulated_context.extend(next_chunk)

        last_result = _call_llm(accumulated_context)

        if last_result and _all_fields_found(last_result):

            _write_to_db(
                resume_id=resume_id,
                info=last_result,
                feedback="personal info extracted successfully",
                db_path=db_path,
            )

            print(f"[INFO] All fields found at chunk {next_order} ✅")
            return last_result

        print(f"[INFO] Attempt {attempt}: still missing fields.")

    null_fields = _get_null_fields(last_result) if last_result else {}

    _write_to_db(
        resume_id=resume_id,
        info=last_result or {},
        feedback=null_fields,
        db_path=db_path,
    )

    print(f"[WARN] Missing fields: {null_fields}")

    return last_result


# ---------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------

def _call_llm(
    chunks: list[dict],
) -> Optional[dict]:

    context = "\n".join(
        f"[Chunk {c['order']} | {c['type']}]: {c['content']}"
        for c in chunks
    )

    prompt = f"""
You are a precise resume parser.

Extract personal information from the resume chunks below.

Return ONLY valid JSON.

Use null for missing fields.

{{
  "name": null,
  "email": null,
  "phone": null,
  "location": null,
  "linkedin": null,
  "github": null,
  "portfolio": null
}}

Resume:

{context}
"""

    try:

        response = llm_chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()

        return _safe_parse_json(raw)

    except Exception as e:

        print(f"[ERROR] LLM call failed: {e}")

        return None


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def _all_fields_found(info: dict) -> bool:
    return all(v is not None for v in info.values())


def _get_null_fields(info: dict) -> dict:
    return {
        k: None
        for k, v in info.items()
        if v is None
    }


def _fetch_chunks(
    resume_id: str,
    db_path: str,
    from_order: int,
    limit: int,
) -> list[dict]:

    with get_connection(db_path) as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT chunk_order,
                   chunk_type,
                   chunk_content
            FROM resume_chunks
            WHERE resume_id = ?
              AND chunk_order >= ?
              AND chunk_order < ?
            ORDER BY chunk_order
            """,
            (
                resume_id,
                from_order,
                from_order + limit,
            ),
        )

        rows = cursor.fetchall()

    return [
        {
            "order": row[0],
            "type": row[1],
            "content": row[2],
        }
        for row in rows
    ]


def _safe_parse_json(raw: str) -> Optional[dict]:

    try:

        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw,
            flags=re.MULTILINE,
        ).strip()

        return json.loads(cleaned)

    except json.JSONDecodeError as e:

        print(f"[WARN] JSON parse failed: {e}")

        print(raw)

        return None


def _write_to_db(
    resume_id: str,
    info: dict,
    feedback: str | dict,
    db_path: str,
):

    execute(
        """
        UPDATE resume_chunks
        SET candidate_name = ?,
            personal_info = ?,
            personal_info_feedback = ?
        WHERE resume_id = ?
        """,
        (
            info.get("name"),
            json.dumps(info),
            feedback if isinstance(feedback, str)
                     else json.dumps(feedback),
            resume_id,
        ),
        db_path,
    )


