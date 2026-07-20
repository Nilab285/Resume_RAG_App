import json
import re
from typing import Optional

from config import DB_PATH, LLM_MODEL
from db import get_connection
from llm_manager import llm_chat


# ─────────────────────────────────────────────
# ALLOWED COLUMNS
# ─────────────────────────────────────────────

SEARCHABLE_COLUMNS = {
    "skills_text": "Skills, programming languages, frameworks, tools",
    "work_details_text": "Job titles, companies, locations, responsibilities, technologies",
    "certifications_text": "Certifications, training programs, issuing bodies",
    "education": "Degrees, universities, fields of study",
    "education_exp": "Years of education (numeric)",
    "work_exp": "Years of total work experience (numeric)",
    "candidate_name": "Full name of candidate",
    "personal_info": "Location, city, email, phone number",
}

# ------------------------------------------------------------
# JSON Extractor
# ------------------------------------------------------------

def extract_json(raw: str) -> dict:
    """
    Extract the first valid JSON object from an LLM response.
    Handles markdown, explanations and extra text.
    """

    # Remove markdown fences
    clean = re.sub(
        r"```(?:json)?|```",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip()

    # Find first JSON object
    start = clean.find("{")

    if start == -1:
        raise ValueError("No JSON object found.")

    brace_count = 0

    for i in range(start, len(clean)):

        if clean[i] == "{":
            brace_count += 1

        elif clean[i] == "}":
            brace_count -= 1

            if brace_count == 0:

                json_text = clean[start:i + 1]

                print("\n[EXTRACTED JSON]")
                print(json_text)

                return json.loads(json_text)

    raise ValueError("Incomplete JSON object.")

# ─────────────────────────────────────────────
# SAMPLE DATA
# ─────────────────────────────────────────────

def get_sample_data(
    db_path: str = DB_PATH,
    limit: int = 2,
) -> list[dict]:
    """
    Fetch sample rows from resume_profiles
    for LLM context.
    """

    with get_connection(db_path) as conn:

        cur = conn.cursor()

        try:

            cur.execute(
                f"""
                SELECT
                    resume_id,
                    candidate_name,
                    skills_text,
                    work_details_text,
                    certifications_text,
                    personal_info,
                    work_exp,
                    education
                FROM resume_profiles
                LIMIT {limit}
                """
            )

            rows = [dict(r) for r in cur.fetchall()]

        except Exception as e:

            print(f"[WARN] Could not fetch sample data: {e}")

            rows = []

    return rows


# ─────────────────────────────────────────────
# STEP 1 — RELEVANCE CHECK
# ─────────────────────────────────────────────

def llm_check_relevance(
    user_query: str,
) -> dict:
    """
    Decide whether the query
    is related to resume search.
    """

    prompt = f"""
You are a resume database assistant.

User Query:

"{user_query}"

Determine whether this query is asking to
search/filter resumes.

Resume related examples:

- Find Python developers
- Show AWS certified candidates
- Candidates with 5 years experience
- Java backend engineers

NOT resume related:

- What is Docker?
- Explain Kubernetes
- Hello
- Tell me a joke

Return ONLY JSON.

{{
    "is_relevant": true,
    "reason": "short explanation"
}}
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

    print(f"\n[RELEVANCE RAW]\n{raw}")

    try:

        clean = re.sub(
            r"```(?:json)?|```",
            "",
            raw,
        ).strip()

        return json.loads(clean)

    except Exception:

        return {
            "is_relevant": True,
            "reason": "Parse fallback",
        }
    
# ─────────────────────────────────────────────
# STEP 2 — FILTER EXTRACTION
# ─────────────────────────────────────────────
    
def llm_extract_filters(
    user_query: str,
    db_path: str = DB_PATH,
) -> dict:
    """
    Convert a natural language resume search query into
    structured JSON filters.
    """

    prompt = f"""
You are an expert Resume Database Query Parser.

Your task is to convert a user's resume search request into
structured JSON that will later be converted into a SQL query.

==================================================
DATABASE SCHEMA
==================================================

candidate_name
    Contains the candidate's full name.

--------------------------------------------------

skills_text
    Contains technical skills including:
    - Programming languages
    - Frameworks
    - Libraries
    - Databases
    - Cloud platforms
    - DevOps tools
    - Operating systems
    - AI / ML technologies
    - Software tools

--------------------------------------------------

work_details_text
    Contains professional work information including:
    - Job titles
    - Company names
    - Responsibilities
    - Technologies used
    - Project descriptions
    - Business domains
    - Industry experience

--------------------------------------------------

certifications_text
    Contains:
    - Professional certifications
    - Training programs
    - Certification providers

--------------------------------------------------

education
    Contains:
    - Degree
    - University
    - College
    - Specialization
    - Field of study

--------------------------------------------------

work_exp
    Contains:
    Total years of professional experience.

--------------------------------------------------

education_exp
    Contains:
    Total years of formal education.

--------------------------------------------------

personal_info
    Contains personal information such as:
    - Location
    - City
    - State
    - Country
    - Email
    - Phone number

==================================================
USER QUERY
==================================================

"{user_query}"

==================================================
RULES
==================================================

1. Extract ONLY information explicitly mentioned in the user's query.

2. Never infer or assume additional search filters.

3. Never assume skills.

4. Never assume certifications.

5. Never assume education.

6. Never assume years of experience.

7. One query may generate multiple filters.

Example

User:
Resume of Data Scientist

Correct

work_details_text LIKE "Data Scientist"

Incorrect

skills_text LIKE "Python"

education LIKE "Data Science"

certifications_text LIKE "Data Science"

work_exp > 5

--------------------------------------------------

Operator Mapping

Text search
→ LIKE

Exact match
→ =

Greater than
→ >

Less than
→ <

Greater than or equal
→ >=

Less than or equal
→ <=

Use ONLY these operators.

==================================================
OUTPUT FORMAT
==================================================

Return ONLY ONE valid JSON object.

Do NOT explain your reasoning.

Do NOT include markdown.

Do NOT include code fences.

Do NOT include examples.

Do NOT include text before the JSON.

Do NOT include text after the JSON.

Your response MUST begin with {{

Your response MUST end with }}

JSON Schema

{{
    "filters": [
        {{
            "column": "skills_text",
            "operator": "LIKE",
            "value": "Python"
        }}
    ],
    "logic": "AND"
}}

If no filters can be extracted, return

{{
    "filters": [],
    "logic": "AND"
}}
"""

    print("\n" + "=" * 60)
    print("[FILTER PROMPT]")
    print("=" * 60)
    print(prompt)

    response = llm_chat(
        messages=[
            {
                "role": "system",
                "content": """
You are an expert Resume Search Query Parser.

Your only responsibility is converting resume search
queries into structured JSON.

Rules

- Return ONLY valid JSON.
- Never explain your reasoning.
- Never include markdown.
- Never include code fences.
- Never infer information.
- Never invent filters.
- Never invent operators.
- Always follow the schema supplied in the user prompt.
- Your response must begin with '{' and end with '}'.
"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,
    )

    raw = (response.choices[0].message.content or "").strip()

    print("\n" + "=" * 60)
    print("[FILTER RAW]")
    print("=" * 60)
    print(raw)

    try:

        parsed = extract_json(raw)

        print("\n[FILTER PARSED]")
        print(json.dumps(parsed, indent=2))

        return parsed

    except Exception as e:

        print(f"[WARN] Could not parse filter JSON: {e}")

        return {
            "filters": [],
            "logic": "AND",
        }
# ─────────────────────────────────────────────
# STEP 3 — SAFE SQL BUILDER
# ─────────────────────────────────────────────

def build_sql_query(
    filter_payload: dict,
) -> tuple[Optional[str], Optional[list]]:

    filters = filter_payload.get("filters", [])

    logic = (
        filter_payload.get("logic", "AND")
        .upper()
    )

    if logic not in ("AND", "OR"):
        logic = "AND"

    conditions = []
    params = []

    for f in filters:

        column = (
            f.get("column")
            or ""
        ).strip()

        operator = (
            f.get("operator")
            or "LIKE"
        ).upper().strip()

        value = f.get("value")

        if column not in SEARCHABLE_COLUMNS:

            print(
                f"[WARN] Invalid column : {column}"
            )

            continue

        if operator not in (
            "LIKE",
            "=",
            ">",
            "<",
            ">=",
            "<=",
        ):

            print(
                f"[WARN] Invalid operator : {operator}"
            )

            continue

        if value is None or str(value).strip() == "":

            continue

        if operator == "LIKE":

            conditions.append(
                f"{column} LIKE ?"
            )

            params.append(
                f"%{value}%"
            )

        else:

            conditions.append(
                f"{column} {operator} ?"
            )

            params.append(value)

    if not conditions:

        return None, None

    where_clause = f" {logic} ".join(
        conditions
    )

    sql = f"""
    SELECT
        resume_id,
        candidate_name,
        work_exp,
        education_exp,
        skills_text,
        certifications_text,
        personal_info

    FROM resume_profiles

    WHERE {where_clause}

    ORDER BY candidate_name
    """

    return sql.strip(), params


# ─────────────────────────────────────────────
# STEP 4 — EXECUTE SQL
# ─────────────────────────────────────────────

def execute_query(
    sql: str,
    params: list,
    db_path: str = DB_PATH,
) -> list[dict]:

    print(f"\n[SQL]\n{sql}")
    print(f"[PARAMS] {params}")

    with get_connection(db_path) as conn:

        cur = conn.cursor()

        try:

            cur.execute(
                sql,
                params,
            )

            rows = [
                dict(r)
                for r in cur.fetchall()
            ]

        except Exception as e:

            print(
                f"[ERROR] {e}"
            )

            rows = []

    return rows

# ─────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────

def handle_user_query(
    user_query: str,
    db_path: str = DB_PATH,
) -> dict:
    """
    Natural language query → matching resumes.
    """

    print(f"\n{'=' * 60}")
    print(f"[QUERY] {user_query}")
    print(f"{'=' * 60}")

    # --------------------------------------------------
    # STEP 1 : Relevance Check
    # --------------------------------------------------

    relevance = llm_check_relevance(
        user_query=user_query,
    )

    print(
        f"\n[RELEVANCE] "
        f"{relevance.get('is_relevant')} "
        f"| {relevance.get('reason')}"
    )

    if not relevance.get("is_relevant", True):

        return {
            "status": "not_relevant",
            "message": (
                "Your query doesn't appear to be related to "
                "resume or candidate search.\n\n"
                "Examples:\n"
                "• Find Python developers\n"
                "• Show AWS certified candidates\n"
                "• Find Java developers with 5 years experience"
            ),
            "results": [],
        }

    # --------------------------------------------------
    # STEP 2 : Extract Filters
    # --------------------------------------------------

    filter_payload = llm_extract_filters(
        user_query=user_query,
        db_path=db_path,
    )

    print(
        "\n[FILTERS]"
    )

    print(
        json.dumps(
            filter_payload,
            indent=2,
        )
    )

    # --------------------------------------------------
    # STEP 3 : Build SQL
    # --------------------------------------------------

    sql, params = build_sql_query(
        filter_payload
    )

    if sql is None:

        return {
            "status": "unclear_query",
            "message": (
                "I couldn't understand the search criteria.\n\n"
                "Examples:\n"
                "• Find Java developers\n"
                "• Candidates with AWS certification\n"
                "• Python developers with Docker"
            ),
            "results": [],
        }

    # --------------------------------------------------
    # STEP 4 : Execute SQL
    # --------------------------------------------------

    rows = execute_query(
        sql=sql,
        params=params,
        db_path=db_path,
    )

    print(
        f"\n[RESULTS] {len(rows)} candidate(s)"
    )

    for row in rows:

        print(
            f" -> {row['resume_id']} | "
            f"{row['candidate_name']} | "
            f"{row['work_exp']} years"
        )

    return {
        "status": "success",
        "query": user_query,
        "filters": filter_payload,
        "sql": sql,
        "params": params,
        "result_count": len(rows),
        "results": rows,
    }

