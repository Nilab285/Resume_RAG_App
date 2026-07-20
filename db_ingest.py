
import sys
from typing import List, Dict

from ingest import extract_document_structure
from config import DB_PATH
from db import get_connection, executemany


def init_db(db_path: str = DB_PATH):
    print("[1/4] Initializing database...")

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resume_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_id TEXT NOT NULL,
                candidate_name TEXT,
                personal_info TEXT,
                chunk_content TEXT,
                chunk_type TEXT,
                chunk_section TEXT,
                chunk_order INTEGER,
                skills TEXT,
                education TEXT,
                education_exp TEXT,
                work_details TEXT,
                work_exp TEXT
            )
        """)

        conn.commit()

    print("[1/4] Database ready.")


def build_resume_rows(pdf_path: str, resume_id: str) -> List[Dict]:
    print("[2/4] Extracting document structure (Docling is processing, please wait)...")

    chunks = extract_document_structure(pdf_path)

    print(f"[2/4] Extraction complete. Total chunks found: {len(chunks)}")

    if len(chunks) == 0:
        print("[WARNING] No chunks were extracted. Check if the PDF is readable or if ingest.py is returning data.")
        return []

    print("[DEBUG] Sample chunks:")
    for c in chunks[:3]:
        print("  ", c)

    rows = [
        {
            "resume_id": resume_id,
            "candidate_name": None,
            "personal_info": None,
            "chunk_content": chunk["chunk_content"],
            "chunk_type": chunk["chunk_type"],
            "chunk_section": chunk["chunk_section"],
            "chunk_order": chunk["chunk_order"],
            "skills": None,
            "education": None,
            "education_exp": None,
            "work_details": None,
            "work_exp": None,
        }
        for chunk in chunks
    ]

    return rows


def insert_resume_rows(rows: List[Dict], db_path: str = DB_PATH):
    if not rows:
        print("[3/4] No rows to insert. Skipping.")
        return

    print(f"[3/4] Inserting {len(rows)} rows into database...")

    rows_to_insert = [
        (
            row["resume_id"],
            row["candidate_name"],
            row["personal_info"],
            row["chunk_content"],
            row["chunk_type"],
            row["chunk_section"],
            row["chunk_order"],
            row["skills"],
            row["education"],
            row["education_exp"],
            row["work_details"],
            row["work_exp"],
        )
        for row in rows
    ]

    executemany(
        """
        INSERT INTO resume_chunks (
            resume_id,
            candidate_name,
            personal_info,
            chunk_content,
            chunk_type,
            chunk_section,
            chunk_order,
            skills,
            education,
            education_exp,
            work_details,
            work_exp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows_to_insert,
        db_path,
    )

    print("[3/4] Rows inserted successfully.")


def process_resume(pdf_path: str, resume_id: str, db_path: str = DB_PATH):
    init_db(db_path)
    rows = build_resume_rows(pdf_path, resume_id)
    insert_resume_rows(rows, db_path)

    print(f"[4/4] Done! Inserted {len(rows)} chunks for resume_id='{resume_id}'")

