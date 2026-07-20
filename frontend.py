import os
import tempfile

import pandas as pd
import streamlit as st

from config import DB_PATH
from ingestion_pipeline import ingest_resume
from user_query_runner import handle_user_query


# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------

st.set_page_config(
    page_title="Resume RAG",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Resume RAG System")


# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.header("Resume Upload")

uploaded_file = st.sidebar.file_uploader(
    "Choose Resume (PDF)",
    type=["pdf"],
)

ingest_button = st.sidebar.button(
    "🚀 Ingest Resume",
    use_container_width=True,
)


# ======================================================
# INGESTION
# ======================================================

if ingest_button:

    if uploaded_file is None:

        st.sidebar.warning("Please select a PDF.")

    else:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as tmp:

            tmp.write(uploaded_file.read())

            pdf_path = tmp.name

        progress = st.sidebar.progress(
            0,
            text="Starting..."
        )

        progress.progress(
            20,
            text="Running pipeline..."
        )

        result = ingest_resume(
            pdf_path
        )

        progress.progress(
            100,
            text="Completed"
        )

        os.remove(pdf_path)

        if result["status"] == "success":

            st.sidebar.success(
                "Resume Ingested Successfully!"
            )

            st.sidebar.info(
                f"Resume ID\n\n{result['resume_id']}"
            )

        else:

            st.sidebar.error(
                f"Step : {result['step']}"
            )

            st.sidebar.error(
                result["message"]
            )


# ======================================================
# SEARCH
# ======================================================

st.header("🔍 Search Candidates")

query = st.text_input(
    "Enter your query",
    placeholder="Example : Java developers with Spring Boot and 5 years experience",
)

search_button = st.button(
    "Search",
    type="primary",
)


if search_button:

    if not query.strip():

        st.warning(
            "Please enter a search query."
        )

        st.stop()

    with st.spinner(
        "Searching candidates..."
    ):

        result = handle_user_query(
            user_query=query,
            db_path=DB_PATH,
        )

    if result["status"] == "success":

        st.success(
            f"{result['result_count']} candidate(s) found."
        )

        if result["results"]:

            df = pd.DataFrame(
                result["results"]
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

            with st.expander(
                "Generated SQL"
            ):

                st.code(
                    result["sql"],
                    language="sql",
                )

            with st.expander(
                "Extracted Filters"
            ):

                st.json(
                    result["filters"]
                )

        else:

            st.info(
                "No matching candidates found."
            )

    elif result["status"] == "not_relevant":

        st.warning(
            result["message"]
        )

    elif result["status"] == "unclear_query":

        st.warning(
            result["message"]
        )

    else:

        st.error(
            result.get(
                "message",
                "Unknown error."
            )
        )