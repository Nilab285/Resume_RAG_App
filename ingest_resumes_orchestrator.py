from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

from ingestion_pipeline import ingest_resume


def ingest_resumes_orch(
    pdf_paths: List[str],
    max_workers: int = 4,
) -> Dict:
    """
    Parallel Resume Ingestion Orchestrator.

    Parameters
    ----------
    pdf_paths : List[str]
        List of PDF paths.

    max_workers : int
        Number of resumes to process simultaneously.

    Returns
    -------
    {
        "total": int,
        "success": int,
        "failed": int,
        "results": [
            {
                "status": "...",
                "resume_id": "...",
                "step": "...",
                "message": "..."
            }
        ]
    }
    """

    if not pdf_paths:
        return {
            "total": 0,
            "success": 0,
            "failed": 0,
            "results": [],
        }

    print("=" * 70)
    print("PARALLEL RESUME INGESTION STARTED")
    print("=" * 70)
    print(f"Total resumes : {len(pdf_paths)}")
    print(f"Workers       : {max_workers}")
    print()

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        future_to_pdf = {
            executor.submit(
                ingest_resume,
                pdf_path
            ): pdf_path
            for pdf_path in pdf_paths
        }

        for future in as_completed(future_to_pdf):

            pdf_path = future_to_pdf[future]

            try:

                result = future.result()

            except Exception as e:

                result = {
                    "status": "failed",
                    "resume_id": None,
                    "step": "Unhandled Exception",
                    "message": str(e),
                }

            result["pdf_path"] = pdf_path
            results.append(result)

            status = result["status"].upper()

            print(
                f"[{status}] "
                f"{pdf_path}"
            )

    success_count = sum(
        1
        for r in results
        if r["status"] == "success"
    )

    failed_count = len(results) - success_count

    print()
    print("=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print(f"Total   : {len(results)}")
    print(f"Success : {success_count}")
    print(f"Failed  : {failed_count}")

    return {
        "total": len(results),
        "success": success_count,
        "failed": failed_count,
        "results": results,
    }


