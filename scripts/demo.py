"""End-to-end smoke test: upload a PDF, wait for ingestion, ask a question.

Usage:
    python scripts/demo.py path/to/file.pdf "your question?"
"""

import sys
import time

import httpx

API = "http://localhost:8000/api"


def main(pdf_path: str, question: str) -> None:
    filename = pdf_path.replace("\\", "/").split("/")[-1]

    with httpx.Client(timeout=600) as client:
        # 1. Register the document, get a presigned upload URL.
        resp = client.post(
            f"{API}/documents",
            json={"filename": filename, "mime_type": "application/pdf"},
        )
        resp.raise_for_status()
        doc = resp.json()
        document_id = doc["document_id"]
        print(f"document_id = {document_id}")

        # 2. Upload the bytes straight to object storage.
        with open(pdf_path, "rb") as f:
            put = httpx.put(doc["upload_url"], content=f.read(), timeout=120)
            put.raise_for_status()
        print("uploaded file to storage")

        # 3. Kick off async ingestion.
        client.post(f"{API}/documents/{document_id}/ingest").raise_for_status()
        print("ingestion queued, polling...")

        # 4. Poll job status until done.
        while True:
            job = client.get(f"{API}/documents/{document_id}/job").json()
            print(f"  stage={job['stage']} status={job['status']}")
            if job["status"] in ("succeeded", "failed"):
                break
            time.sleep(2)

        if job["status"] == "failed":
            print(f"ingestion failed: {job['error']}")
            return

        # 5. Ask a question.
        ans = client.post(
            f"{API}/query",
            json={"question": question, "document_id": document_id},
        ).json()

        print("\n=== ANSWER ===")
        print(ans["answer"])
        print("\n=== CITATIONS ===")
        for c in ans["citations"]:
            print(f"[{c['marker']}] page {c['page_no']} (score {c['score']}): {c['snippet'][:120]}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
