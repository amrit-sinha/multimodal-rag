"""Demo for multimodal (text -> image) search.

Uploads one or more image files, waits for ingestion, then runs a text query
and prints the images ranked by CLIP visual-semantic similarity.

Usage:
    python scripts/demo_image_search.py "a red square" samples/red_square.png samples/blue_circle.png
"""

import mimetypes
import sys
import time

import httpx

API = "http://localhost:8000/api"


def _ingest(client: httpx.Client, path: str) -> tuple[str, str]:
    filename = path.replace("\\", "/").split("/")[-1]
    mime = mimetypes.guess_type(filename)[0] or "image/png"

    doc = client.post(f"{API}/documents", json={"filename": filename, "mime_type": mime}).json()
    document_id = doc["document_id"]

    with open(path, "rb") as f:
        httpx.put(doc["upload_url"], content=f.read(), timeout=120).raise_for_status()

    client.post(f"{API}/documents/{document_id}/ingest").raise_for_status()
    while True:
        job = client.get(f"{API}/documents/{document_id}/job").json()
        if job["status"] in ("succeeded", "failed"):
            print(f"  {filename}: {job['status']}")
            break
        time.sleep(1)
    return document_id, filename


def main(query: str, paths: list[str]) -> None:
    with httpx.Client(timeout=600) as client:
        print("Ingesting images...")
        uploaded: dict[str, str] = {}
        for p in paths:
            doc_id, filename = _ingest(client, p)
            uploaded[doc_id] = filename

        # Search globally, then keep only images from *this* run so older test
        # uploads don't pollute the ranking shown in a demo.
        print(f"\nSearching images for: {query!r}")
        resp = client.post(
            f"{API}/search/images",
            json={"query": query, "top_k": max(20, len(paths) * 5)},
        )
        if resp.status_code >= 400:
            body = resp.json()
            print(f"search failed ({resp.status_code}): {body.get('detail', body)}")
            return

        hits = [
            h for h in resp.json()["hits"]
            if h["document_id"] in uploaded
        ]
        if not hits:
            print("  (no hits among images uploaded in this run)")
            return

        for i, hit in enumerate(hits, start=1):
            name = uploaded[hit["document_id"]]
            print(f"  {i}. {name}  score={hit['score']}")

        winner = hits[0]
        print(f"\nTop match: {uploaded[winner['document_id']]}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2:])
