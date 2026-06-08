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


def _ingest(client: httpx.Client, path: str) -> str:
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
    return document_id


def main(query: str, paths: list[str]) -> None:
    with httpx.Client(timeout=600) as client:
        print("Ingesting images...")
        for p in paths:
            _ingest(client, p)

        print(f"\nSearching images for: {query!r}")
        resp = client.post(f"{API}/search/images", json={"query": query}).json()
        for i, hit in enumerate(resp["hits"], start=1):
            print(f"  {i}. score={hit['score']} doc={hit['document_id']} url={'yes' if hit['image_url'] else 'no'}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2:])
