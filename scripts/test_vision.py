"""Standalone Ollama vision smoke test (no app/Docker needed).

Sends an image to an Ollama vision model and prints what it sees. Use it to
verify a model's vision actually works on your machine.

Run from the host (Ollama on localhost:11434). Requires: pip install httpx

Examples:
    # Test the default model on a bundled sample image:
    python scripts/test_vision.py samples/red_square.png

    # Test a real photo fetched from the internet (no local file needed):
    python scripts/test_vision.py

    # Try a different model, or force CPU (avoids the 4GB-GPU CUDA crash):
    python scripts/test_vision.py samples/red_square.png --model gemma4:12b
    python scripts/test_vision.py samples/red_square.png --cpu
"""

import argparse
import base64
import time

import httpx

OLLAMA = "http://localhost:11434"


def load_image(path: str | None) -> bytes:
    if path:
        with open(path, "rb") as f:
            return f.read()
    # No path given: grab a real photograph so we test true vision, not a
    # synthetic flat-color shape.
    print("No image given; fetching a real photo from picsum.photos ...")
    return httpx.get("https://picsum.photos/seed/dog/384", follow_redirects=True, timeout=60).content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="Path to an image file (optional)")
    parser.add_argument("--model", default="gemma4:12b", help="Ollama model tag")
    parser.add_argument("--prompt", default="Describe this image in one sentence. What objects and colors do you see?")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (num_gpu=0)")
    parser.add_argument("--ollama", default=OLLAMA, help="Ollama base URL")
    args = parser.parse_args()

    data = load_image(args.image)
    print(f"image bytes: {len(data)}")
    b64 = base64.b64encode(data).decode()

    options = {"temperature": 0.0}
    if args.cpu:
        options["num_gpu"] = 0

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt, "images": [b64]}],
        "stream": False,
        "options": options,
    }

    print(f"model={args.model}  cpu={args.cpu}  -> POST {args.ollama}/api/chat")
    started = time.perf_counter()
    try:
        resp = httpx.post(f"{args.ollama}/api/chat", json=payload, timeout=600)
    except Exception as exc:  # noqa: BLE001
        print(f"REQUEST FAILED: {exc}")
        return
    elapsed = time.perf_counter() - started

    print(f"status={resp.status_code}  ({elapsed:.1f}s)")
    if resp.status_code == 200:
        print("\nMODEL SAW:\n" + resp.json()["message"]["content"])
    else:
        print("\nERROR BODY:\n" + resp.text[:800])


if __name__ == "__main__":
    main()
