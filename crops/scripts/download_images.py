"""Download observation images listed in a CSV's `image_url` column."""

import argparse
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILE = "classifier/data/chickpeas/observations-749539.csv"
DEFAULT_OUT = "classifier/img"


def repo_path(relative: str | Path) -> Path:
    return REPO_ROOT / relative


def download_images(csv_path: Path, out_dir: Path, limit: int | None = None) -> None:
    df = pd.read_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    urls = df[["id", "image_url"]].dropna(subset=["image_url"])
    if limit is not None:
        urls = urls.head(limit)
    total = len(urls)
    print(f"Downloading up to {total} image(s) from {csv_path.name}")

    for i, (obs_id, url) in enumerate(zip(urls["id"], urls["image_url"]), start=1):
        ext = Path(urlparse(url).path).suffix or ".jpg"
        dest = out_dir / f"{obs_id}{ext}"
        if dest.exists():
            print(f"[{i}/{total}] skip (exists) {dest.name}")
            continue
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print(f"[{i}/{total}] saved {dest.name}")
        except requests.RequestException as exc:
            print(f"[{i}/{total}] failed {url}: {exc}")

    print(f"Done. Images in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        default=DEFAULT_FILE,
        help="Observations CSV path relative to repo root",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of images to download (default: all rows with image_url)",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="Output directory for images, relative to repo root",
    )
    args = parser.parse_args()
    download_images(repo_path(args.file), repo_path(args.out), args.limit)


if __name__ == "__main__":
    main()
