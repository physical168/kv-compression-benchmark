#!/usr/bin/env python3
"""Rebuild movie_kv_bundle.zip with POSIX paths (safe for Kaggle upload on Windows)."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parent / "movie_kv_bundle"
OUT_ZIP = Path(__file__).resolve().parent / "movie_kv_bundle.zip"

SOURCES = (
    "run_movie_kv_pregen.py",
    "kv_cache_text_qa_server_new.py",
    "text_kvpress_patch.py",
)


def sync_sources() -> None:
    parent = BUNDLE_DIR.parent
    for name in SOURCES:
        shutil.copy2(parent / name, BUNDLE_DIR / name)


def pack() -> None:
    sync_sources()
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(BUNDLE_DIR.rglob("*")):
            if path.is_file():
                rel = path.relative_to(BUNDLE_DIR).as_posix()
                zf.write(path, f"movie_kv_bundle/{rel}")
    bad = [n for n in zipfile.ZipFile(OUT_ZIP).namelist() if "\\" in n]
    if bad:
        raise SystemExit(f"zip has backslash paths: {bad[:3]}")
    print(f"Wrote {OUT_ZIP} ({len(zipfile.ZipFile(OUT_ZIP).namelist())} files)")


if __name__ == "__main__":
    pack()
