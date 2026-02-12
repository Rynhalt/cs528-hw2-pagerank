#!/usr/bin/env python3
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

from google.cloud import storage

# Matches href="123.html" or href='123.html' with optional spaces; case-insensitive
LINK_RE = re.compile(r'href\s*=\s*["\']\s*(\d+)\.html\s*["\']', re.IGNORECASE)


@dataclass
class GraphCounts:
    n: int
    out_counts: List[int]
    in_counts: List[int]


def list_html_blobs(bucket_name: str) -> List[storage.Blob]:
    """List blobs under prefix and return those that look like numeric *.html files."""
    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name))

    html_blobs = []
    for b in blobs:
        name = b.name.split("/")[-1]
        if name.endswith(".html") and name[:-5].isdigit():
            html_blobs.append(b)

    if not html_blobs:
        raise RuntimeError(f"No numeric .html files found under gs://{bucket_name}")

    return html_blobs


def infer_n_from_blobs(blobs: Iterable[storage.Blob]) -> int:
    """Infer N as max(index)+1 from blob names like .../123.html."""
    max_idx = -1
    for b in blobs:
        name = b.name.split("/")[-1]
        idx = int(name[:-5])
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1


def extract_outgoing_targets(html_text: str) -> List[int]:
    """Return list of targets extracted from the HTML text."""
    return [int(m.group(1)) for m in LINK_RE.finditer(html_text)]


def load_link_counts_from_gcs(
    bucket_name: str,
    expected_n: Optional[int] = None,
    max_files: Optional[int] = None,
) -> GraphCounts:
    """
    Reads HTML pages from GCS, computes outgoing and incoming link counts.
    - expected_n: if provided, arrays are sized to this; out-of-range links ignored.
    - max_files: debug option to only read first K files.
    """
    blobs = list_html_blobs(bucket_name)

    if max_files is not None:
        blobs = blobs[:max_files]

    n = expected_n if expected_n is not None else infer_n_from_blobs(blobs)

    out_counts = [0] * n
    in_counts = [0] * n

    for b in blobs:
        fname = b.name.split("/")[-1]
        src = int(fname[:-5])
        if src < 0 or src >= n:
            continue

        raw = b.download_as_bytes()
        text = raw.decode("utf-8", errors="ignore")
        targets = extract_outgoing_targets(text)

        # Filter to [0, n)
        filtered = [t for t in targets if 0 <= t < n]

        out_counts[src] += len(filtered)
        for t in filtered:
            in_counts[t] += 1

    return GraphCounts(n=n, out_counts=out_counts, in_counts=in_counts)