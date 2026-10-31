"""
verify_utils.py - Verification & Checksum Utilities in Python
Replaces complex in-C++ verification loops with fast, robust Python checks.
"""

import os
import hashlib
from typing import Dict, Any, List

def compute_file_md5(filepath: str) -> str:
    """Computes MD5 hash of a file on disk."""
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_bytes_md5(data: bytes) -> str:
    """Computes MD5 hash of in-memory bytes."""
    return hashlib.md5(data).hexdigest()

def verify_image_files(path_a: str, path_b: str) -> Dict[str, Any]:
    """
    Compares two generated image files (e.g., CPU vs GPU outputs).
    Checks file existence, exact MD5 checksum equality, and counts byte mismatches.
    """
    res = {
        "path_a": path_a,
        "path_b": path_b,
        "exists_a": os.path.exists(path_a),
        "exists_b": os.path.exists(path_b),
        "md5_a": "",
        "md5_b": "",
        "size_a": 0,
        "size_b": 0,
        "mismatches": 0,
        "matched": False,
        "error": None,
    }

    if not res["exists_a"]:
        res["error"] = f"File not found: {path_a}"
        return res
    if not res["exists_b"]:
        res["error"] = f"File not found: {path_b}"
        return res

    res["size_a"] = os.path.getsize(path_a)
    res["size_b"] = os.path.getsize(path_b)
    res["md5_a"] = compute_file_md5(path_a)
    res["md5_b"] = compute_file_md5(path_b)

    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        bytes_a = fa.read()
        bytes_b = fb.read()

    min_len = min(len(bytes_a), len(bytes_b))
    diff_count = sum(1 for i in range(min_len) if bytes_a[i] != bytes_b[i])
    diff_count += abs(len(bytes_a) - len(bytes_b))
    res["mismatches"] = diff_count

    res["matched"] = (res["md5_a"] == res["md5_b"]) and (diff_count == 0)
    return res

def verify_numerical(actual: List[float], expected: float, tol: float = 1e-5) -> Dict[str, Any]:
    """Verifies that all elements in actual match expected within tolerance."""
    mismatches = 0
    first_diff = None
    for idx, val in enumerate(actual):
        if abs(val - expected) > tol:
            mismatches += 1
            if first_diff is None:
                first_diff = (idx, val, expected)
    return {
        "total": len(actual),
        "mismatches": mismatches,
        "matched": (mismatches == 0),
        "first_mismatch": first_diff,
    }

