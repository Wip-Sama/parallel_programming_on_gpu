"""
bench_utils.py - Reusable Benchmarking and Statistical Aggregation Utilities
"""

import os
import csv
import statistics
from typing import List, Dict, Any, Optional

def ensure_dir(path: str) -> str:
    """Ensures directory exists, creating parents if necessary."""
    os.makedirs(path, exist_ok=True)
    return path

def aggregate_metrics(values: List[float]) -> Dict[str, float]:
    """Computes mean, standard deviation, min, and max for a numeric list."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    
    n = len(values)
    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values) if n > 1 else 0.0
    return {
        "mean": mean_val,
        "std": std_val,
        "min": min(values),
        "max": max(values),
        "count": n,
    }

def load_csv(filepath: str) -> List[Dict[str, str]]:
    """Loads CSV rows as dictionaries."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def save_csv(filepath: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    """Writes dictionaries to a CSV file."""
    if not rows:
        return
    ensure_dir(os.path.dirname(os.path.abspath(filepath)))
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

