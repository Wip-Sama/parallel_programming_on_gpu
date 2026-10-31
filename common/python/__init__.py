"""
GPU Programming Lab Common Python Utilities
"""

from .plot_utils import (
    create_surface,
    draw_title,
    draw_axes,
    draw_legend,
    save_surface,
    draw_bar,
    draw_grouped_bars,
    PALETTE,
)
from .verify_utils import (
    compute_file_md5,
    verify_image_files,
    verify_numerical as verify_arrays,
)
from .bench_utils import (
    aggregate_metrics,
    ensure_dir,
    load_csv,
    save_csv,
)

__all__ = [
    "create_surface",
    "draw_title",
    "draw_axes",
    "draw_legend",
    "save_surface",
    "draw_bar",
    "draw_grouped_bars",
    "PALETTE",
    "compute_file_md5",
    "verify_image_files",
    "verify_arrays",
    "aggregate_metrics",
    "ensure_dir",
    "load_csv",
    "save_csv",
]
