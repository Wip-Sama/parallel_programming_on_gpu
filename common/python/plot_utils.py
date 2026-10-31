"""
plot_utils.py - Reusable Cairo Plotting Utilities for GPU Labs
Clean, publication-quality chart generation without external GUI dependencies.
"""

import os
import cairo
from typing import List, Tuple, Dict, Any, Optional

# Harmonious design palette
PALETTE = {
    "bg": (0.98, 0.98, 0.99),
    "card": (1.0, 1.0, 1.0),
    "border": (0.85, 0.88, 0.92),
    "grid": (0.88, 0.90, 0.93),
    "axis": (0.70, 0.73, 0.78),
    "text_dark": (0.12, 0.15, 0.20),
    "text_muted": (0.40, 0.45, 0.50),
    "primary": (0.20, 0.45, 0.85),    # Clean Blue
    "secondary": (0.10, 0.70, 0.45),  # Emerald Green
    "accent": (0.85, 0.55, 0.15),     # Warm Amber
    "purple": (0.55, 0.35, 0.80),     # Soft Purple
    "red": (0.85, 0.25, 0.25),        # Coral Red
    "cyan": (0.15, 0.65, 0.85),       # Ocean Cyan
    "gray": (0.50, 0.55, 0.62),       # Slate
}

def create_surface(width: int, height: int, bg_color: Tuple[float, float, float] = PALETTE["bg"]) -> Tuple[cairo.ImageSurface, cairo.Context]:
    """Creates a Cairo ImageSurface with high-DPI canvas painted with background color."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(*bg_color)
    ctx.paint()
    return surface, ctx

def draw_title(ctx: cairo.Context, title: str, subtitle: Optional[str] = None, x: float = 50.0, y: float = 45.0) -> None:
    """Draws main chart title and optional descriptive subtitle."""
    ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(20)
    ctx.set_source_rgb(*PALETTE["text_dark"])
    ctx.move_to(x, y)
    ctx.show_text(title)

    if subtitle:
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(12)
        ctx.set_source_rgb(*PALETTE["text_muted"])
        ctx.move_to(x, y + 25)
        ctx.show_text(subtitle)

def draw_axes(
    ctx: cairo.Context,
    left: float,
    top: float,
    right: float,
    bottom: float,
    max_y: float,
    ticks: int = 5,
    y_label_fmt: str = "{:.1f}",
    y_unit: str = "ms"
) -> None:
    """Draws gridlines, Y-axis labels, and axis borders."""
    plot_w = right - left
    plot_h = bottom - top

    # Axes lines
    ctx.set_source_rgb(*PALETTE["axis"])
    ctx.set_line_width(1.5)
    ctx.move_to(left, top)
    ctx.line_to(left, bottom)
    ctx.line_to(right, bottom)
    ctx.stroke()

    # Gridlines and Y labels
    ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(11)

    for i in range(ticks + 1):
        ratio = i / ticks
        y_val = max_y * ratio
        y_pos = bottom - ratio * plot_h

        # Gridline
        ctx.set_source_rgb(*PALETTE["grid"])
        ctx.set_line_width(1.0)
        ctx.move_to(left, y_pos)
        ctx.line_to(right, y_pos)
        ctx.stroke()

        # Label
        ctx.set_source_rgb(*PALETTE["text_muted"])
        label_text = f"{y_label_fmt.format(y_val)} {y_unit}".strip()
        ctx.move_to(left - 65, y_pos + 4)
        ctx.show_text(label_text)

def draw_legend(
    ctx: cairo.Context,
    start_x: float,
    y: float,
    items: List[Tuple[str, Tuple[float, float, float]]],
    item_spacing: float = 120.0
) -> None:
    """Draws horizontal legend items."""
    ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(11)

    for i, (label, color) in enumerate(items):
        item_x = start_x + i * item_spacing
        # Color badge
        ctx.set_source_rgb(*color)
        ctx.rectangle(item_x, y, 14, 12)
        ctx.fill()

        # Text
        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.move_to(item_x + 20, y + 10)
        ctx.show_text(label)

def draw_bar(
    ctx: cairo.Context,
    x: float,
    y: float,
    width: float,
    height: float,
    color: Tuple[float, float, float]
) -> None:
    """Draws a single colored bar."""
    ctx.set_source_rgb(*color)
    ctx.rectangle(x, y, width, height)
    ctx.fill()

def draw_grouped_bars(
    ctx: cairo.Context,
    left: float,
    top: float,
    right: float,
    bottom: float,
    max_y: float,
    categories: List[Dict[str, Any]],
    series_keys: List[str],
    colors: List[Tuple[float, float, float]],
    bar_width: float = 18.0,
    gap: float = 4.0
) -> None:
    """
    Renders grouped bars across multiple categories with value callouts and labels.
    """
    plot_w = right - left
    plot_h = bottom - top
    num_cats = len(categories)
    col_w = plot_w / num_cats
    num_series = len(series_keys)
    group_width = num_series * bar_width + (num_series - 1) * gap

    for c_idx, cat in enumerate(categories):
        cx = left + c_idx * col_w + col_w / 2.0
        start_x = cx - group_width / 2.0

        for s_idx, key in enumerate(series_keys):
            val = cat.get(key, 0.0)
            bx = start_x + s_idx * (bar_width + gap)
            bh = (val / max_y) * plot_h if max_y > 0 else 0
            by = bottom - bh

            draw_bar(ctx, bx, by, bar_width, bh, colors[s_idx])

            # Value label
            if bh > 15:
                ctx.set_source_rgb(*PALETTE["text_dark"])
                ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
                ctx.set_font_size(9)
                ctx.move_to(bx - 2, by - 5)
                ctx.show_text(f"{val:.1f}")

        # Category label below
        cat_label = str(cat.get("label", ""))
        lines = cat_label.split("\n")
        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(11)
        for l_idx, line in enumerate(lines):
            ctx.move_to(cx - 20, bottom + 18 + l_idx * 15)
            ctx.show_text(line)

def save_surface(surface: cairo.ImageSurface, output_path: str) -> None:
    """Writes surface to PNG and logs output."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    surface.write_to_png(output_path)
    print(f"[+] Saved plot: {output_path}")

