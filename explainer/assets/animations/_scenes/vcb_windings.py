"""
V_cb as a geometric series over girth-cycle windings.

Sequence:
  1. Title.
  2. A purple bar appears showing the n=1 contribution (2/3)^8.
  3. n=2 bar appears (much smaller).
  4. n=3 bar appears (tiny).
  5. n=4, ... ellipsis.
  6. The bars are added: a horizontal red line rises to show the cumulative
     sum = 256/6305 = 0.04060.
  7. A green line appears showing the PDG measured value 0.0406 ± 0.0009.
  8. They overlap — the framework matches at +0.00σ.

Render:
  manim -qm explainer/assets/animations/_scenes/vcb_windings.py VcbWindings
"""

from manim import (
    Scene, Rectangle, Line, Text, MathTex, VGroup, Create, FadeIn, Write,
    AnimationGroup, Indicate, Transform,
    UP, DOWN, LEFT, RIGHT, ORIGIN, PI,
    WHITE, BLACK, GRAY, RED, BLUE, GREEN, YELLOW, PURPLE, TEAL, ORANGE,
    config,
)
import numpy as np

config.background_color = "#0a0a14"

# Bar values (the actual geometric series terms)
TERMS = [
    (2 / 3) ** 8,    # n=1
    (2 / 3) ** 16,   # n=2
    (2 / 3) ** 24,   # n=3
    (2 / 3) ** 32,   # n=4
]
TOTAL = sum(TERMS) + (2 / 3) ** 40 + (2 / 3) ** 48  # truncated; exact = 256/6305
EXACT_TOTAL = 256 / 6305  # 0.04060...

# Visual scale: 1 unit of y-axis = 0.01 (so 0.039 ≈ 3.9 units tall)
Y_SCALE = 80


class VcbWindings(Scene):
    def construct(self):
        title = Text(
            "V_cb as a geometric series over girth-cycle windings",
            font_size=22, color=WHITE, weight="BOLD",
        ).to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1.0)

        # ── Axes ───────────────────────────────────────────────────────────
        x_axis = Line([-5.5, -2.5, 0], [5.5, -2.5, 0], color=WHITE, stroke_width=1.5)
        y_axis = Line([-5.5, -2.5, 0], [-5.5, 3, 0], color=WHITE, stroke_width=1.5)
        x_label = Text("winding number n", font_size=18, color=GRAY).next_to(x_axis, DOWN, buff=0.35)
        y_label = Text("amplitude contribution", font_size=18, color=GRAY).rotate(PI / 2).next_to(y_axis, LEFT, buff=0.15)
        self.play(Create(x_axis), Create(y_axis), FadeIn(x_label), FadeIn(y_label), run_time=0.8)

        # ── Bars: each (2/3)^{8n}, plotted at heights TERMS[i] * Y_SCALE ───
        x_positions = [-3.8, -1.8, 0.2, 2.0]
        bar_width = 1.0
        colors = [PURPLE, "#9b8cff", "#c8bdfb", "#dcd1ff"]
        # (n-label, value-label) pairs as LaTeX
        labels = [
            (r"n=1", r"(2/3)^{8} \approx 0.039"),
            (r"n=2", r"(2/3)^{16} \approx 0.0015"),
            (r"n=3", r"(2/3)^{24}"),
            (r"n=4", None),
        ]

        bars = []
        bar_labels = []
        for i, term in enumerate(TERMS):
            h = term * Y_SCALE
            cx = x_positions[i]
            bar = Rectangle(
                width=bar_width, height=h,
                fill_color=colors[i], fill_opacity=0.9,
                stroke_color=colors[i], stroke_width=1.5,
            ).move_to([cx, -2.5 + h / 2, 0])
            bars.append(bar)
            n_lbl, val_lbl = labels[i]
            parts = [MathTex(n_lbl, font_size=26, color=WHITE)]
            if val_lbl:
                parts.append(MathTex(val_lbl, font_size=20, color=WHITE))
            lbl = VGroup(*parts).arrange(DOWN, buff=0.12).next_to(bar, DOWN, buff=0.5)
            bar_labels.append(lbl)

        # Play bars in sequence with pauses
        for i in range(len(bars)):
            self.play(Create(bars[i]), FadeIn(bar_labels[i]), run_time=0.7)
            self.wait(0.25)

        # Ellipsis to indicate more windings
        ellipsis = Text("...  all retained simultaneously by the waterline", font_size=18, color=GRAY, slant="ITALIC")
        ellipsis.move_to([3.5, -1.0, 0])
        self.play(FadeIn(ellipsis), run_time=0.6)
        self.wait(0.4)

        # ── Sum line (framework total) ─────────────────────────────────────
        sum_y = -2.5 + EXACT_TOTAL * Y_SCALE
        sum_line = Line([-5.5, sum_y, 0], [5.5, sum_y, 0], color=RED, stroke_width=3.5)
        sum_label = MathTex(
            r"\sum_{n\geq 1}(2/3)^{8n} = \frac{256}{6305} = 0.04060",
            font_size=30, color=RED,
        ).next_to([5.3, sum_y, 0], LEFT, buff=0.2).shift(UP * 0.35)

        self.play(Create(sum_line), Write(sum_label), run_time=1.4)
        self.wait(0.5)

        # ── Measured value (overlap with framework) ────────────────────────
        pdg_label = Text(
            "measured: 0.0406 ± 0.0009    →    match within experimental error",
            font_size=18, color=GREEN, weight="BOLD",
        ).next_to(sum_label, DOWN, buff=0.3)
        self.play(FadeIn(pdg_label), Indicate(sum_line, color=GREEN, scale_factor=1.05), run_time=1.2)
        self.wait(2.0)
