"""
MDL waterline animation for explainer/story/04-recurrence-and-the-mdl-waterline.md.

Sequence:
  1. Many candidate compressions appear as colored bars at various heights.
  2. A horizontal red line (the waterline = L_raw) sweeps in from the left.
  3. Bars below the line collapse to grey "noise"; bars above stay colored
     and persist together (the "plurally retained" set).
  4. The dominant bar lights up (purple) and rises a tick to mark it as
     the MDL minimum within the retained set.

Render:
  manim -qm explainer/assets/animations/_scenes/mdl_waterline.py MDLWaterline
"""

from manim import (
    Scene, Rectangle, Line, Text, MathTex, VGroup, Create, Write, FadeIn,
    FadeOut, MoveAlongPath, AnimationGroup, Transform, Indicate, Group,
    UP, DOWN, LEFT, RIGHT, ORIGIN, DEGREES, PI,
    WHITE, BLACK, GRAY, GREY, RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, TEAL,
    config,
)
import numpy as np

# Match the explainer's dark theme
config.background_color = "#0a0a14"


# Bar heights (in y-axis units): mix of above and below the L_raw line.
# The bars are designed so:
#   - bar 0 = "srs" (dominant) at the top
#   - bar 1 = "srs*" (mirror chirality) just below
#   - bars 2-8 above the line at decreasing heights (subdominant retained)
#   - bars 9-12 below the line (discarded as noise)
BAR_DATA = [
    ("srs",      4.4, PURPLE),   # dominant
    ("srs*",     4.0, TEAL),     # mirror chirality, retained
    ("",         3.4, TEAL),
    ("",         2.8, TEAL),
    ("",         2.4, TEAL),
    ("",         2.1, TEAL),
    ("",         1.85, TEAL),
    ("",         1.65, TEAL),
    ("",         1.52, TEAL),    # just barely above
    ("",         1.30, GREY),    # below
    ("",         1.05, GREY),
    ("",         0.80, GREY),
    ("",         0.55, GREY),
]
L_RAW = 1.5  # waterline height (in same units as bar heights)


class MDLWaterline(Scene):
    def construct(self):
        # ── Axes scaffolding ───────────────────────────────────────────────
        # x-axis sits at y = -3.0; bars rise from there.
        x_axis = Line([-6, -3, 0], [6, -3, 0], color=WHITE, stroke_width=1.5)
        y_axis = Line([-6, -3, 0], [-6, 3.5, 0], color=WHITE, stroke_width=1.5)

        y_label = Text("compression savings →", font_size=18, color=GRAY)
        y_label.rotate(PI / 2).next_to(y_axis, LEFT, buff=0.15)

        x_label = Text("candidate compressions", font_size=18, color=GRAY)
        x_label.next_to(x_axis, DOWN, buff=0.45)

        title = Text(
            "The MDL waterline — every above-threshold compression is retained",
            font_size=24, color=WHITE, weight="BOLD",
        ).to_edge(UP, buff=0.3)

        self.play(Write(title), run_time=1.2)
        self.play(Create(x_axis), Create(y_axis), FadeIn(x_label), FadeIn(y_label), run_time=0.8)

        # ── Bars: built initially as colored rectangles all rising together ─
        n_bars = len(BAR_DATA)
        x_start = -5.3
        x_end = 5.3
        bar_width = (x_end - x_start) / n_bars * 0.7
        x_gap = (x_end - x_start) / n_bars

        bars = []
        labels = []
        for i, (name, height, color) in enumerate(BAR_DATA):
            cx = x_start + (i + 0.5) * x_gap
            base_y = -3
            top_y = base_y + height
            bar = Rectangle(
                width=bar_width,
                height=height,
                fill_color=color,
                fill_opacity=0.9,
                stroke_color=color,
                stroke_width=1.2,
            ).move_to([cx, (base_y + top_y) / 2, 0])
            bars.append(bar)

            if name:
                label = Text(name, font_size=16, color=WHITE).next_to(bar, UP, buff=0.12)
                labels.append(label)

        # Build all bars together
        self.play(
            AnimationGroup(*[Create(b) for b in bars], lag_ratio=0.06),
            run_time=1.8,
        )
        if labels:
            self.play(*[FadeIn(l) for l in labels], run_time=0.6)
        self.wait(0.5)

        # ── Waterline sweeps in from the left ─────────────────────────────
        waterline_y = -3 + L_RAW
        waterline = Line(
            [-6, waterline_y, 0],
            [-6, waterline_y, 0],
            color=RED,
            stroke_width=4,
        )
        waterline_label = MathTex(r"L_{\mathrm{raw}}", color=RED, font_size=36)
        waterline_label.next_to([6, waterline_y, 0], RIGHT, buff=0.15)

        self.play(Create(waterline), run_time=0.4)
        self.play(
            Transform(waterline, Line([-6, waterline_y, 0], [6, waterline_y, 0],
                                       color=RED, stroke_width=4)),
            run_time=1.4,
        )
        self.play(Write(waterline_label), run_time=0.5)
        self.wait(0.5)

        # ── Bars below the line collapse to "noise" (grey, shorter) ───────
        # We already coloured them GREY; now shrink them dramatically to
        # emphasize discard, and dim them.
        below_indices = [i for i, (_, h, _) in enumerate(BAR_DATA) if h < L_RAW]
        below_anims = []
        for i in below_indices:
            old = bars[i]
            new = Rectangle(
                width=bar_width * 0.6,
                height=0.12,
                fill_color="#444444",
                fill_opacity=0.7,
                stroke_color="#666666",
                stroke_width=0.8,
            ).move_to([old.get_center()[0], -2.94, 0])
            below_anims.append(Transform(old, new))
        self.play(*below_anims, run_time=1.0)

        # ── Annotation: "retained simultaneously" + "discarded" ───────────
        retained_text = Text(
            "retained (above the line) — plurally, simultaneously",
            font_size=18, color=TEAL, slant="ITALIC",
        )
        retained_text.move_to([-1.9, 3.0, 0])
        discarded_text = Text(
            "discarded as noise",
            font_size=18, color=GRAY, slant="ITALIC",
        )
        discarded_text.move_to([3.0, -2.3, 0])
        self.play(FadeIn(retained_text), FadeIn(discarded_text), run_time=0.7)
        self.wait(0.4)

        # ── Highlight the dominant (purple) — MDL minimum within retained ─
        self.play(Indicate(bars[0], color=YELLOW, scale_factor=1.15), run_time=1.2)
        dominant_label = Text(
            "dominant compression (unique within the retained set)",
            font_size=18, color=YELLOW, slant="ITALIC",
        ).move_to([-2.0, -0.2, 0])
        # Arrow-style emphasis instead of new mobject
        self.play(FadeIn(dominant_label), run_time=0.6)
        self.wait(2.0)
