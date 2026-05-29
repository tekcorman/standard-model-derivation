"""
The 12-observable cross-validation — central resolvent, channels fire,
each lights up green against its measured value.

Sequence:
  1. Central purple node: the non-backtracking resolvent on srs.
  2. Twelve channel boxes around it (7 quark + 4 lepton + 1 cosmological).
  3. Each channel fires in sequence: an arrow flashes, the box lights up,
     a "match" tick mark appears.
  4. Final state: all 12 lit, with a "12 / 12 match within ~1σ" summary.

Render:
  manim -qm explainer/assets/animations/_scenes/twelve_observables.py TwelveObservables
"""

from manim import (
    Scene, Rectangle, Circle, Line, Text, MathTex, Arrow, VGroup, Create, FadeIn, Write,
    AnimationGroup, Indicate, Transform, FadeOut,
    UP, DOWN, LEFT, RIGHT, ORIGIN, PI,
    WHITE, BLACK, GRAY, RED, BLUE, GREEN, YELLOW, PURPLE, TEAL, ORANGE,
    config,
)
import numpy as np

config.background_color = "#0a0a14"

# 12 channels arrayed around the central resolvent.
# (latex_label, angle_degrees, sector_color)
CHANNELS = [
    # Top row: quark sector
    (r"y_t",            144, "#9b8cff"),
    (r"y_b",            116, "#9b8cff"),
    (r"V_{us}",          93, "#9b8cff"),
    (r"V_{cb}",          72, "#9b8cff"),
    (r"V_{ub}",          50, "#9b8cff"),
    (r"\delta_r",        28, "#9b8cff"),
    (r"\delta\rho",       0, "#9b8cff"),
    # Bottom row: lepton/PMNS + cosmological
    (r"y_\tau",         -30, "#56e0c8"),
    (r"\theta_{12}",    -60, "#56e0c8"),
    (r"\theta_{13}",    -90, "#56e0c8"),
    (r"\theta_{23}",   -120, "#56e0c8"),
    (r"A_s",           -160, "#ff9b56"),  # cosmological in orange
]


class TwelveObservables(Scene):
    def construct(self):
        title = Text(
            "Same resolvent, twelve channels, all match measurement — zero fitted constants",
            font_size=20, color=WHITE, weight="BOLD",
        ).to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=1.0)

        # ── Central resolvent node ─────────────────────────────────────────
        center = np.array([0, 0, 0])
        resolvent = Rectangle(
            width=3.8, height=1.4,
            fill_color=PURPLE, fill_opacity=0.9,
            stroke_color="#2d2a78", stroke_width=2,
        ).move_to(center)
        resolvent_text1 = MathTex(r"G = (I - u\,B)^{-1}", font_size=34, color=WHITE).move_to(center + UP * 0.25)
        resolvent_text2 = MathTex(r"\text{on srs,}\quad a = (2/3)^8", font_size=24, color=WHITE).move_to(center + DOWN * 0.28)
        resolvent_group = VGroup(resolvent, resolvent_text1, resolvent_text2)
        self.play(FadeIn(resolvent_group), run_time=0.8)
        self.wait(0.4)

        # ── Lay out all 12 channel boxes ───────────────────────────────────
        radius = 3.2
        channel_groups = []
        for label, angle_deg, color in CHANNELS:
            angle = angle_deg * PI / 180
            pos = np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
            box = Rectangle(
                width=1.2, height=0.5,
                fill_color=color, fill_opacity=0.35,  # start dim
                stroke_color=color, stroke_width=1.4,
            ).move_to(pos)
            text = MathTex(label, font_size=26, color=WHITE).move_to(pos)
            channel_groups.append((VGroup(box, text), pos, color))

        # Show all 12 in their dim state
        self.play(
            AnimationGroup(*[FadeIn(g[0]) for g in channel_groups], lag_ratio=0.04),
            run_time=1.2,
        )
        self.wait(0.4)

        # ── Each channel "fires" — arrow flashes, box lights up bright,
        #    a green tick appears next to it ─────────────────────────────────
        for grp, pos, color in channel_groups:
            box, text = grp[0], grp[1]
            arrow = Line(center, pos, color=color, stroke_width=2.5)
            # Brighten the box (raise fill opacity)
            bright_box = Rectangle(
                width=1.2, height=0.5,
                fill_color=color, fill_opacity=0.9,
                stroke_color=color, stroke_width=2.0,
            ).move_to(pos)
            # Green tick next to the box, on the side facing outward
            tick_offset = pos / np.linalg.norm(pos) * 0.55
            tick = Text("✓", font_size=26, color=GREEN, weight="BOLD").move_to(pos + tick_offset)

            self.play(
                Create(arrow),
                Transform(box, bright_box),
                run_time=0.18,
            )
            self.play(FadeIn(tick), FadeOut(arrow), run_time=0.18)

        # ── Final summary ──────────────────────────────────────────────────
        summary = Text(
            "12 / 12   —   all within ~1σ of measurement   —   zero fitted constants",
            font_size=22, color=GREEN, weight="BOLD",
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(summary), Indicate(resolvent_group, color=YELLOW, scale_factor=1.08), run_time=1.6)
        self.wait(2.0)
