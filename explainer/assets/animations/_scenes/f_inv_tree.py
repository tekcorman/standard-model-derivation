"""
F_inv(E) Cayley tree growing from the identity.

Sequence:
  1. Start with a single yellow root node labeled 'ε' (the identity).
  2. Three single-toggle children sprout outward (labeled a, b, c).
  3. From each gen-1 child, two gen-2 grandchildren sprout (one toggle
     cancels by involution, leaving 2 of 3).
  4. From each gen-2 node, two gen-3 great-grandchildren sprout.
  5. Caption: "The substrate is this graph extended indefinitely."

Render:
  manim -qm explainer/assets/animations/_scenes/f_inv_tree.py FInvTree
"""

from manim import (
    Scene, Circle, Line, Text, VGroup, Create, FadeIn, Write,
    AnimationGroup, Indicate,
    UP, DOWN, LEFT, RIGHT, ORIGIN, PI,
    WHITE, BLACK, GRAY, YELLOW, PURPLE, TEAL, ORANGE,
    config,
)
import numpy as np

config.background_color = "#0a0a14"


def node(label, position, color=PURPLE, radius=0.30):
    circ = Circle(radius=radius, color=color, fill_opacity=0.9, stroke_width=2.0)
    circ.move_to(position)
    text = Text(label, font_size=18, color=WHITE).move_to(position)
    return VGroup(circ, text)


def edge(p_from, p_to, color="#888888"):
    return Line(p_from, p_to, color=color, stroke_width=1.4)


class FInvTree(Scene):
    def construct(self):
        title = Text(
            "The substrate emerges: F_inv(E) Cayley tree from the identity",
            font_size=22, color=WHITE, weight="BOLD",
        ).to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1.0)

        # Layout: root at (-5, 0); 3 fan-out angles for gen-1; each gen-1
        # spawns 2 gen-2 nodes; each gen-2 spawns 2 gen-3 nodes.
        # Gen-1 angles (relative to +x): -45°, 0°, +45°
        # Gen-2 spread: ±20° around gen-1 direction
        # Gen-3 spread: ±12° around gen-2 direction
        root_pos = np.array([-5.5, 0, 0])

        # ── Gen 0: root ────────────────────────────────────────────────────
        root = node("ε", root_pos, color=YELLOW, radius=0.34)
        self.play(FadeIn(root), run_time=0.6)
        self.wait(0.3)

        # ── Gen 1 ──────────────────────────────────────────────────────────
        gen1_data = [
            ("a", -45 * PI / 180),
            ("b",   0 * PI / 180),
            ("c", +45 * PI / 180),
        ]
        gen1_dist = 2.0
        gen1_nodes = []
        gen1_edges = []
        for label, angle in gen1_data:
            pos = root_pos + gen1_dist * np.array([np.cos(angle), np.sin(angle), 0])
            n = node(label, pos, color=PURPLE, radius=0.28)
            e = edge(root_pos, pos)
            gen1_nodes.append(n)
            gen1_edges.append(e)

        self.play(
            AnimationGroup(*[Create(e) for e in gen1_edges], lag_ratio=0.15),
            AnimationGroup(*[FadeIn(n) for n in gen1_nodes], lag_ratio=0.15),
            run_time=1.5,
        )
        self.wait(0.5)

        # ── Gen 2: each gen-1 spawns 2 gen-2 nodes ─────────────────────────
        gen2_labels_per_gen1 = [
            ["ab", "ac"],  # from a
            ["ba", "bc"],  # from b
            ["ca", "cb"],  # from c
        ]
        gen2_dist = 1.6
        gen2_spread = 22 * PI / 180

        gen2_nodes = []
        gen2_edges = []
        for parent_idx, (label, parent_angle) in enumerate(gen1_data):
            parent_pos = root_pos + gen1_dist * np.array([np.cos(parent_angle), np.sin(parent_angle), 0])
            for child_idx, child_label in enumerate(gen2_labels_per_gen1[parent_idx]):
                offset = -gen2_spread if child_idx == 0 else +gen2_spread
                ang = parent_angle + offset
                cpos = parent_pos + gen2_dist * np.array([np.cos(ang), np.sin(ang), 0])
                n = node(child_label, cpos, color=TEAL, radius=0.22)
                e = edge(parent_pos, cpos)
                gen2_nodes.append(n)
                gen2_edges.append(e)

        self.play(
            AnimationGroup(*[Create(e) for e in gen2_edges], lag_ratio=0.07),
            AnimationGroup(*[FadeIn(n) for n in gen2_nodes], lag_ratio=0.07),
            run_time=1.6,
        )
        self.wait(0.5)

        # ── Gen 3: each gen-2 spawns 2 gen-3 nodes (no labels, just dots) ──
        gen3_dist = 1.1
        gen3_spread = 14 * PI / 180

        gen3_nodes = []
        gen3_edges = []
        for parent_idx, (label, parent_angle) in enumerate(gen1_data):
            parent_pos = root_pos + gen1_dist * np.array([np.cos(parent_angle), np.sin(parent_angle), 0])
            for child_idx in range(2):
                offset = -gen2_spread if child_idx == 0 else +gen2_spread
                ang = parent_angle + offset
                cpos = parent_pos + gen2_dist * np.array([np.cos(ang), np.sin(ang), 0])
                # Gen-3 children of this gen-2 node
                for gc_idx in range(2):
                    gc_offset = -gen3_spread if gc_idx == 0 else +gen3_spread
                    gc_ang = ang + gc_offset
                    gcpos = cpos + gen3_dist * np.array([np.cos(gc_ang), np.sin(gc_ang), 0])
                    n = Circle(radius=0.13, color=ORANGE, fill_opacity=0.85, stroke_width=1.2).move_to(gcpos)
                    e = edge(cpos, gcpos)
                    gen3_nodes.append(n)
                    gen3_edges.append(e)

        self.play(
            AnimationGroup(*[Create(e) for e in gen3_edges], lag_ratio=0.03),
            AnimationGroup(*[FadeIn(n) for n in gen3_nodes], lag_ratio=0.03),
            run_time=1.4,
        )

        # ── Closing caption ────────────────────────────────────────────────
        caption = Text(
            "Each step adds at most 2 children per node — one toggle cancels by involution.",
            font_size=18, color=GRAY, slant="ITALIC",
        ).to_edge(DOWN, buff=0.6)
        self.play(Write(caption), run_time=1.2)
        self.wait(2.0)
