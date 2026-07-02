#!/usr/bin/env python3
"""Dark/visible bit-budget analysis: compute the Φ-weight ratio of
canonically-excluded substrate compressions to total compressions.

Hypothesis: substrate compressions split into
    visible := compressions reachable by canonical categorical operations
              (abelianization, conjugation, ...)
    dark    := compressions A2 admits but which are not canonical
              (arbitrary relator quotients that over-collapse the substrate)

The framework's dark-extraction map gives a specific coefficient (5/12 in the
Feshbach amplitude). If the categorical-walker dark/visible ratio matches
5/12 (or another framework constant), that's quantitative evidence the
walker's "over-collapsing moves" structurally ARE the dark sector.

Compute at multiple states (F(E) initial, F_inv(E) post-involutivity, lean
cascade halt) and tabulate.
"""
from __future__ import annotations
import os, sys, math
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from substrate_state import (  # noqa: E402
    initial_state, op_0_4_involutive, op_1_8_conjugation,
    op_1_10_abelianization,
)
from categorical_walker import (  # noqa: E402
    enumerate_minimal_schema_epis, evaluate_conjugation, schema_L,
)

# ---------------- Canonicity classification ----------------

# Templates classified as canonical group-theoretic operations:
#   (0,0)         = involutivity        — defining relation of F_inv(E)
#   (0,1,0,1)     = commutator [a,b]    — gives abelianization under involutivity
#   conjugation   = quotient by inner-automorphism action — canonical move
# All other relator schemas are non-canonical (A2-admissible but not corresponding
# to a named group-theoretic construction).
CANONICAL_TEMPLATES = {
    (0, 0),         # involutivity → F_inv(E)
    (0, 1, 0, 1),   # commutator → abelianization (under involutivity)
}

CANONICAL_NON_RELATOR_MOVES = {'CONJ'}   # conjugation by inner automorphism

def is_canonical(template) -> bool:
    if template == 'CONJ':
        return True
    return tuple(template) in CANONICAL_TEMPLATES

# ---------------- Analysis at a state ----------------

def analyze_state(state, label: str, max_relator_length: int = 4):
    """Enumerate all A2-admissible compressions, classify, sum Φ."""
    print(f"\n{'='*100}")
    print(f"State: {label}")
    print(f"  classes:       {state.n_classes}")
    print(f"  refinements:   {state.refinements}")
    print(f"{'='*100}")

    # Schema-relator candidates
    schema_cands = enumerate_minimal_schema_epis(state, max_relator_length)
    # Conjugation (only if not already imposed)
    cands = list(schema_cands)
    if 'conjugation' not in state.refinements:
        c = evaluate_conjugation(state)
        if c is not None:
            cands.append(c)

    # Sort by Φ - L (best first)
    cands.sort(key=lambda c: -c.get('net', c['Phi'] - c['L']))

    visible_phi = 0.0
    dark_phi = 0.0
    visible_count = 0
    dark_count = 0

    print(f"\n  All candidate compressions (length ≤ {max_relator_length}):")
    print(f"  {'kind':<8}{'template':<22}{'Φ':>8}{'L':>9}{'Δ':>9}{'classes':>15}{'class':>10}")
    print('  ' + '-' * 86)
    for c in cands:
        tmpl = c.get('template')
        if tmpl is None:
            tmpl_disp = 'CONJ'
            tmpl_key = 'CONJ'
        else:
            tmpl_disp = str(tmpl)
            tmpl_key = tmpl

        canonical = is_canonical(tmpl_key)
        cls = 'visible' if canonical else 'dark'
        kind = c.get('kind', 'schema')

        if canonical:
            visible_phi += c['Phi']
            visible_count += 1
        else:
            dark_phi += c['Phi']
            dark_count += 1

        print(f"  {kind:<8}{tmpl_disp:<22}{c['Phi']:>8.3f}{c['L']:>9.3f}"
              f"{c['Phi']-c['L']:>+9.3f}{c['n_before']:>7} →{c['n_after']:>5}{cls:>10}")

    total = visible_phi + dark_phi
    print(f"\n  Summary:")
    print(f"    Visible (canonical) Φ:    {visible_phi:>8.3f} bits  ({visible_count} moves)")
    print(f"    Dark (non-canonical) Φ:   {dark_phi:>8.3f} bits  ({dark_count} moves)")
    print(f"    Total:                    {total:>8.3f} bits  ({len(cands)} moves)")
    if total > 0:
        print(f"    Visible / total:          {visible_phi/total:>8.4f}")
        print(f"    Dark / total:             {dark_phi/total:>8.4f}")
        print(f"    Dark / visible:           {dark_phi/visible_phi if visible_phi > 0 else float('inf'):>8.4f}")
    return {
        'visible_phi': visible_phi, 'dark_phi': dark_phi, 'total_phi': total,
        'visible_count': visible_count, 'dark_count': dark_count,
    }

# ---------------- Framework constants for comparison ----------------

FRAMEWORK_CONSTANTS = {
    '5/12 (dark Feshbach amplitude)': 5/12,
    '7/12 (1 - dark)':                 7/12,
    '5/7 (dark/visible if 5/12)':      5/7,
    'k*-1 / k* = 2/3 (NB survival)':   2/3,
    '1/3 (1 - 2/3)':                   1/3,
    '1/2':                              1/2,
    'α_1_full = 256/6305':             256/6305,
    'sin²θ_W = 3/8':                   3/8,
    '5/8 (1 - sin²θ_W)':               5/8,
    '0.8488 (Ω_DM/Ω_m)':               0.8488,
}

def compare_to_constants(value: float, label: str):
    print(f"\n  Comparison of {label} = {value:.4f} to framework constants:")
    print(f"    {'constant':<40}{'value':>10}{'|Δ|':>10}{'rel':>10}")
    print('    ' + '-' * 70)
    matches = sorted(FRAMEWORK_CONSTANTS.items(), key=lambda x: abs(x[1] - value))
    for name, v in matches[:5]:
        diff = abs(v - value)
        rel = diff / max(abs(v), 1e-9)
        flag = '★' if rel < 0.05 else ''
        print(f"    {name:<40}{v:>10.4f}{diff:>10.4f}{rel:>9.2%} {flag}")

# ---------------- Main ----------------

if __name__ == '__main__':
    print("=" * 100)
    print("DARK/VISIBLE BIT-BUDGET ANALYSIS")
    print("=" * 100)
    print("""
Hypothesis: substrate compressions split into
  - VISIBLE: canonical categorical operations (abelianization, conjugation, ...)
  - DARK: A2-admissible non-canonical relator quotients

Test: compute Φ-weight ratio at multiple substrate states and compare to 5/12.
""")

    E, n_max = 6, 4
    print(f"Configuration: E={E}, n_max={n_max}")

    # State 1: F(E) initial
    s_init = initial_state(E=E, n_max=n_max)
    r1 = analyze_state(s_init, 'F(E) initial — no relations imposed')

    # State 2: After involutivity (F_inv(E))
    s_inv = op_0_4_involutive(s_init)
    r2 = analyze_state(s_inv, 'F_inv(E) — after involutivity')

    # State 3: After conjugation
    s_conj = op_1_8_conjugation(s_inv)
    r3 = analyze_state(s_conj, 'F_inv(E)/conj — after conjugation')

    # State 4: Lean cascade halt (after abelianization)
    s_lean = op_1_10_abelianization(s_conj)
    r4 = analyze_state(s_lean, 'Lean cascade halt — after abelianization')

    # ============ Headline ============
    print(f"\n{'='*100}")
    print("HEADLINE: Dark/visible Φ ratio at each substrate state")
    print(f"{'='*100}")
    print(f"\n  {'state':<45}{'visible Φ':>12}{'dark Φ':>10}{'dark/total':>13}{'dark/visible':>15}")
    print('  ' + '-' * 95)
    for label, r in [
        ('F(E) initial', r1),
        ('F_inv(E) post-involutivity', r2),
        ('F_inv(E)/conj post-conjugation', r3),
        ('Lean cascade halt', r4),
    ]:
        if r['total_phi'] > 0:
            d_t = r['dark_phi'] / r['total_phi']
            d_v = r['dark_phi'] / r['visible_phi'] if r['visible_phi'] > 0 else float('inf')
        else:
            d_t = d_v = 0
        print(f"  {label:<45}{r['visible_phi']:>12.3f}{r['dark_phi']:>10.3f}{d_t:>13.4f}{d_v:>15.4f}")

    # Compare each state's dark/total to framework constants
    print(f"\n{'='*100}")
    print("Framework-constant comparison: dark/total ratio at each state")
    print(f"{'='*100}")
    for label, r in [
        ('F(E) initial', r1),
        ('F_inv(E) post-involutivity', r2),
        ('F_inv(E)/conj post-conjugation', r3),
        ('Lean cascade halt', r4),
    ]:
        if r['total_phi'] > 0:
            ratio = r['dark_phi'] / r['total_phi']
            compare_to_constants(ratio, f"dark/total at {label}")

    # Also compare cumulative: total visible (sum across all stages) vs
    # total dark observed at any stage
    cum_visible = r1['visible_phi'] + r2['visible_phi'] + r3['visible_phi'] + r4['visible_phi']
    cum_dark = r1['dark_phi'] + r2['dark_phi'] + r3['dark_phi'] + r4['dark_phi']
    print(f"\n{'='*100}")
    print(f"Cumulative across all stages:")
    print(f"  visible Φ_cum: {cum_visible:.3f} bits")
    print(f"  dark Φ_cum:    {cum_dark:.3f} bits")
    print(f"  dark / total:  {cum_dark / (cum_visible + cum_dark):.4f}")
    print(f"  dark / visible: {cum_dark / cum_visible if cum_visible > 0 else 'inf':.4f}")
    if cum_visible > 0:
        compare_to_constants(cum_dark / (cum_visible + cum_dark), 'cumulative dark/total')
