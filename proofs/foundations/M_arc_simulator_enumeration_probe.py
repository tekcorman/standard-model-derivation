#!/usr/bin/env python3
"""
M_arc_simulator_enumeration_probe.py
====================================

First simulator-driven enumeration of the M-arc choice space, per the theory
writeup an internal working note §6.

Runs every (J variant) × (basis) × (SU(3) embedding) × (U(1)_Y formula) ×
(reduction) tuple through the C1-C4 hard-constraint filter in
`simulator.gating.spectral_consistency`.  Reports pass/fail breakdown,
identifies surviving tuples, and produces the input data for the next
session's M-arc verdict doc.

The point of this probe is NOT to do new analytic work — it is to
SYSTEMATICALLY check every framework-motivated spectral-triple candidate
and replace surprise-after-surprise with enumerate-and-filter.

No graded content changes.  This probe consumes the M1 + M2 operator
constructors and the C1-C4 axioms; if 0 candidates survive, the verdict
names exactly which CC axiom fails.  If 1-few survive, M3-M5 proceed
on those.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from simulator.menus import spectral_triple as st  # noqa: E402
from simulator.gating import spectral_consistency as sc  # noqa: E402


def _fmt_diag(d):
    if isinstance(d, dict):
        parts = []
        for k, v in d.items():
            if isinstance(v, float):
                parts.append(f'{k}={v:.3e}')
            else:
                parts.append(f'{k}={v}')
        return ', '.join(parts)
    return repr(d)


def main():
    print('=' * 100)
    print('M-arc simulator enumeration — C1-C4 filter over the spectral-triple choice space')
    print('=' * 100)

    menu = st.enumerate_full_menu(include_unbuilt_basis=True)
    print(f'\nTotal menu candidates (including unbuilt PS-tensor-product basis): '
          f'{len(menu)}')
    constructable = [c for c in menu if c.is_constructable]
    not_constructable = [c for c in menu if not c.is_constructable]
    print(f'  Constructable (built basis + matching SU(3)): {len(constructable)}')
    print(f'  Not constructable (PS-tensor-product not built): '
          f'{len(not_constructable)}')

    print('\n' + '-' * 100)
    print('Running C1-C4 over constructable tuples')
    print('-' * 100)

    results = [sc.evaluate(c) for c in constructable]
    survivors = [r for r in results if r.passes_all]

    # Pass/fail breakdown per axiom
    c1_pass = sum(1 for r in results if r.c1_ok)
    c2_pass = sum(1 for r in results if r.c2_ok)
    c3_pass = sum(1 for r in results if r.c3_ok)
    c4_pass = sum(1 for r in results if r.c4_ok)
    print(f'\n  Per-axiom pass counts (out of {len(results)}):')
    print(f'    C1 Lie closure              : {c1_pass}')
    print(f'    C2 gauge equiv with D_F     : {c2_pass}')
    print(f'    C3 factors commute on ℂ^8   : {c3_pass}')
    print(f'    C4 CC J axioms              : {c4_pass}')

    # Failure-pattern histogram
    pattern_counter: Counter[tuple[bool, bool, bool, bool]] = Counter()
    for r in results:
        pattern_counter[(r.c1_ok, r.c2_ok, r.c3_ok, r.c4_ok)] += 1
    print('\n  Failure patterns  (C1, C2, C3, C4) → count:')
    for pattern, n in sorted(pattern_counter.items(), key=lambda kv: -kv[1]):
        marker = 'PASS' if all(pattern) else 'FAIL'
        print(f'    {pattern}  → {n:3d}   ({marker})')

    # Per-tuple summary table
    print('\n' + '-' * 100)
    print('Per-tuple results')
    print('-' * 100)
    for r in results:
        c = r.choice
        print(f'  {c.j_variant:5s} | {c.basis:18s} | {c.su3_embedding:25s} | '
              f'{c.u1y_formula:32s} | {c.reduction:8s} | '
              f'C1={int(r.c1_ok)} C2={int(r.c2_ok)} C3={int(r.c3_ok)} C4={int(r.c4_ok)}'
              f'  {"PASS" if r.passes_all else "FAIL"}')

    # Survivors
    print('\n' + '-' * 100)
    print(f'Survivors of C1-C4: {len(survivors)}')
    print('-' * 100)
    if not survivors:
        print('  None.')
    else:
        for r in survivors:
            print(f'  {r.choice.name}')
            print(st.describe_choice(r.choice))

    # Sample diagnostics for one failing tuple per category
    print('\n' + '-' * 100)
    print('Sample diagnostics (one tuple per distinct failure pattern)')
    print('-' * 100)
    seen_patterns = set()
    for r in results:
        pat = (r.c1_ok, r.c2_ok, r.c3_ok, r.c4_ok)
        if pat in seen_patterns:
            continue
        seen_patterns.add(pat)
        print(f'\n  Pattern {pat}: {r.choice.name}')
        print(f'    C1 detail: {r.c1_detail}')
        print(f'    C2 diag  : {_fmt_diag(r.c2_diag)}')
        print(f'    C3 diag  : {_fmt_diag(r.c3_diag)}')
        print(f'    C4 diag  : {_fmt_diag(r.c4_diag)}')

    # Not-constructable summary
    if not_constructable:
        print('\n' + '-' * 100)
        print(f'Not-constructable tuples ({len(not_constructable)})')
        print('-' * 100)
        for c in not_constructable[:5]:
            print(f'  {c.name}')
        if len(not_constructable) > 5:
            print(f'  ... and {len(not_constructable) - 5} more requiring '
                  f'ps_tensor_product basis (M2.refined deliverable).')

    print('\n' + '=' * 100)
    print('M-arc simulator enumeration: sentinel done.')
    print('=' * 100)


if __name__ == '__main__':
    main()
