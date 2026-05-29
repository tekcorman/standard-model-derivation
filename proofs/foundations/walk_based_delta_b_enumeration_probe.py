#!/usr/bin/env python3
"""
walk_based_delta_b_enumeration_probe.py
=======================================

First enumeration run of the walk-based Δb search per scoping doc
`walk_based_delta_b_search_scoping_2026-05-14.md`.

Enumerates β-contribution candidates from the framework's substrate
primitives and filters by:
  - exact match to MSSM target Δb = (+5/2, +25/6, +4)
  - substrate-bounded denominators (≤ 24 = α_GUT⁻¹)
  - MDL parsimony

Reports survivors with structural P1/P2/P5 review prompts.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from simulator.menus import beta_contributions as bc  # noqa: E402
from simulator.gating import delta_b_match as gate  # noqa: E402


def main():
    print('=' * 100)
    print('Walk-based Δb enumeration — filter for MSSM target Δb = (+5/2, +25/6, +4)')
    print('=' * 100)

    candidates = bc.enumerate_full_menu()
    print(f'\nTotal candidates: {len(candidates)}')

    matching, all_results = gate.filter_matching(candidates)
    print(f'  Exact target match: {len(matching)} / {len(all_results)}')

    if not matching:
        print('\n  No candidates produce Δb = (+5/2, +25/6, +4).')
        print('  Walk-based path closed for this enumeration.')
        # Show the closest candidates by L1 distance to target
        from fractions import Fraction
        target = bc.MSSM_DELTA_B
        def l1_dist(db):
            return sum(abs(d - t) for d, t in zip(db, target))
        nonmatching = [r for r in all_results if not r.matches_target]
        nonmatching.sort(key=lambda r: l1_dist(r.delta_b))
        print('\n  Top 5 closest non-matching candidates:')
        for r in nonmatching[:5]:
            print(f'    {r.candidate.name:50s}  Δb = {r.delta_b}')
            print(f'      L1 distance from target: {float(l1_dist(r.delta_b)):.3f}')
    else:
        print('\n' + '-' * 100)
        print(f'Matching candidates ({len(matching)}), sorted by MDL bits ascending')
        print('-' * 100)
        for r in matching:
            print(f'\n  Candidate: {r.candidate.name}')
            print(f'    Origin: {r.candidate.origin_description}')
            print(r.summary())
            print(f'    Bundles ({len(r.candidate.bundles)}):')
            # Group bundles for compact display
            grouped: dict = {}
            for b in r.candidate.bundles:
                key = (b.statistics, b.rep_3, b.rep_2, b.Y)
                grouped[key] = grouped.get(key, 0) + b.mult
            for (stat, r3, r2, Y), n in sorted(grouped.items()):
                stat_short = 'S' if stat == 'scalar' else 'F'
                print(f'      [{stat_short}] ({r3}, {r2}, Y={Y})  ×{n}')

    # Diversity check: how many DISTINCT Δb values are in the enumeration?
    unique_db = set(r.delta_b for r in all_results)
    print(f'\n  Total distinct Δb triples in enumeration: {len(unique_db)}')

    print('\n' + '=' * 100)
    print('walk_based_delta_b_enumeration_probe.py: sentinel done.')
    print('=' * 100)


if __name__ == '__main__':
    main()
