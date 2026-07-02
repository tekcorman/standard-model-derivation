#!/usr/bin/env python3
"""
---
derives: V_cb
inputs:
  - alpha_1
script_version: 1.0.0
doc: TODO
doc_section: TODO
doc_version_required: 0.0.1
mechanism: edge_local_commensurate
rigor_status: heuristic
rigor_route: ihara_bass_d8_symbolic
---

V_cb = alpha_1 * (1 + alpha_1)

Composite CKM V_cb theorem. Bare amplitude is alpha_1 = (2/3)^(g-2) = (2/3)^8
— the NB walk survival at girth-2 on srs, piped from the alpha_1 YAML row.
The commensurate girth-cycle detour correction (1 + alpha_1) comes from the
"edge-local commensurate" class of dark_correction_theorem_2026-04-14.md §4c.5b:
at integer walk length L = g-2, the commensurate-phase detour coefficient is
c = 1, distinct from V_us's Feshbach amplitude class which uses √5/4.

The explicit Ihara-Bass tree-resolvent derivation of the c=1 coefficient at
d=8 on srs is an open structural rigor route (tagged `ihara_bass_d8_symbolic`
in the frontmatter). The current derivation is structural (framework-internal,
no existing physics inputs), but the explicit walk-operator computation closing
c=1 from first principles is pending. Note: the current theorem-grade V_cb
derivation lives in `proofs/flavor/vcb_hashimoto_bfs.py` (A2 geometric series,
256/6305); this file is the older commensurate-detour route.
"""

import sys


def derive(alpha_1: float) -> dict:
    """Return the composite V_cb prediction.

    V_cb = alpha_1 * (1 + alpha_1)

    Parameters
    ----------
    alpha_1 : float
        Bare NB walk survival at girth-2 = ((k*-1)/k*)^(g-2). On srs with
        k*=3, g=10: (2/3)^8 ≈ 0.039018.

    Returns
    -------
    dict with keys:
        predicted : float — V_cb
        checks : dict of intermediate quantities
    """
    if not 0.0 < alpha_1 < 1.0:
        raise ValueError(f"alpha_1 must be in (0,1); got {alpha_1}")
    bare = alpha_1
    correction_factor = 1.0 + alpha_1
    predicted = bare * correction_factor
    return {
        'predicted': predicted,
        'checks': {
            'alpha_1': alpha_1,
            'bare': bare,
            'correction_factor': correction_factor,
            'bare_correction_term': bare * alpha_1,
        },
    }


def main():
    # Framework constants (hardcoded, no YAML dependency)
    alpha_1 = (2.0 / 3.0) ** 8

    inputs = {'alpha_1': alpha_1}
    result = derive(**inputs)

    print(f"# PREDICT name=V_cb value={result['predicted']:.15f}")
    print()
    print("V_cb = alpha_1 * (1 + alpha_1)  (edge-local commensurate girth-cycle detour)")
    print(f"  alpha_1            = {inputs['alpha_1']:.15f}")
    print(f"  bare = alpha_1     = {result['checks']['bare']:.15f}")
    print(f"  correction factor  = 1 + alpha_1 = {result['checks']['correction_factor']:.15f}")
    print(f"  V_cb               = {result['predicted']:.15f}")


if __name__ == '__main__':
    main()
