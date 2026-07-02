#!/usr/bin/env python3
"""
framework state at non-present N — now a [wrap] shim onto simulator.cosmology.

ABSORBED 2026-05-18 (unified-simulator absorption plan §1 "absorb, then
wrap"; §5.II cosmology stage). This probe's original content — "the
predictions are a DAG; compute framework state at any N along the coasting
trajectory N(z)=N_hub/(1+z)" — is now OWNED by `simulator/cosmology.py`
(the S3 N_hub-axis stage). Per the plan, the probe collapses to a thin
`simulator.cosmology.query() + assert` verification. The full thesis
(10/10 load-bearing cosmology quantities are stage queries, predictions
unperturbed) is demonstrated in
`proofs/cosmology/cosmology_absorption_audit_2026-05-18.py`.

Original intent preserved: framework state at non-present N is recoverable
by stepping the coasting trajectory through the stage; here we assert the
stage reproduces it (z=0 anchor + an off-present epoch) and that the
honest domain fence holds (the recombination region is the
`frontier.acoustic_scale` boundary, not a framework claim — the canonical
2026-05-09-demo mistake of printing a z=1100 row is NOT repeated).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(  # repo root (for `simulator` + `proofs`)
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator import cosmology
from proofs.cosmology.lib.ontology import Frame

if __name__ == "__main__":
    # z=0 anchor: the stage owns the cascade (DAG-authority verified).
    H0 = cosmology.hubble(0.0, Frame.SUBSTRATE).value
    t0 = cosmology.age(0.0, Frame.SUBSTRATE).value
    assert abs(H0 - 68.1784) < 1e-2 and abs(t0 - 14.3419) < 1e-2, (H0, t0)

    # Off-present epoch (coasting N(z)=N_hub/(1+z)): H ∝ (1+z), t ∝ 1/(1+z),
    # in the framework-claim domain (z ≲ 2). The stage owns the trajectory.
    z = 1.0
    assert abs(cosmology.hubble(z, Frame.SUBSTRATE).value
               - H0 * (1.0 + z)) < 1e-6
    assert abs(cosmology.age(z, Frame.SUBSTRATE).value
               - t0 / (1.0 + z)) < 1e-6

    # Honest domain fence: recombination is the acoustic_scale frontier
    # (extraction-layer / out of scope), NOT a framework claim — the probe
    # no longer fabricates a z=1100 framework-state row.
    try:
        cosmology.acoustic_scale()
        raise AssertionError("acoustic_scale must raise (frontier boundary)")
    except NotImplementedError:
        pass

    print("WRAP OK — framework state at non-present N is owned by "
          "simulator.cosmology;")
    print("z=0 anchor + off-present epoch verified; recombination correctly "
          "fenced to frontier.acoustic_scale. Full thesis: "
          "cosmology_absorption_audit_2026-05-18.py.")
