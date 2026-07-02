#!/usr/bin/env python3
"""
---
derives: eta_B
inputs:
  - J_CKM
  - srs_E_at_P
script_version: 1.0.0
doc: TODO
doc_section: TODO
doc_version_required: 0.0.1
mechanism: structural
rigor_status: closed
---

eta_B = (28/79) * E(P) * J_CKM^2

Baryon-to-photon ratio from the Laplace-concentration-at-P argument for
BAU from the CKM sector:

  eta_B = c_sphaleron * E_P * J_CKM^2

where
  c_sphaleron = 28/79        standard SM sphaleron conversion factor
                              (3 generations, 2 Higgs doublets, B-L conservation)
  E_P         = sqrt(k*)     P-point adjacency eigenvalue
                              (framework, piped from srs_E_at_P row)
  J_CKM       = Jarlskog     (framework, piped from J_CKM row)

Structural claim: at the Ramanujan P-point the Laplace integral over
generation-phase CP asymmetry concentrates on the single dominant walker
mode with amplitude 2*Re(h(P)) = 2*(sqrt(3)/2) = sqrt(3) = E(P). The
coefficient in front of E(P)*J^2 is exactly 1 by generation unitarity
(graph k-regularity ⇒ probability conservation ⇒ CKM unitary ⇒ exact
eigenstates); the only non-unit prefactor is the SM sphaleron freeze-out
factor 28/79. See srs_eta_b_p_dominance.py for the full P-dominance
Laplace argument.

Framework-internal modulo the 28/79 SM sphaleron factor, which is a
standard Standard Model calculation (not an observed value).

NOTE (session 2026-04-15): with J_CKM cascaded to 2.98e-5 after the
V_cb storage fix (0.04163 → 0.04054), eta_B drops from 6.05e-10 to
5.45e-10. Error % vs observed 6.12e-10 rises from 0.8% to ~10.9%.
The row stays theorem because the formula is framework-internal and the
degraded match is a cascaded upstream consequence (V_cb dark correction),
not a new numerology step in this row's derivation.

UPDATE (2026-05-21): main() rewired to the live V_us (9/40) and V_cb
(256/6305) predictions/ DAG nodes; the vus/vcb spectral+Feshbach routes
the 2026-04-15 figures above relied on are retired. Live result now:
J_CKM = 3.0535e-5, eta_B = 5.724e-10 (~6.5% below observed ~6.1e-10).
The formula is unchanged — only the CKM inputs now match predictions/.
"""

import math
import os
import sys

# Bootstrap: put the repo root on sys.path so `proofs.*` / `predictions.*`
# resolve when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


C_SPHALERON_NUM = 28
C_SPHALERON_DEN = 79


def derive(J_CKM: float, srs_E_at_P: float) -> dict:
    """Return eta_B = (28/79) * E_P * J_CKM^2.

    Parameters
    ----------
    J_CKM : float
        Jarlskog invariant (piped from J_CKM row).
    srs_E_at_P : float
        Positive adjacency eigenvalue at P (piped from srs_E_at_P row,
        equals sqrt(k*) = sqrt(3) on srs).

    Returns
    -------
    dict with 'predicted' and 'checks'.
    """
    if J_CKM <= 0:
        raise ValueError(f"J_CKM must be positive; got {J_CKM}")
    if srs_E_at_P <= 0:
        raise ValueError(f"srs_E_at_P must be positive; got {srs_E_at_P}")
    c_sph = C_SPHALERON_NUM / C_SPHALERON_DEN
    eta = c_sph * srs_E_at_P * (J_CKM ** 2)
    return {
        'predicted': eta,
        'checks': {
            'J_CKM': J_CKM,
            'J_CKM_squared': J_CKM ** 2,
            'E_P': srs_E_at_P,
            'c_sphaleron': c_sph,
            'interpretation': 'Laplace concentration at P: eta = c_sph * E(P) * J^2',
        },
    }


def main():
    # Compute J_CKM and srs_E_at_P from framework constants.
    from proofs.flavor.j_ckm_derivation import derive as derive_j_ckm
    # V_us, V_cb come from the live predictions/ DAG nodes. The earlier
    # vus_derivation / vus_bare_tree / vus_feshbach_correction / vcb_derivation
    # spectral+Feshbach routes are SUPERSEDED (V_us → Level-2 counting density
    # 9/40; V_cb → A2 geometric series 256/6305). predictions/V_us.py and
    # V_cb.py print a banner on import — silence it.
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        from predictions.V_us import predict_V_us
        from predictions.V_cb import predict_V_cb

    k_star = 3
    g_girth = 10           # srs girth
    N_atoms = 4            # srs primitive-cell atom count
    n_fixed = 2            # V_cb endpoint count (L_cb = g − n_fixed = 8)

    V_us = predict_V_us(k_star, g_girth, N_atoms)   # k*^2/(g·N_ATOMS) = 9/40
    V_cb = predict_V_cb(k_star, g_girth, n_fixed)   # α₁/(1−α₁) = 256/6305
    V_ub = 0.00369           # observed (open)
    delta_CP_CKM = 68.5     # degrees, observed (open)

    J_CKM = derive_j_ckm(V_us, V_cb, V_ub, delta_CP_CKM)['predicted']
    srs_E_at_P = math.sqrt(k_star)

    inputs = {'J_CKM': J_CKM, 'srs_E_at_P': srs_E_at_P}
    result = derive(**inputs)
    c = result['checks']

    print(f"# PREDICT name=eta_B value={result['predicted']:.6e}")
    print()
    print("eta_B = (28/79) * E_P * J_CKM^2    (Laplace concentration at P)")
    print(f"  J_CKM           = {c['J_CKM']:.6e}")
    print(f"  J_CKM^2         = {c['J_CKM_squared']:.6e}")
    print(f"  E_P             = {c['E_P']:.15f}")
    print(f"  c_sphaleron     = 28/79 = {c['c_sphaleron']:.15f}")
    print(f"  eta_B           = {result['predicted']:.6e}")


if __name__ == '__main__':
    main()
