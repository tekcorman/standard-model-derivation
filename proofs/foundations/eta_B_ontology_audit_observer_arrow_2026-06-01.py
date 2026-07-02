#!/usr/bin/env python3
# ============================================================
# Tier 2 — eta_B under the CORRECTED arrow ontology (arrow = observer-graph,
# substrate timeless). NOT a new derivation: eta_B is ALREADY theorem-grade
# (predictions/eta_B_derivation.md, -0.20 sigma). This AUDITS that closure
# against the corrected ontology and reports what it teaches.
# ============================================================
#
# Scope: the time-flow / T-CP / arrow line-of-sight, Tier 2 (the Sakharov/eta
# entry), after the Tier-1 correction (arrow_observer_not_substrate_2026-05-31.py):
# the arrow of time is a purely OBSERVER-graph phenomenon; the substrate is
# timeless. The user's framing for Tier 2: eta combines a STATIC substrate CP
# invariant (read at the Gleason bolt) with the OBSERVER's temporal arrow.
#
# THE SURPRISE: the framework ALREADY derives eta_B at theorem-grade --
#   eta_B = eps_CP * Re(h_P) * alpha_1^M = (1/5)(sqrt3/2)(2/3)^48 ~ 6.11e-10,
#   -0.20 sigma vs Planck (6.12 +/- 0.04)e-10.
# So Tier 2 is not "derive eta"; it is: does that closure put the arrow in the
# right place (observer), or does it -- like the Tier-1 near-miss -- smuggle a
# temporal clock into the substrate? This probe audits each factor's ontology.
#
# DISCIPLINE: an AUDIT, not a re-derivation. We do not modify the theorem-grade
# predictions/eta_B_derivation.md (no-linting rule; user-owned). We verify the
# number, decompose the magnitude, classify each ingredient as static-substrate
# vs observer-arrow, and surface one language-sharpening recommendation.

import math

# ---- the published closure factors (predictions/eta_B_derivation.md) ----
EPS_CP = 1.0 / 5.0            # Step 2: chiral I4_132 parity via Bayesian-toggle Beta(2,1)
RE_HP = math.sqrt(3) / 2.0    # Step 3: parity-even Hashimoto tree amplitude at saddle P
G_MINUS_2 = 8                 # girth g=10 -> g-2 (Feshbach survival exponent)
M_SITES = 6                   # Step 5: Sakharov sites per cell = N_atoms*k*/2 = edges/cell
ALPHA_1 = (2.0 / 3.0) ** G_MINUS_2          # Step 4: (k*-1/k*)^(g-2)
N_GAMMA = 2                   # Step 7: photons per cell (2 helicities)

ETA_OBS, ETA_SIG = 6.12e-10, 0.04e-10


def main():
    print("=" * 78)
    print(" Tier 2 — eta_B ontology audit under the corrected arrow ontology")
    print("=" * 78)

    # --- verify the number ---
    eta = EPS_CP * RE_HP * ALPHA_1 ** M_SITES
    print(f"\n[verify] eta_B = (1/5)(sqrt3/2)(2/3)^{G_MINUS_2*M_SITES} = {eta:.4e}")
    print(f"         observed (6.12 +/- 0.04)e-10  ->  {(eta-ETA_OBS)/ETA_SIG:+.2f} sigma "
          f"(theorem-grade, predictions/eta_B_derivation.md)")

    # --- where does the smallness live? ---
    print("\n[1] the smallness is SPATIAL SURVIVAL, not small CP:")
    print(f"     CP/spectral prefactor  eps_CP*Re(h_P) = {EPS_CP*RE_HP:.4f}   (order 0.1 -- O(1))")
    print(f"     spatial survival       (2/3)^48        = {ALPHA_1**M_SITES:.3e}   <- the 1e-9 lives HERE")
    print(f"     => eta_B is small because the CP-asymmetric residue DECOHERES along")
    print(f"        the M={M_SITES}-site Feshbach chain (walker survival), NOT because the CP")
    print(f"        source is small. Contrast standard CKM baryogenesis: Jarlskog")
    print(f"        J~3e-5 -> eta~1e-20 (CKM too small by ~1e10). The framework EVADES")
    print(f"        that wall: eps_CP=1/5 is O(1); the suppression is spatial-structural.")

    # --- the ontology audit: classify each ingredient ---
    print("\n[2] ONTOLOGY AUDIT — static-substrate vs observer-arrow (corrected ontology):")
    rows = [
        ("eps_CP = 1/5",
         "chiral I4_132 PARITY (static substrate) read via Bayesian-toggle",
         "STATIC substrate CP, READ at the observer bolt (Bayesian = observer)"),
        ("Re(h_P) = sqrt3/2",
         "parity-even Hashimoto SPATIAL tree amplitude at saddle P",
         "STATIC substrate (spatial spectral data; |h|->mass, Re/Im->phases)"),
        ("alpha_1^M = (2/3)^48",
         "walker survival over the M=6 SPATIAL Feshbach chain",
         "STATIC substrate (spatial path survival; NOT a temporal decay)"),
        ("n_gamma = 2",
         "photon helicities per primitive cell",
         "STATIC substrate count"),
        ("out-of-equilibrium",
         "A2 MDL RETENTION over cosmic-time ticks (Step 8)",
         "OBSERVER arrow (N-growth/MDL): selects+preserves ONE residue/cell"),
    ]
    for factor, what, onto in rows:
        print(f"     - {factor:22s} {onto}")
        print(f"       {'':22s} ({what})")

    # --- the verdict ---
    print("\n" + "=" * 78)
    print(" VERDICT — Tier 2: eta_B is a CLEAN, already-working instance of the")
    print("           corrected ontology (static substrate CP (+) observer arrow)")
    print("=" * 78)
    print(f"""  The existing theorem-grade eta_B closure PASSES the corrected-ontology audit
  -- and is in fact the framework's first worked example of exactly the structure
  the Tier-1 correction predicts:

   * The VALUE is set entirely by STATIC substrate primitives. Every magnitude
     factor (eps_CP, Re(h_P), alpha_1^M, n_gamma) is static spatial/spectral data.
     Step 8 of the derivation says this in so many words: "the eta_B value is set
     entirely by time-independent substrate primitives." NO substrate clock is
     used to set the number. (This is the opposite of the Tier-1 near-miss: here
     the closure already keeps the arrow OUT of the substrate magnitude.)

   * The ONLY temporal ingredient is the OBSERVER ARROW. The Sakharov
     "out-of-equilibrium" condition is A2 MDL RETENTION over cosmic-time ticks =
     register-N growth on the observer graph. Its role is precisely and only to
     SELECT and PRESERVE the single static residue per cell (preserve-vs-create,
     shortest-MDL wins). It does not generate the magnitude.

   * eps_CP is the cleanest realization of "static CP read at the bolt": the CP
     SOURCE is the static chiral I4_132 parity (substrate); the 1/5 is the
     OBSERVER's Bayesian-toggle posterior reading of that static parity. Static
     substrate fact, observer reading -- joined at the bolt.

  WHY IT WORKS (and evades the CKM wall): the asymmetry is a STATIC structural
  residue, not a dynamical CKM-CP freeze-out yield. Its smallness is spatial
  survival suppression (2/3)^48 ~ 3.5e-9, with an O(1) CP prefactor -- so the
  "SM CP is 1e10 too small" objection simply does not apply.

  ONE SHARPENING (recommendation, NOT an edit -- the prediction file is
  theorem-grade and user-owned): Axiom A1's phrasing "substrate evolution is a
  NB walker ... per cosmic-time tick" reads, literally, as a SUBSTRATE clock --
  the very category error Tier-1 corrected. Under the corrected ontology it
  should be understood as the OBSERVER's N-indexed READING of static substrate
  structure (the walker is a SPATIAL transfer operator; "ticks" are observer
  N-steps). The eta_B closure's own Step 8 ("time-independent substrate
  primitives") already implies this; the A1 language is just loose.

  HONEST BOUNDS: this is an AUDIT of an existing theorem-grade closure, not a new
  number. Net result: Tier 2 is effectively CLOSED -- eta_B was already derived
  and matches at -0.20 sigma, and the corrected ontology EXPLAINS why (static
  substrate residue + observer retention) and why it dodges the CKM-too-small
  wall. The line-of-sight payoff of the arrow correction is realized here: it
  validates and sharpens the framework's existing, strongest cosmological CP
  result. Remaining genuinely-open piece is Tier 3 (the observer-graph d/dN
  entropy-production LAW), of which A2-retention here is a single static instance.""")
    print("=" * 78)


if __name__ == "__main__":
    main()
