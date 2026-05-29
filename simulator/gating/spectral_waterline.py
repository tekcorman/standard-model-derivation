"""
S2 — the SPECTRAL-MODE rate-distortion waterline (the uniform_Q_density
Theorem-A gate). Route-4 Gap-1.

DISTINCT from `gating/waterfilling.py` — DO NOT conflate (the conflation
of these two was a documented session error; handoff §3k.1 gap-1):

  • `waterfilling.py`  : the NET-ENSEMBLE A2-T filter — *which crystal
    net* (srs-forced post-R-9). Boltzmann over realizations.
  • `spectral_waterline.py` (THIS) : the RATE-DISTORTION-ON-SPECTRAL-
    MODES filter — *which Q-space Fourier mode M_n* survives at the
    channel's structural scale N. Binary Rissanen/BIC threshold.

Theorem A (predictions/uniform_Q_density_derivation.md §8-58; Rissanen
1978 two-part MDL + Pinsker/χ² Cover–Thomas 17.3.2): at the MDL optimum
a Q-space Fourier mode of amplitude ε on support Δφ is RETAINED iff

        ε² · Δφ  >  log(N) / N            (binary; NOT soft (|M|−θ)₊)

with N the channel's structural sample scale. For the δρ channel
N = 2|E| = 12 (the c_S = 1/(2|E|) constant; live structural_counts
n_directed_edges = 12). Reference logic validated standalone in
`proofs/foundations/delta_rho_C1_waterlevel_derivation_2026-05-17.py`
(live: at N∈{4,6,12} the M₂ mode is ZEROED ⇒ leading; binary full
retention ⇒ −9.54% overshoot; the closing crossover N≈55 is NOT a
framework constant — fitting it = refused numerology).

Pure functions; no fitted constants; the threshold is the DERIVED
log(N)/N, the scale is the DERIVED 2|E|. New first-class gate ADDED to
gating/ (no existing gating file modified — shared-infra discipline).
"""

from __future__ import annotations

import math

# δρ-channel structural sample scale (c_S = 1/(2|E|); live srs
# structural_counts: n_directed_edges = 12). NOT a free parameter.
DELTA_RHO_N = 2 * 6   # 2|E|, |E| = n_edges_per_cell = 6  ⇒ 12


def rissanen_threshold(N: float) -> float:
    """The derived binary MDL retention threshold log(N)/N (Theorem A)."""
    if N <= 1.0:
        raise ValueError(f"spectral_waterline: N must be > 1, got {N}")
    return math.log(N) / N


def retain(eps: float, delta_phi: float, N: float = DELTA_RHO_N) -> bool:
    """Theorem-A binary retention: a Q-space mode of amplitude `eps` on
    support `delta_phi` survives the rate-distortion waterline at
    structural scale `N` iff eps²·Δφ > log(N)/N. Binary (Rissanen/BIC),
    NOT a soft attenuation. Default N = 2|E| = 12 (the δρ channel)."""
    return (eps * eps * delta_phi) > rissanen_threshold(N)


def retained_modes(modes, delta_phi: float = 1.0,
                    N: float = DELTA_RHO_N) -> list:
    """Filter an iterable of (name, eps) spectral modes to those that
    clear the Theorem-A waterline at scale N. Returns the retained list;
    what is NOT returned is the *discarded* set (Route-4's object — see
    `discarded_modes`)."""
    return [(nm, e) for (nm, e) in modes if retain(e, delta_phi, N)]


def discarded_modes(modes, delta_phi: float = 1.0,
                     N: float = DELTA_RHO_N) -> list:
    """The complement: modes ZEROED by the Theorem-A binary gate at scale
    N. This is exactly the set whose contribution the filtered model
    discards — the raw-minus-filtered object Route-4 must enumerate."""
    return [(nm, e) for (nm, e) in modes if not retain(e, delta_phi, N)]


def summary() -> dict:
    return {
        "gate": "Theorem-A rate-distortion-on-spectral-modes (binary "
                "Rissanen/BIC); DISTINCT from waterfilling net-ensemble",
        "rule": "retain iff eps^2 * delta_phi > log(N)/N",
        "delta_rho_scale_N": DELTA_RHO_N,
        "delta_rho_threshold_logN_over_N": rissanen_threshold(DELTA_RHO_N),
        "reference": "predictions/uniform_Q_density_derivation.md §8-58; "
                     "delta_rho_C1_waterlevel_derivation_2026-05-17.py",
    }


if __name__ == "__main__":
    s = summary()
    print("spectral_waterline (Route-4 Gap-1) — Theorem-A gate")
    for k, v in s.items():
        print(f"  {k}: {v}")
    # sanity vs the validated C1 reference: at N=12 a unit-ish M₂ amplitude
    # on Δφ=1 must be ZEROED (the documented finding).
    thr = rissanen_threshold(12)
    print(f"  log(12)/12 = {thr:.6f}")
    # C1: |M2|≈0.27 (Inv#3 measured), Δφ=1 ⇒ eps²Δφ = 0.0729 < 0.2071 ⇒ ZEROED
    assert not retain(0.27, 1.0, 12), "C1: M2 must be zeroed at N=12"
    assert retain(0.99, 1.0, 12), "a near-unit mode must clear at N=12"
    print("  OK: matches the validated C1 waterlevel finding "
          "(M₂ zeroed at N=2|E|=12).")
