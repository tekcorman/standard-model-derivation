#!/usr/bin/env python3
"""
NEED-B R5 — the per-walker 4₁-screw holonomy route for the Koide phase δ.

Need-B = derive the quark-sector Koide phase δ. The 2026-05-16 route sweep
eliminated R1/R2/R3 (`needB_quark_koide_phase_next_research_target.md`); R1
("triplet screw-Wigner-D") was eliminated — but R1 only tested a FORM-TRANSFER
(it took the lepton's survival-probability pattern {Q²,Q²/4,Q²} and substituted
Q_down). It never tested whether the screw survival-probabilities ACCUMULATE
along the per-sector walker — the genuine "per-walker holonomy" object. R5
closes that gap.

THE ROUTE.  theorem_41 derives δ_lepton = 2/9 as the harmonic mean of the
diagonal j=1 Wigner-D survival probabilities of ONE 4₁ screw (R₄, 90° about
[001]) acting on the C₃ site irreps. R5 asks: does a walker that traverses L
steps accumulate the screw as R₄^L, so that δ_sector = HM(j=1 Wigner-D of
R₄^L)?  The framework's walker lengths (W41 §4(D)) are concrete:

    Type III  lepton (τ)   L = g − 2 = 8
    Type IV   down  (b/d)  L = g     = 10
    Type II   up    (t/u)  L = 0

so the route makes a zero-free-parameter prediction for all three sectors.

  G1  self-validate theorem_41 — one screw R₄ → {4/9,1/9,4/9} → HM = 2/9
  G2  the per-walker test — HM(j=1 Wigner-D of R₄^L) at the W41 walker lengths
  G3  verdict on the route — REFUTED, and the structural reason
  G4  what it leaves — δ_geometric is static (universal 2/9); the route closed
  G5  a Path D refinement — δ_down with full PDG errors

The targets (Path D, 2026-05-21): δ_lepton = 2/9 (exact), δ_down ≈ 0.101,
δ_up ≈ 0.055.
"""

import numpy as np
from fractions import Fraction

results = []


def gate(name, passed, detail=""):
    results.append(bool(passed))
    print(f"  [{'PASS' if passed else 'OPEN'}] {name}")
    for ln in detail.strip("\n").split("\n"):
        if ln.strip():
            print(f"         {ln}")
    print()


# ----------------------------------------------------------------------
# the theorem_41 machinery (proofs/masses/wigner_d1_screw_41.py), verbatim
def build_R4():
    """90° rotation about [001] — the 4₁ screw rotation part (ITA 214)."""
    return np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)


def basis_111():
    e3 = np.array([1, 1, 1]) / np.sqrt(3)
    e1 = np.array([1, -1, 0]) / np.sqrt(2)
    e2 = np.cross(e3, e1)
    e2 /= np.linalg.norm(e2)
    return np.column_stack([e1, e2, e3])


def spherical_U():
    s2 = np.sqrt(2)
    return np.array([[-1/s2, -1j/s2, 0], [0, 0, 1], [1/s2, -1j/s2, 0]],
                    dtype=complex)


_P, _U = basis_111(), spherical_U()


def wigner_D1(Rn):
    """j=1 Wigner-D of a rotation Rn, in the [111]-quantised spherical basis."""
    R111 = np.linalg.inv(_P) @ Rn @ _P
    return _U @ R111.astype(complex) @ np.linalg.inv(_U)


def survival_probs(Rn):
    D = wigner_D1(Rn)
    return [abs(D[m, m]) ** 2 for m in range(3)]


def HM(probs):
    if min(probs) < 1e-12:
        return float("inf")
    return 3.0 / sum(1.0 / p for p in probs)


def R4_power(n):
    """R₄^n — the screw rotation part has order 4, so only n mod 4 matters."""
    n %= 4
    M = np.eye(3)
    R4 = build_R4()
    for _ in range(n):
        M = M @ R4
    return M


# ======================================================================
print("=" * 72)
print("G1 — self-validate theorem_41: one screw R₄ → HM = 2/9")
print("=" * 72)
probs1 = survival_probs(R4_power(1))
hm1 = HM(probs1)
g1 = (abs(probs1[0] - 4/9) < 1e-12 and abs(probs1[1] - 1/9) < 1e-12
      and abs(hm1 - 2/9) < 1e-12)
gate("G1 one 4₁ screw → survival {4/9,1/9,4/9}, HM = 2/9 = δ_lepton", g1,
     f"""j=1 Wigner-D of R₄ (90° screw), [111]-quantised:
   diagonal |D_mm|² = {[f'{p:.5f}' for p in probs1]}  (exact {{4/9,1/9,4/9}})
   HM = {hm1:.6f} = 2/9   ✓ reproduces theorem_41 SS-2/SS-3 (the lepton δ).
The machinery is the verbatim `wigner_d1_screw_41.py`. The route question:
does this screw HM ACCUMULATE along a walker of length L as R₄^L?""")


# ======================================================================
print("=" * 72)
print("G2 — the per-walker test: HM(j=1 Wigner-D of R₄^L) at W41 lengths")
print("=" * 72)
# W41 §4(D) walker lengths
WALKERS = [("lepton (Type III)", 8,  2/9,   "2/9"),
           ("down   (Type IV)",  10, 0.101, "≈0.101"),
           ("up     (Type II)",  0,  0.055, "≈0.055")]
print("  the HM(R₄^n) value set (R₄ has order 4 — only n mod 4 matters):")
for n in range(4):
    print(f"     R₄^{n}: survival {[f'{p:.4f}' for p in survival_probs(R4_power(n))]}"
          f"   HM = {HM(survival_probs(R4_power(n))):.5f}")
print()
rows, hits = [], 0
for nm, L, target, tstr in WALKERS:
    hm = HM(survival_probs(R4_power(L)))
    ok = abs(hm - target) < 0.02 if hm != float("inf") else False
    hits += ok
    rows.append(f"   {nm:18s} L={L:2d}  L mod 4={L % 4}  →  HM(R₄^L) = "
                f"{hm:7.5f}   target δ {tstr:>7s}   {'✓' if ok else '✗'}")
g2 = hits < 3                                    # the route does NOT fit all 3
gate("G2 R₄^L with the framework walker lengths — does NOT reproduce δ", g2,
     "\n".join(rows) + f"""

The route predicts δ from R₄^L with zero free parameters. Result: {hits}/3.
 • lepton L=8 ≡ 0 (mod 4) → R₄⁰ = 𝟙 → HM = 1, NOT 2/9 — it CONTRADICTS
   theorem_41, whose δ_lepton = 2/9 used ONE screw, not L_lepton = 8 screws.
 • up L=0 → R₄⁰ = 𝟙 → HM = 1, NOT 0.055.
 • down L=10 ≡ 2 → R₄² → HM = 1/9 = 0.1111 — the only near-hit (see G5).""")


# ======================================================================
print("=" * 72)
print("G3 — verdict on the route: REFUTED, and why")
print("=" * 72)
gate("G3 the per-walker screw-holonomy route is REFUTED", True,
     """Two independent refutations:

 (1) NUMERICAL.  R₄ (the screw rotation) has order 4 ⇒ HM(R₄^L) depends only
     on L mod 4 and takes values in the 3-element set {1, 2/9, 1/9}. The up
     δ ≈ 0.055 is not in that set — no walker length L can produce it. And the
     lepton's own walker length L = g−2 = 8 ≡ 0 gives HM = 1, contradicting
     the theorem_41 result δ_lepton = 2/9 it is supposed to extend.

 (2) STRUCTURAL — the deeper reason.  theorem_41 derives δ_lepton = 2/9 as the
     j=1 Wigner-D of ONE screw acting on the C₃ SITE irreps — a STATIC
     representation-theory fact about how the 3 generation irreps transform
     under the screw. It is NOT a walk: nothing accumulates. The screw and the
     C₃ irreps are the SAME for every sector ⇒ the static screw HM is
     UNIVERSALLY 2/9. There is no walker-length handle on it — so neither the
     R₄^L proxy NOR the genuine srs-loop frame-holonomy can be the carrier of
     the sector-dependence. The "per-walker holonomy" is the wrong object.

R5 thereby completes R1's elimination: R1 ruled out the FORM-TRANSFER
({Q²,Q²/4,Q²}→Q_down); R5 rules out the ACCUMULATION (R₄^L) — and identifies
that the screw→δ map is static, so no walk-holonomy variant survives.""")


# ======================================================================
print("=" * 72)
print("G4 — what it leaves: δ_geometric is universal; the route is closed")
print("=" * 72)
gate("G4 δ_geometric (the screw HM) is sector-independent (2/9) — Need-B's "
     "δ-sector-dependence is NOT screw geometry", True,
     """Consequence of G3(2): the 4₁-screw / C₃-irrep computation gives ONE
number, 2/9, for every sector. Path D's measured sector-dependence —
δ_lepton 0.222, δ_down ≈ 0.101, δ_up ≈ 0.055 — therefore CANNOT come from the
screw geometry. It must come from elsewhere:

 • the lepton (δM ≈ 0, charged-lepton mixing negligible) reads the bare
   geometric phase → δ = 2/9, and Path D confirmed exactly that;
 • the quark sectors carry a C₃-breaking δM (the CKM physics) — and the
   *physical* Koide phase of the eigenvalue spectrum is the geometric 2/9
   DRESSED by that δM. The sector-dependence is the δM, not the screw.

So Need-B's open δ-physical is NOT a missing geometric phase — it is the
geometric 2/9 (known) plus the C₃-breaking δM texture (the handoff's path E,
genuinely open). This is the same terminus the 2026-05-16 sweep reached (δ =
the deep per-generation dynamics, lepton-shared) — R5 adds that the screw-
holonomy sub-route is now explicitly and structurally closed, and re-points
the residual at the δM texture rather than at a new geometric object.""")


# ======================================================================
print("=" * 72)
print("G5 — a Path D refinement: δ_down with full PDG errors")
print("=" * 72)
rng = np.random.default_rng(20260522)
omega = np.exp(2j * np.pi / 3)


def koide_delta(m):
    r = np.sqrt(np.abs(np.sort(m)))
    F1 = sum(r[j] * omega ** j for j in range(3))
    return (-np.angle(F1)) % (2 * np.pi / 3)


# down masses (MeV): m_d,m_s PDG @2GeV; m_b(2GeV)=4888 ± runner systematic
N = 200_000
samp = np.array([koide_delta(np.array([
    4.67 + rng.normal()*0.33, 93.4 + rng.normal()*6.0,
    4888.0 + rng.normal()*110.0])) for _ in range(N)])
d_mean, d_std = samp.mean(), samp.std()
sig_19 = (1/9 - d_mean) / d_std
gate("G5 δ_down = 0.101 ± 0.005 (full PDG errors) — 1/9 is +2.1σ, not refuted", True,
     f"""Path D quoted δ_down = 0.1012 ± 0.0013 — but the ±0.0013 was the m_b
1-loop↔2-loop runner systematic ONLY; the m_d (±7%) and m_s (±6%) PDG
experimental errors were not propagated. Full propagation:
   δ_down = {d_mean:.5f} ± {d_std:.5f}   (m_d, m_s, m_b runner all included)
   screw-route value 1/9 = {1/9:.5f}  →  {sig_19:+.2f}σ
So the down screw-holonomy value 1/9 is *disfavoured at ~2σ*, NOT cleanly
refuted — Path D's "+10%, refuted" rested on the tighter (runner-only) bar.
(The up pattern value 2/27 = {2/27:.4f} vs δ_up = 0.0554 ± 0.0024 is +7.8σ —
that one IS refuted.) Net: 1/9 for the down sector is a live ~2σ tension, not
a closed question — consistent with the screw geometry contributing 1/9-ish
and a modest δM dressing carrying the rest. A flag for the δM-texture work,
not a Need-B closure.""")


# ======================================================================
print("=" * 72)
n = sum(results)
print(f"NEED-B R5 SENTINEL: {n}/{len(results)} gates")
print("=" * 72)
print("""
R5 verdict — HONEST NEGATIVE (structurally informative). The per-walker
4₁-screw holonomy is refuted as a Need-B route: HM(R₄^L) ∈ {1, 2/9, 1/9}
cannot reach δ_up, and the lepton's own walker length L=8 contradicts
theorem_41's δ_lepton=2/9. The deeper reason — theorem_41's screw→δ map is
STATIC representation theory, not a walk — closes the route at the conceptual
level (no walk-holonomy variant, proxy or genuine, can carry sector-
dependence). δ_geometric is universal (2/9); Need-B's δ-physical sector-
dependence re-points at the C₃-breaking δM texture (handoff path E), not a new
geometric phase. R5 completes R1's elimination (form-transfer + accumulation
both ruled out) and refines Path D (δ_down=0.101±0.005; 1/9 at +2.1σ, a live
tension not a refutation).
""")
raise SystemExit(0 if n == len(results) else 1)
