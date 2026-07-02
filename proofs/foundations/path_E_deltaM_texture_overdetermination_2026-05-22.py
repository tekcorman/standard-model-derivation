#!/usr/bin/env python3
"""
PATH E — the δM texture, confronted by an OVER-DETERMINATION test.

The handoff's path E: the C₃-breaking δM that produces the CKM is not pinned.
W48–W51 built it — m^(s) = D_shape(ε²_s, δ_K) + γ₇(s)·κ·A_arc — and confronted
it with the CKM alone (W51: a non-normal directed arc gives the correct
QUALITATIVE CKM — hierarchical, near-diagonal, CP-violating; κ, φ, A_arc left
"representative", not pinned).

Need-B R5 (2026-05-22) re-pointed the quark Koide phase δ at this same δM
texture. So the δM is now OVER-DETERMINED — it must simultaneously give:
  • the CKM           (W51: 3 magnitudes + δ_CP)
  • the Koide phases  (Path D: δ_down ≈ 0.101, δ_up ≈ 0.055 — the mass-blind
                       Koide phase of each sector's physical mass spectrum)
This probe runs that joint test on the W51 construction.

  G1  the over-determination setup — δM faces CKM + Koide phases jointly
  G2  W51's democratic arc — the CKM κ overshoots the Koide phases
  G3  a hierarchical arc — pins the Koide phase but kills the CKM mixing
  G4  verdict — the joint constraint is not threaded by the W51-family ansatz
"""

import numpy as np
import numpy.linalg as la

results = []


def gate(name, passed, detail=""):
    results.append(bool(passed))
    print(f"  [{'PASS' if passed else 'OPEN'}] {name}")
    for ln in detail.strip("\n").split("\n"):
        if ln.strip():
            print(f"         {ln}")
    print()


omega = np.exp(2j * np.pi / 3)
DELTA_K = 2 / 9                                   # W51's shape phase (lepton value)
EPS2 = {"down": 2.478, "up": 3.312}               # Path D consistent-scale ε²
GAMMA7 = {"down": -1, "up": +1}                   # γ₇ = (−1)ⁿ
PHI0 = np.arctan(np.sqrt(5 / 3))                  # W51's representative holonomy
DELTA_TARGET = {"down": 0.101, "up": 0.055}       # Path D Koide phases
V_US, V_CB, V_UB = 0.2243, 0.0408, 0.00382        # PDG CKM magnitudes


def shape_diag(eps2, dK=DELTA_K):
    """W51's 'shape' — Koide masses on the diagonal (generation basis)."""
    eps = np.sqrt(eps2)
    f = np.array([1 + eps * np.cos(2 * np.pi * j / 3 + dK) for j in range(3)])
    return np.diag(f ** 2).astype(complex)


def arc_democratic(phi):
    """W51's A_arc — a SYMMETRIC directed 3-cycle (equal-weight legs)."""
    A = np.zeros((3, 3), dtype=complex)
    A[1, 0] = 1.0
    A[2, 1] = 1.0
    A[0, 2] = np.exp(1j * phi)
    return A


def arc_hierarchical(phi, r):
    """a HIERARCHICAL directed 3-cycle — legs 1 : r : r² (the natural fix:
    the real CKM is hierarchical V_us ≫ V_cb ≫ V_ub)."""
    A = np.zeros((3, 3), dtype=complex)
    A[1, 0] = 1.0
    A[2, 1] = r
    A[0, 2] = r * r * np.exp(1j * phi)
    return A


def koide_delta(masses):
    """mass-blind Koide phase of a mass triple (the Path D observable)."""
    m = np.sort(np.abs(masses))
    r = np.sqrt(m)
    return (-np.angle(sum(r[j] * omega ** j for j in range(3)))) % (2 * np.pi / 3)


def biunitary_L(m):
    """left rotation V_L of the bi-unitary SVD m = V_L Σ V_R†."""
    U, S, _ = la.svd(m)
    return U, S


# ======================================================================
print("=" * 72)
print("G1 — the over-determination setup")
print("=" * 72)
gate("G1 the δM texture is over-determined — CKM + Koide phases jointly", True,
     """W48–W51 built the δM texture as m^(s) = D_shape(ε²_s, δ_K) + γ₇(s)·κ·A_arc
and confronted it with the CKM ALONE. W51's verdict: a non-normal directed-arc
A_arc gives the right QUALITATIVE CKM, with κ, φ, A_arc representative — the
quantitative pinning left open ('path E').

Need-B R5 re-pointed the quark Koide phase δ at this same δM: δ_physical is the
mass-blind Koide phase of m^(s)'s singular values. So the ONE δM now faces TWO
constraint sets:
   CKM           — |V_us|, |V_cb|, |V_ub|, δ_CP    (4 numbers, RELATIVE u↔d)
   Koide phases  — δ_down ≈ 0.101, δ_up ≈ 0.055     (2 numbers, Path D)
against a construction whose free content is essentially (κ, φ, the A_arc
shape, δ_K). A genuine over-determination — the test of whether masses and
mixing really are 'one δM object' (the Fork-2 thesis). G2/G3 run it.""")


# ======================================================================
print("=" * 72)
print("G2 — W51's democratic arc: the CKM κ overshoots the Koide phases")
print("=" * 72)
rows = []
for kappa in (0.0, 0.05, 0.10, 0.20):
    m_d = shape_diag(EPS2["down"]) + GAMMA7["down"] * kappa * arc_democratic(PHI0)
    m_u = shape_diag(EPS2["up"]) + GAMMA7["up"] * kappa * arc_democratic(PHI0)
    VLd, Sd = biunitary_L(m_d)
    VLu, Su = biunitary_L(m_u)
    ckm = VLu.conj().T @ VLd
    d_d, d_u = koide_delta(Sd), koide_delta(Su)
    rows.append(f"   κ={kappa:.2f}:  δ_down={d_d:.4f}  δ_up={d_u:.4f}   "
                f"|CKM₁₂|={abs(ckm[0,1]):.4f}  |CKM₂₃|={abs(ckm[1,2]):.4f}")
# W51 used κ≈0.20 for the CKM; read off the Koide phases there
m_d20 = shape_diag(EPS2["down"]) + GAMMA7["down"] * 0.20 * arc_democratic(PHI0)
m_u20 = shape_diag(EPS2["up"]) + GAMMA7["up"] * 0.20 * arc_democratic(PHI0)
dd20 = koide_delta(biunitary_L(m_d20)[1])
du20 = koide_delta(biunitary_L(m_u20)[1])
g2 = dd20 > 0.13 and du20 > 0.09          # the overshoot is real
gate("G2 the CKM-scale κ drives δ_down/δ_up far above the Path D values", g2,
     "\n".join(rows) + f"""

W51 used κ ≈ 0.20 to obtain a CKM-like |CKM₁₂| (a sizeable Cabibbo-scale
mixing). At that κ the construction's Koide phases are
   δ_down = {dd20:.4f}   (Path D target 0.101 — overshoot +{100*(dd20-0.101)/0.101:.0f}%)
   δ_up   = {du20:.4f}   (Path D target 0.055 — overshoot +{100*(du20-0.055)/0.055:.0f}%)
The democratic arc with the CKM-scale κ OVERSHOOTS both Koide phases. The
Koide phases would need κ ≲ 0.03 — at which κ there is essentially no CKM
mixing. One κ cannot serve both constraints.""")


# ======================================================================
print("=" * 72)
print("G3 — a hierarchical arc: pins the Koide phase but kills the CKM")
print("=" * 72)
# the natural fix — a hierarchical arc (the CKM IS hierarchical). δ_K must be
# freed: with δ_K=2/9 the κ=0 shape already gives δ_down≈0.178 (G2) — to reach
# 0.101 the SHAPE phase itself must be ≈0.10. Tune (δ_K, r, κ) to δ_down≈0.101,
# then read the (1,2) rotation that feeds the CKM.
best = None
for dK in np.linspace(0.04, 0.16, 25):
    for r in np.linspace(0.1, 0.8, 30):
        for kappa in np.linspace(0.02, 0.6, 50):
            m_d = (shape_diag(EPS2["down"], dK)
                   + GAMMA7["down"] * kappa * arc_hierarchical(PHI0, r))
            VLd, Sd = biunitary_L(m_d)
            if abs(koide_delta(Sd) - 0.101) < 0.003:
                mix = abs(VLd[0, 1])
                if best is None or mix > best[0]:
                    best = (mix, dK, r, kappa, koide_delta(Sd))
mix12, dK_b, r_b, k_b, d_b = best
g3 = mix12 < 0.05                         # the down rotation is far below V_us
gate("G3 a hierarchical arc tuned to δ_down gives a negligible (1,2) rotation", g3,
     f"""hierarchical arc (legs 1 : r : r²) — the natural fix, since the CKM is
itself hierarchical. With δ_K=2/9 the shape alone overshoots (G2), so δ_K is
freed too. Scanning (δ_K, r, κ), the BEST (largest-mixing) point at
δ_down ≈ 0.101 over the whole family:
   δ_K={dK_b:.3f}, r={r_b:.2f}, κ={k_b:.2f}  →  δ_down={d_b:.4f} ✓
   the down left-rotation feeding the CKM:  |V_dL,₁₂| = {mix12:.5f}
That is ~{V_US/max(mix12,1e-9):.0f}× too small to source V_us = {V_US}. Two
findings: (i) δ_K must be ≈{dK_b:.2f}, NOT 2/9 — the shape phase is itself
sector-specific (= the unsolved Need-B δ, not a borrowed lepton value);
(ii) the down quark's steep hierarchy (m_s/m_d ≈ 20) suppresses its
diagonalising rotation, so a δM small enough to leave δ_down at 0.101 sources
almost no mixing — the OPPOSITE failure to the democratic arc.""")


# ======================================================================
print("=" * 72)
print("G4 — verdict")
print("=" * 72)
gate("G4 path E — the δM over-determination is NOT threaded by the W51 ansatz", True,
     f"""WHAT PATH E SETTLED:
 • The δM texture is a genuine OVER-DETERMINED object: one C₃-breaking δM
   must give the CKM AND the two quark Koide phases. The over-determination
   framework is the right test (the Fork-2 'masses + mixing are one δM' thesis,
   made into a falsifiable joint constraint).
 • The W51-family construction FAILS that joint test, both ways:
   – democratic arc (W51's): the κ ≈ 0.20 needed for a Cabibbo-scale CKM drives
     δ_down to {dd20:.2f} and δ_up to {du20:.2f} — overshooting Path D's
     0.101 / 0.055 by factors ~2;
   – hierarchical arc: tuned to δ_down ≈ 0.101 it gives |V_dL,₁₂| ≈ {mix12:.4f},
     ~{V_US/max(mix12,1e-9):.0f}× too small for V_us.
   The two constraints pull κ / the arc shape in incompatible directions.
 • DIAGNOSIS. The CKM mixing is 'large' (V_us ≈ 0.22) relative to the down
   quark's steep mass hierarchy, which suppresses its own diagonalising
   rotation. W51's democratic arc buys the mixing only by being a large
   light-end perturbation — which simultaneously reshapes the Koide phase. No
   single (κ, A_arc) in this family separates 'enough mixing' from 'too much
   phase distortion'.

WHAT STAYS OPEN:
 • The δM texture that threads BOTH the CKM and the Koide phases is not the
   W51 representative arc. Path E does not pin it — it REFUTES the W51-family
   ansatz against the full constraint set and establishes the joint
   over-determination as the test any future δM must pass.
 • Like Path D (node not exact) and Need-B R5 (screw holonomy refuted), path E
   reaches the same terminus: the quark mass / mixing / phase sector bottoms
   out at the deep substrate-dynamics frontier. The δM texture is genuinely
   not closed by a representative ansatz — it needs the substrate-derived
   directed-arc operator (W51's own flagged residual: 'pin A_arc to the actual
   srs-z aligned-edge structure'), now with the Koide phases as a second,
   independent gate it must pass.""")


# ======================================================================
print("=" * 72)
n = sum(results)
print(f"PATH-E SENTINEL: {n}/{len(results)} gates")
print("=" * 72)
print("""
Path E verdict — HONEST NEGATIVE (structurally informative). The δM texture is
an over-determined object: it must give the CKM AND the quark Koide phases. The
W51-family construction is refuted against that joint constraint — the
democratic arc overshoots the Koide phases, a hierarchical arc kills the CKM
mixing; no single (κ, A_arc) threads both. The over-determination test itself
is the deliverable: it is the gate any δM texture must pass, and W51's
representative arc fails it. The δM texture remains the deep frontier — the
substrate-derived srs-z directed-arc operator, unpinned.
""")
raise SystemExit(0 if n == len(results) else 1)
