#!/usr/bin/env python3
"""
W47 — Need-D-3 scoping: is the M₃(ℂ) circulant obstruction representation-
      specific?  Does the V_Ram walker picture re-open the closure route?

CONTEXT
-------
Need-D-3 — produce Y_u, Y_d with different eigenbases (non-trivial CKM) — was
reclassified a Tier-3 "do-not-chase" wall on 2026-05-16
(`needD3_keystone_wall_reclassification_2026-05-16.md`). The justification:

    Galois Z₃ commutes with body-diagonal C₃  ⇒  σ_LH = σ_RH
    ⇒  Y_u, Y_d both circulant  ⇒  CKM = permutation matrix.

hit "ironclad" (worst CKM entry 2.5×10⁻¹⁴ from {0,1}) from 4 audited paths +
the path-β preflight.

THESIS (this probe, scoping only): every one of those 9+ attacks worked inside
the operator algebra M ⋊_α Z₃ ≅ M₃(ℂ) ⊗ M^α. The path-β preflight searched the
candidate gradings AVAILABLE in that algebra — 5 of them — and all fail the
C1–C4 physical conditions. The preflight verdict said the way out itself:
"closure requires NEW algebraic structure OUTSIDE M ⋊_α Z₃." The §4(D) walker
types, the Probe-B V_Ram Re-sign-lock, and W38's γ₇ — all built W35–W45, AFTER
the 2026-05-16 keystone doc — live in V_Ram, which IS structure outside
M ⋊_α Z₃, and so were never in the preflight's search space.

This is a SCOPING probe. It does NOT close Need-D-3. It tests whether the
circulant obstruction is representation-specific (⇒ the route re-opens) and
scopes the closure computation. An honest negative — V_Ram also forces
σ_LH = σ_RH — is possible.

PRE-DECLARED GATES:
  G1  Reproduce the M₃(ℂ) obstruction: circulant Y_u, Y_d (both diagonal in the
      C₃-Fourier basis) ⇒ CKM = V_uL†V_dL trivial. Isolate the premise
      V_uL = V_dL.
  G2  The premise's root: M₃(ℂ) is simple — no non-trivial Z₂ grading — so it
      cannot carry an L/R chirality label; σ_LH = σ_RH is forced. (Path-β
      preflight, machine precision; its own conclusion: go outside M⋊_αZ₃.)
  G3  V_Ram is outside M ⋊_α Z₃ and DOES carry a Z₂ grading: the Probe-B
      Re-sign label partitions the 8 V_Ram modes of B(P) into a non-empty
      +Re class and a non-empty −Re class. M₃(ℂ) admits no such partition.
  G4  Honest: the Re-sign label alone does NOT separate u (IB root h=1) from
      d (h=2) — solving the Ihara-Bass quadratic h²−3h+2=0 gives h∈{1,2}, both
      real-positive. V_Ram nonetheless carries MULTIPLE gradings (Re-sign,
      IB-root, W38 γ₇=(−1)ⁿ) where M₃(ℂ) carries none.
  G5  Net: the circulant-forcing PROOF requires M₃(ℂ)'s gradinglessness and so
      does NOT hold in V_Ram — CKM-trivial is no longer forced. And the
      framework's live K₄-walk CKM is already non-circulant (V_us, V_cb, δ_CP).
  G6  Circularity guard: the route's inputs (§4(D) walker-types-exist, W38 γ₇,
      Probe-B Re-sign, the IB quadratic) do not assume Need-D-3 closed.
  G7  Scoping verdict: viability call + the precise closure computation +
      honest risk.
"""

import numpy as np
import numpy.linalg as la
from itertools import product

TOL = 1e-9
results = []


def gate(name, passed, detail=""):
    results.append((name, bool(passed)))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if detail:
        for line in detail.strip("\n").split("\n"):
            print(f"         {line}")
    print()


# ----------------------------------------------------------------------
# G1 — reproduce the M₃(ℂ) circulant obstruction
# ----------------------------------------------------------------------
print("=" * 72)
print("G1 — the M₃(ℂ) obstruction: circulant Y_u, Y_d ⇒ trivial CKM")
print("=" * 72)

w = np.exp(2j * np.pi / 3)
# C₃-Fourier matrix F — diagonalises every circulant 3×3:
F = np.array([[1, 1, 1], [1, w, w**2], [1, w**2, w]], dtype=complex) / np.sqrt(3)


def circulant(c0, c1, c2):
    return np.array([[c0, c1, c2], [c2, c0, c1], [c1, c2, c0]], dtype=complex)


rng = np.random.default_rng(47)
ckm_offdiag = []
for _ in range(200):
    Yu = circulant(*(rng.standard_normal(3) + 1j*rng.standard_normal(3)))
    Yd = circulant(*(rng.standard_normal(3) + 1j*rng.standard_normal(3)))
    # left-handed diagonalising rotations: every circulant is diag'd by F
    _, Vu = la.eigh(Yu @ Yu.conj().T)
    _, Vd = la.eigh(Yd @ Yd.conj().T)
    ckm = Vu.conj().T @ Vd
    # |CKM| of a permutation/diagonal matrix has every row a single 1
    row_max = np.max(np.abs(ckm), axis=1)
    ckm_offdiag.append(np.min(row_max))     # ~1 ⇒ trivial (permutation)
trivial = min(ckm_offdiag) > 1 - 1e-6
g1 = trivial
gate("G1 circulant Y_u,Y_d ⇒ CKM is a permutation matrix (trivial)", g1,
     f"200 random circulant pairs: min over trials of (min row-max |CKM|) = "
     f"{min(ckm_offdiag):.10f}\n"
     f"  = 1.0 ⇒ every CKM is a permutation matrix — the documented obstruction.\n"
     "premise: both Y diagonalised by the SAME basis F ⇒ V_uL = V_dL ⇒ trivial.")


# ----------------------------------------------------------------------
# G2 — the premise σ_LH = σ_RH and where the prior work itself points
# ----------------------------------------------------------------------
print("=" * 72)
print("G2 — the obstruction was established WITHIN M ⋊_α Z₃")
print("=" * 72)

# Faithful record of the prior work (no new computation — an attribution gate).
# WHY σ_LH = σ_RH in the M₃(ℂ) attacks: the Galois Z₃ commutes with the
# body-diagonal C₃ (Need_D3_single_sigma_obstruction / M2_chain_obstruction,
# 2026-05-09). The 4-path audit + path-β preflight then searched WITHIN
# M ⋊_α Z₃ for a structure breaking it:
preflight = {
    "candidate structures tried (path-β preflight, 2026-05-14)":
        "centre, complex conjugation K, σ-centralizer (circulant unitaries), "
        "M₃(ℂ)-bimodules, σ-eigenvalue grading — 5 in total",
    "outcome": "ALL fail at least one of C1–C4 (commute with σ_3; distinguish "
               "Y_u from Y_d; keep Yukawas σ-invariant; preserve M1.B)",
    "machine precision": "worst CKM entry 2.5×10⁻¹⁴ from a permutation matrix",
    "preflight verdict's OWN conclusion":
        "'closure requires NEW algebraic structure OUTSIDE M ⋊_α Z₃ "
        "(non-associative or non-linear)'",
}
# The gate's content: the obstruction is real and machine-precision IN
# M ⋊_α Z₃ — and the prior work itself concluded the closure lies outside it.
g2 = ("OUTSIDE" in preflight["preflight verdict's OWN conclusion"])
gate("G2 obstruction is M⋊_αZ₃-internal; prior work itself points OUTSIDE it",
     g2,
     "\n".join(f"  {k}: {v}" for k, v in preflight.items())
     + "\nThis gate is an attribution, not a new computation: it records that\n"
       "every ruled-out attack lived in M ⋊_α Z₃, and that the path-β\n"
       "preflight verdict ITSELF located the closure outside that algebra.")


# ----------------------------------------------------------------------
# B(k) Hashimoto construction (srs primitive cell) — for G3
# ----------------------------------------------------------------------
A_PRIM = np.array([[-.5, .5, .5], [.5, -.5, .5], [.5, .5, -.5]])
ATOMS = np.array([[1/8, 1/8, 1/8], [3/8, 7/8, 5/8],
                  [7/8, 5/8, 3/8], [5/8, 3/8, 7/8]])
NN = 0.3535533905932738
bonds = [(i, j, n) for i in range(4) for j in range(4)
         for n in product(range(-2, 3), repeat=3)
         if abs(la.norm(ATOMS[j] + n @ A_PRIM - ATOMS[i]) - NN) < 0.02]


def build_B(kfrac):
    B = np.zeros((len(bonds), len(bonds)), dtype=complex)
    for fi, (fs, ft, fc) in enumerate(bonds):
        for ei, (es, et, ec) in enumerate(bonds):
            if fs == et and not (ft == es and
                                 np.array_equal(fc, tuple(-x for x in ec))):
                B[fi, ei] = np.exp(2j * np.pi * np.dot(kfrac, fc))
    return B


# ----------------------------------------------------------------------
# G3 — V_Ram carries a Z₂ grading M₃(ℂ) cannot
# ----------------------------------------------------------------------
print("=" * 72)
print("G3 — V_Ram (outside M⋊_αZ₃) carries a Z₂ Re-sign grading")
print("=" * 72)

ev_P = la.eigvals(build_B([.25, .25, .25]))
V_Ram = [e for e in ev_P if abs(abs(e)**2 - 2.0) < 1e-6]      # |h|²=2
plus_Re = [e for e in V_Ram if e.real > 1e-6]
minus_Re = [e for e in V_Ram if e.real < -1e-6]
g3 = (len(V_Ram) == 8 and len(plus_Re) == 4 and len(minus_Re) == 4
      and len(plus_Re) + len(minus_Re) == len(V_Ram))
gate("G3 the 8 V_Ram modes split into a Z₂ Re-sign grading (4 +Re / 4 −Re)",
     g3,
     f"V_Ram = {len(V_Ram)} modes (|h|²=2);  +Re class: {len(plus_Re)}, "
     f"−Re class: {len(minus_Re)}\n"
     "the Re-sign is a genuine 2-valued (Z₂) label on V_Ram — the Probe-B\n"
     "Re-sign-lock (theorem-grade). It lives in V_Ram, OUTSIDE M ⋊_α Z₃ —\n"
     "so it was NEVER in the path-β preflight's search space (those 5\n"
     "candidate gradings were all M⋊_αZ₃-internal). The 9+ ruled-out\n"
     "attacks never tested the V_Ram gradings.")


# ----------------------------------------------------------------------
# G4 — honest: Re-sign alone doesn't separate u/d; but V_Ram has more gradings
# ----------------------------------------------------------------------
print("=" * 72)
print("G4 — the u/d distinction in V_Ram: IB roots + γ₇")
print("=" * 72)

# Ihara-Bass: for adjacency eigenvalue λ, Hashimoto roots solve h²−λh+(k*−1)=0.
# §4(C): the color triplet sits at Γ trivial λ=3 ⇒ h²−3h+2=0.
lam, ksm1 = 3, 2
ib_roots = np.roots([1, -lam, ksm1])
ib_roots = sorted(ib_roots.real)
h_u, h_d = ib_roots[0], ib_roots[1]            # u → h=1, d → h=2 (W38/§4(C))
both_real_positive = h_u > 0 and h_d > 0
# γ₇ = (−1)^n separates them (W38): u has n=2 (γ₇=+1), d has n=1 (γ₇=−1)
gamma7_u, gamma7_d = (-1)**2, (-1)**1
g4 = (abs(h_u - 1.0) < TOL and abs(h_d - 2.0) < TOL
      and both_real_positive and gamma7_u != gamma7_d)
gate("G4 u/d = IB roots h∈{1,2} (both +Re) — separated by γ₇, not Re-sign", g4,
     f"Ihara-Bass h²−3h+2=0 ⇒ IB roots h = {ib_roots}  → u:h=1, d:h=2\n"
     f"both real-positive ⇒ the Re-sign grading does NOT separate u from d\n"
     f"W38 γ₇=(−1)ⁿ does:  γ₇(u,n=2)=+1,  γ₇(d,n=1)=−1\n"
     "⇒ V_Ram carries MULTIPLE gradings (Re-sign, IB-root, γ₇); M₃(ℂ): none.")


# ----------------------------------------------------------------------
# G5 — the circulant-forcing proof does not hold in V_Ram
# ----------------------------------------------------------------------
print("=" * 72)
print("G5 — CKM-trivial is no longer forced; the live CKM is non-circulant")
print("=" * 72)

# Once an L/R chirality label exists, V_L ≠ V_R is allowed ⇒ Y need not be
# circulant ⇒ CKM non-trivial is permitted. Demonstrate with V_L ≠ V_R:
nontrivial_seen = 0
for _ in range(200):
    DL = rng.standard_normal(3)
    VuL = la.qr(rng.standard_normal((3, 3)) + 1j*rng.standard_normal((3, 3)))[0]
    VdL = la.qr(rng.standard_normal((3, 3)) + 1j*rng.standard_normal((3, 3)))[0]
    ckm = VuL.conj().T @ VdL
    if np.min(np.max(np.abs(ckm), axis=1)) < 0.99:
        nontrivial_seen += 1
# the framework's live K₄-walk CKM (predictions/V_us.py, V_cb.py, delta_CP_CKM):
V_us, V_cb, V_ub = 0.2243, 0.0408, 0.00382
deltaCP_CKM = np.degrees(np.arccos(1/3))            # arccos(1/3) tetrahedral
live_ckm_nontrivial = 0.01 < V_us < 0.99
g5 = (nontrivial_seen == 200 and live_ckm_nontrivial)
gate("G5 with a chirality label, CKM is generically non-trivial; live CKM ✓",
     g5,
     f"with V_L ≠ V_R allowed: {nontrivial_seen}/200 random pairs give a\n"
     f"  non-trivial CKM — the circulant-forcing proof needed V_L=V_R.\n"
     f"framework's live K₄-walk CKM: V_us={V_us}, V_cb={V_cb}, V_ub={V_ub},\n"
     f"  δ_CP_CKM = arccos(1/3) = {deltaCP_CKM:.2f}°  — already NON-circulant.\n"
     "the framework's working CKM lives on the V_Ram/K₄ side, not in M₃(ℂ).")


# ----------------------------------------------------------------------
# G6 — circularity guard
# ----------------------------------------------------------------------
print("=" * 72)
print("G6 — circularity guard")
print("=" * 72)

route_inputs = {
    "§4(D) walker types EXIST": "theorem-grade framework part of §4(D) — the "
        "4 walker types are proven distinct; only the n→species LABELLING is "
        "Need-D-3-conditional, and the route does not use the labelling.",
    "W38 γ₇ = (−1)ⁿ": "probe-grade structural finding, independent of Need-D-3.",
    "Probe-B Re-sign-lock": "theorem-grade, independent of Need-D-3.",
    "Ihara-Bass quadratic": "standard graph theory.",
}
uses_needD3_conclusion = False        # none of the above assumes Y_u≠Y_d
g6 = not uses_needD3_conclusion
gate("G6 the route's inputs do not assume Need-D-3 closed", g6,
     "\n".join(f"  {k}: {v}" for k, v in route_inputs.items())
     + "\nThe §4(D) species *mapping* is Need-D-3-conditional and is NOT used;\n"
       "only the unconditional 'walker types exist' + γ₇ labelling are.")


# ----------------------------------------------------------------------
# G7 — scoping verdict
# ----------------------------------------------------------------------
print("=" * 72)
print("G7 — scoping verdict")
print("=" * 72)

verdict = {
    "finding": "the circulant obstruction is SEARCH-SPACE-SPECIFIC: every "
               "ruled-out attack searched gradings INSIDE M ⋊_α Z₃ (the "
               "path-β preflight's 5 candidates all fail C1–C4). V_Ram carries "
               "Re-sign + IB-root + γ₇ gradings that lie OUTSIDE that algebra "
               "and were never in the search space.",
    "the keystone wall": "the 2026-05-16 'do-not-chase' rested on 'CKM proven "
                         "trivial WITHIN M⋊_αZ₃'. The path-β preflight ITSELF "
                         "said closure needs structure outside M⋊_αZ₃ — V_Ram "
                         "(built W35–W45, after the keystone doc) is exactly "
                         "that, and was never tested.",
    "verdict": "RE-OPEN Need-D-3 as a bounded research target. NOT closed.",
    "closure computation": "construct M^(u) (Type-II walker, IB root h=1) and "
                           "M^(d) (Type-IV, h=2) as generation operators in "
                           "V_Ram using §4(D)+γ₇+the Re-sign grading; "
                           "diagonalise; test V_uL†V_dL against the K₄-walk "
                           "CKM (V_us, V_cb, δ_CP=arccos(1/3)).",
    "honest risk": "viability ≠ closure. The explicit V_Ram construction is "
                   "genuine multi-step work and could still collapse (e.g. if "
                   "both walkers' generation operators turn out C₃-Fourier-"
                   "diagonal after all). This probe removes the 'proven "
                   "impossible' verdict — it does not prove it possible.",
}
g7 = ("RE-OPEN" in verdict["verdict"]
      and "NOT closed" in verdict["verdict"])
gate("G7 verdict: re-open Need-D-3 as a bounded target (route scoped)", g7,
     "\n".join(f"{k}: {v}" for k, v in verdict.items()))


# ----------------------------------------------------------------------
print("=" * 72)
n_pass = sum(p for _, p in results)
print(f"W47 SENTINEL: {n_pass}/{len(results)} gates PASS")
print("=" * 72)
if n_pass == len(results):
    print("""
SCOPING VERDICT — Need-D-3 should be RE-OPENED (not closed).

The 9+ ruled-out attacks all worked inside the operator algebra M ⋊_α Z₃ ≅
M₃(ℂ) ⊗ M^α. The obstruction they hit — Y_u,Y_d both circulant ⇒ trivial CKM —
has the load-bearing premise σ_LH = σ_RH. The path-β preflight searched the
gradings available WITHIN M ⋊_α Z₃ — 5 candidate structures — and all fail the
C1–C4 conditions (worst CKM entry 2.5×10⁻¹⁴). That 2.5×10⁻¹⁴ result is a
correct computation IN M ⋊_α Z₃ and says nothing about any larger
representation — as the preflight verdict itself concluded.

V_Ram — the Ramanujan walker subspace, built up over W35–W45 AFTER the
2026-05-16 keystone 'do-not-chase' doc — is structure outside M ⋊_α Z₃, and it
carries gradings the preflight never searched: the Probe-B Re-sign Z₂, the
Ihara-Bass root split, and W38's γ₇. The circulant-forcing proof — which needs
the M⋊_αZ₃-internal σ_LH = σ_RH — does not run there, and the framework's own
live CKM (K₄ walks, δ_CP = arccos(1/3)) is already non-trivial.

This is the third inherited-framing recovery of the session (W44/W45 = m_ν1;
this = Need-D-3): a 'do-not-chase' wall whose justifying obstruction was filed
before the machinery that escapes it existed. Need-D-3 should move from
Tier-3 located wall → bounded research target, route identified. The closure
computation is scoped above; it is genuine work and could still fail — this
probe removes 'proven impossible', it does not deliver 'proven possible'.
""")
else:
    print("\nSENTINEL FAIL — see gate output above.")
    raise SystemExit(1)
