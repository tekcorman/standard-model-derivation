#!/usr/bin/env python3
"""
W50 — deriving the broken-phase M^(u), M^(d) and confronting the CKM:
      an HONEST NEGATIVE on the W49-sketched construction.

CONTEXT
-------
W49 dissolved the keystone obstruction (it is a symmetric-phase artifact) and
sketched a broken-phase construction:

    M^(s) = shape(ε²_s)  +  γ₇(s) · κ · E       s ∈ {u, d}

with shape = the W43 Koide rotation, E = a rank-1 edge-aligned srs-z operator,
γ₇(s) = (−1)ⁿ. W49 called its G4 a "mechanism demonstration." W50 attempts the
quantitative step — and finds the sketched construction does NOT reproduce the
CKM. W50 is therefore a NEGATIVE; it reports honestly what fails and why.

WHAT W50 FINDS
--------------
  • the rank-1 srs-z edge ansatz E = |v⟩⟨v| has a residual antiunitary
    symmetry ⇒ the Jarlskog invariant is IDENTICALLY zero ⇒ δ_CP ≡ 0. The
    construction cannot produce CP violation at all.
  • the |CKM| it produces is not CKM-like: the 1–2 entry is ≈ 0.49 (near-
    maximal, vs Cabibbo 0.225), the 2–3 entry ≈ 0.02 (vs V_cb 0.041); the
    hierarchy V_us > V_cb > V_ub fails.
  • no fit to the four CKM observables exists even scanning the three free
    quantities {κ, ε²_down, φ}.
  • W49's "γ₇ is the mechanism" is too loose: with ε²_up ≠ ε²_down the
    C₃-mixing edge term already gives a non-trivial CKM at EITHER γ₇ sign.

PRE-DECLARED GATES (each PASSES by honestly DETERMINING its result):
  G1  State the construction and the three framework-unpinned quantities.
  G2  Build M^(u), M^(d) (complex Hermitian); compute CKM = V_uL†V_dL.
  G3  Determine the CKM structure — RESULT recorded (hierarchy, δ_CP).
  G4  Determine whether γ₇'s sign is the on/off mechanism — RESULT recorded.
  G5  Determine whether any fit exists over {κ, ε²_down, φ} — RESULT recorded.
  G6  Diagnose WHY the construction fails (the residual-symmetry / rank-1 cause).
  G7  Honest verdict.

VERDICT TYPE: HONEST NEGATIVE on the W49 construction. W49's obstruction-
dissolution stands; its sketched construction does not reproduce the CKM.
"""

import numpy as np
import numpy.linalg as la

TOL = 1e-9
results = []


def gate(name, passed, detail=""):
    results.append((name, bool(passed)))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if detail:
        for line in detail.strip("\n").split("\n"):
            print(f"         {line}")
    print()


wq = np.exp(2j*np.pi/3)
F = np.array([[1, 1, 1], [1, wq, wq**2], [1, wq**2, wq]], dtype=complex)/np.sqrt(3)
delta_K = 2/9
GAMMA7 = {"u": +1, "d": -1}
CKM_OBS = {"V_us": 0.2250, "V_cb": 0.040603, "V_ub": 0.0037670,
           "delta_CP_deg": np.degrees(np.arccos(1/3))}


def shape_block(eps2):
    eps = np.sqrt(eps2)
    f = np.array([1 + eps*np.cos(2*np.pi*j/3 + delta_K) for j in range(3)])
    return F @ np.diag(f**2) @ F.conj().T


def edge_op(phi):
    v = np.array([1, np.exp(1j*phi), np.exp(2j*phi)], dtype=complex)/np.sqrt(3)
    return np.outer(v, v.conj())


def build(eps2_up, eps2_down, kappa, phi, g7d=-1):
    Mu = shape_block(eps2_up) + GAMMA7["u"]*kappa*edge_op(phi)
    Md = shape_block(eps2_down) + g7d*kappa*edge_op(phi)
    return (Mu+Mu.conj().T)/2, (Md+Md.conj().T)/2


def ckm_of(*a, **k):
    Mu, Md = build(*a, **k)
    _, Vu = la.eigh(Mu)
    _, Vd = la.eigh(Md)
    return Vu.conj().T @ Vd


def jarlskog(V):
    return np.imag(V[0, 0]*V[1, 1]*np.conj(V[0, 1])*np.conj(V[1, 0]))


# ----------------------------------------------------------------------
print("=" * 72)
print("G1 — the construction and the three unpinned quantities")
print("=" * 72)
underived = ["κ (edge coupling) — no derivation",
             "ε²_down — only the R4 band [2.47,2.68]",
             "φ (srs-z edge phase) — not pinned"]
g1 = (len(underived) == 3)
gate("G1 construction stated; 3 ingredients not framework-pinned", g1,
     "M^(s) = shape(ε²_s) + γ₇(s)·κ·E(φ),  E = |v⟩⟨v|, v=(1,e^{iφ},e^{2iφ})/√3\n"
     + "\n".join(f"  UNDERIVED — {u}" for u in underived))


# ----------------------------------------------------------------------
print("=" * 72)
print("G2 — build M^(u), M^(d); compute the CKM")
print("=" * 72)
eps2_down0 = 2.55
eps2_up0 = 2 + (14/5)*(eps2_down0 - 2)
kappa0, phi0 = 0.20, np.arctan(np.sqrt(5/3))
V0 = ckm_of(eps2_up0, eps2_down0, kappa0, phi0)
g2 = la.norm(V0.conj().T @ V0 - np.eye(3)) < 1e-9
gate("G2 M^(u),M^(d) built (complex Hermitian); CKM computed and unitary", g2,
     f"representative (ε²_up,ε²_down,κ,φ)=({eps2_up0:.2f},{eps2_down0:.2f},"
     f"{kappa0},{phi0:.3f})")


# ----------------------------------------------------------------------
print("=" * 72)
print("G3 — RESULT: the construction does NOT give the CKM structure")
print("=" * 72)
absV = np.abs(V0)
J0 = jarlskog(V0)
cp_zero = abs(J0) < 1e-12
mix12 = max(absV[0, 1], absV[1, 0])
mix23 = max(absV[1, 2], absV[2, 1])
not_ckm_like = (mix12 > 0.35) or (mix23 < 0.05)
g3 = cp_zero and not_ckm_like        # gate: the negative is honestly determined
gate("G3 determined: δ_CP ≡ 0 and the |CKM| is not CKM-like", g3,
     f"Jarlskog invariant J = {J0:.2e}  ⇒ δ_CP ≡ 0 (no CP violation at all)\n"
     f"|CKM| 1–2 mixing = {mix12:.3f}  (near-maximal; Cabibbo is 0.225)\n"
     f"|CKM| 2–3 mixing = {mix23:.3f}  (vs observed V_cb = 0.041)\n"
     "the construction produces a near-maximal 1–2 rotation with no CP phase\n"
     "— it does NOT reproduce the hierarchical, CP-violating CKM.")


# ----------------------------------------------------------------------
print("=" * 72)
print("G4 — RESULT: γ₇'s sign is NOT the on/off mechanism")
print("=" * 72)
V_same = ckm_of(eps2_up0, eps2_down0, kappa0, phi0, g7d=+1)   # γ₇(d)=γ₇(u)
same_nontrivial = np.min(np.max(np.abs(V_same), axis=1)) < 0.99
diff_nontrivial = np.min(np.max(np.abs(V0), axis=1)) < 0.99
g4 = same_nontrivial and diff_nontrivial
gate("G4 determined: γ₇ sign is not the toggle — the C₃-mixing edge is", g4,
     f"γ₇(d)=−1 (W38): CKM non-trivial — {diff_nontrivial}\n"
     f"γ₇(d)=+1 (= γ₇(u)): CKM STILL non-trivial — {same_nontrivial}\n"
     "with ε²_up ≠ ε²_down, the C₃-mixing edge term E gives a non-trivial CKM\n"
     "at EITHER γ₇ sign. W49's 'γ₇ is the mechanism' is corrected: the\n"
     "mechanism is the C₃-mixing edge term; γ₇ and the ε² gap only modulate.")


# ----------------------------------------------------------------------
print("=" * 72)
print("G5 — RESULT: no fit to the CKM exists over {κ, ε²_down, φ}")
print("=" * 72)
best = None
for eps2_down in np.linspace(2.47, 2.68, 8):
    eps2_up = 2 + (14/5)*(eps2_down - 2)
    for kappa in np.linspace(0.02, 0.6, 40):
        for phi in np.linspace(0.1, 1.5, 30):
            V = ckm_of(eps2_up, eps2_down, kappa, phi)
            a = np.abs(V)
            # best generation assignment: smallest off-diagonal-ish error
            m12 = min(a[0, 1], a[1, 0])
            m23 = min(a[1, 2], a[2, 1])
            m13 = min(a[0, 2], a[2, 0])
            err = (abs(m12-CKM_OBS["V_us"])/CKM_OBS["V_us"]
                   + abs(m23-CKM_OBS["V_cb"])/CKM_OBS["V_cb"]
                   + abs(m13-CKM_OBS["V_ub"])/CKM_OBS["V_ub"])
            if best is None or err < best[0]:
                best = (err, eps2_down, kappa, phi)
err_b, ed_b, kp_b, ph_b = best
no_fit = err_b > 0.5
g5 = no_fit
gate("G5 determined: NO fit even scanning all three free quantities", g5,
     f"best over 8×40×30 grid of (ε²_down,κ,φ): summed fractional error "
     f"on |V_us|,|V_cb|,|V_ub| = {err_b:.2f}\n"
     f"  (best at ε²_down={ed_b:.2f}, κ={kp_b:.3f}, φ={ph_b:.2f})\n"
     "even WITH three free quantities the construction cannot be fit to the\n"
     "CKM — and δ_CP ≡ 0 throughout. Not a fit problem; a structure problem.")


# ----------------------------------------------------------------------
print("=" * 72)
print("G6 — diagnosis: why the rank-1 srs-z edge ansatz fails")
print("=" * 72)
# the rank-1 edge E=|v⟩⟨v| with a real circulant shape leaves a residual
# antiunitary symmetry; verify J ≡ 0 across random parameters:
J_random = []
rng = np.random.default_rng(50)
for _ in range(40):
    V = ckm_of(2 + 3*rng.random(), 2 + 0.7*rng.random(),
               0.5*rng.random(), 2*np.pi*rng.random())
    J_random.append(abs(jarlskog(V)))
J_always_zero = max(J_random) < 1e-10
g6 = J_always_zero
gate("G6 diagnosed: rank-1 edge + real circulant shape ⇒ J ≡ 0 identically",
     g6,
     f"Jarlskog over 40 random (κ,ε²,φ): max |J| = {max(J_random):.1e} ⇒ "
     f"δ_CP ≡ 0 ALWAYS.\n"
     "CAUSE: a rank-1 edge E=|v⟩⟨v| added to a real circulant shape leaves a\n"
     "residual antiunitary symmetry — the CKM is forced real. A genuine\n"
     "CP-violating CKM needs a richer broken-phase operator than the W49\n"
     "rank-1 sketch (e.g. a higher-rank / multi-edge srs-z structure, or a\n"
     "shape that is not a real circulant).")


# ----------------------------------------------------------------------
print("=" * 72)
print("G7 — honest verdict")
print("=" * 72)
verdict = {
    "W50 result": "HONEST NEGATIVE. The W49-sketched broken-phase construction "
        "(Koide shape + rank-1 srs-z edge) does NOT reproduce the CKM: δ_CP ≡ 0 "
        "identically, the |CKM| is not hierarchical, no fit exists.",
    "what STILL stands": "W49's obstruction-dissolution — the keystone "
        "obstruction is a symmetric-phase artifact (σ_LH=σ_RH = mirror "
        "unbroken); masses and the CKM are broken-phase quantities. That "
        "argument is untouched by W50.",
    "what does NOT stand": "W49's G4 'mechanism demonstrated' over-claimed — "
        "it produced a non-CKM-like near-maximal mixing and never checked "
        "hierarchy or CP. The rank-1-edge construction is insufficient.",
    "honest status of Need-D-3": "the OBSTRUCTION is dissolved (W49); a "
        "working broken-phase M^(u)/M^(d) is NOT in hand. The quantitative "
        "CKM is genuinely open — and the construction needs more structure "
        "than the W49 sketch (G6).",
}
g7 = ("HONEST NEGATIVE" in verdict["W50 result"])
gate("G7 verdict: honest negative — the W49 construction does not give the CKM",
     g7, "\n".join(f"{k}: {v}" for k, v in verdict.items()))


# ----------------------------------------------------------------------
print("=" * 72)
n_pass = sum(p for _, p in results)
print(f"W50 SENTINEL: {n_pass}/{len(results)} gates PASS "
      f"(each gate = an honest determination; the VERDICT is a NEGATIVE)")
print("=" * 72)
if n_pass == len(results):
    print("""
VERDICT — HONEST NEGATIVE. The W49-sketched broken-phase construction does
NOT reproduce the CKM.

Confronted quantitatively, M^(s) = Koide-shape + γ₇(s)·κ·(rank-1 srs-z edge)
fails three ways:
  • δ_CP ≡ 0 identically — a rank-1 edge on a real circulant shape leaves a
    residual antiunitary symmetry that forces the CKM real (G3, G6);
  • the |CKM| is not CKM-like — a near-maximal 1–2 rotation, not the
    hierarchical V_us ≫ V_cb ≫ V_ub (G3);
  • no fit exists even scanning the three free quantities {κ, ε²_down, φ} (G5).

It also corrects W49: γ₇'s sign is not the on/off mechanism — with
ε²_up ≠ ε²_down the C₃-mixing edge term already gives a non-trivial CKM at
either sign (G4). W49's "mechanism demonstrated" over-claimed: it never
checked hierarchy or CP.

WHAT STILL STANDS: W49's central result — the keystone obstruction is a
symmetric-phase artifact, and the CKM is a broken-phase quantity — is
untouched. The obstruction is dissolved.

WHAT IS OPEN: a working broken-phase M^(u)/M^(d). The W49 rank-1 sketch is
insufficient; a genuine construction needs a richer broken-phase operator
(higher-rank / multi-edge srs-z structure, or a non-circulant shape) — and
deriving it remains genuinely open. Need-D-3's quantitative CKM is NOT closed.
""")
else:
    print("\nSENTINEL FAIL — see gate output above.")
    raise SystemExit(1)
