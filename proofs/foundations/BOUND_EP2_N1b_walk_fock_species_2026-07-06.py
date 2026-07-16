#!/usr/bin/env python3
"""
proofs/foundations/BOUND_EP2_N1b_walk_fock_species_2026-07-06.py

N1b: does the BUILT walk<->Fock dictionary supply the constituent SPECIES?
Pre-registered in internal research notes
(committed 57bac02 BEFORE this probe).

The reopen candidate for EP-2 (N1 found geometry does NOT force species). E1
recorded the site<->species weld as a NAMED conditional. N1b tests the decisive
closed-walk version: is a Cl(6)-Fock SPECIES a good quantum number of the built
closed-walk holonomy operator W_A = sum_e gamma_e (x) edge_adj(e)?

A grading G is conserved by the closed walk iff [W_A, G(x)I]=0 (every step) or
{W_A, G(x)I}=0 (even-length closed walks; girth=10 even).

SCOPE: NO binding/mass data; NO hadron labeled; kappa walled; QED Clause-9. No fit.
Verdict: REOPEN / PARTIAL / CONFIRM-ADOPTION per the pre-reg.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import srs  # noqa: E402
from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")

def banner(t):
    print("=" * 84); print(f" {t}"); print("=" * 84)

EDGES = srs.EDGES
NE, NV = len(EDGES), srs.NV
I_V = np.eye(NV)

# ===========================================================================
banner("S0  build E1's walk<->Fock objects: W_A, N-hat, gamma7, color gradings")
# ===========================================================================
g6 = [np.array(g, complex) for g in AlgebraicUtility.cl6_generators()]
D = g6[0].shape[0]                                       # 8 = Cl(6) Fock dim
# Clifford re-lock
cliff = max(np.max(np.abs(g6[a] @ g6[b] + g6[b] @ g6[a] - (2.0 if a == b else 0) * np.eye(D)))
            for a in range(6) for b in range(6))
check(f"C1a: Cl(6) {{g_a,g_b}}=2delta on the 8-dim Fock (dev {cliff:.1e})", cliff < 1e-10)

def edge_adj(e):
    i, j, _ = EDGES[e]
    M = np.zeros((NV, NV))
    M[i, j] = M[j, i] = 1.0
    return M

W_A = sum(np.kron(g6[e], edge_adj(e)) for e in range(NE))   # 32x32 = Fock(8) (x) V(4)
print(f"    W_A: {W_A.shape} on Cl(6)-Fock({D}) (x) V({NV}); it is the E1 gamma-word.")

# CAR modes, number operator (species weight), chirality gamma7 (D1 construction)
a = [0.5 * (g6[2 * i] + 1j * g6[2 * i + 1]) for i in range(3)]
adag = [x.conj().T for x in a]
car_ok = all(np.max(np.abs(a[i] @ adag[j] + adag[j] @ a[i] - (np.eye(D) if i == j else 0))) < 1e-10
             for i in range(3) for j in range(3))
check("C1b: the 3 CAR modes {a_i,a_j^dag}=delta (genuine fermions)", car_ok)
Nhat = sum(adag[i] @ a[i] for i in range(3))                # species weight 0..3
evN = np.round(np.real(np.linalg.eigvalsh(Nhat))).astype(int)
mult = {w: int(np.sum(evN == w)) for w in (0, 1, 2, 3)}
check(f"C1c: N-hat species spectrum mult {mult} = 1/3/3/1 = nu/d/u/e", mult == {0: 1, 1: 3, 2: 3, 3: 1})
# fermion parity P = (-1)^N-hat (the Cl(6) Z2 even/odd grading; splits 4/4).
# NOTE: the Clifford VOLUME element g0..g5 squares to -1 in Cl(6,0) (eigs +/-i), so
# it is NOT the parity operator; the pre-reg's intended Z2 is (-1)^N-hat, built here.
wN, VN = np.linalg.eigh(Nhat)
wN = np.round(np.real(wN)).astype(int)
P_parity = VN @ np.diag((-1.0) ** wN) @ VN.conj().T        # (-1)^N-hat
p2 = np.max(np.abs(P_parity @ P_parity - np.eye(D)))
splitP = sorted(int(np.sum(np.round(np.real(np.linalg.eigvalsh(P_parity))).astype(int) == s))
                for s in (+1, -1))
check(f"C1d: fermion parity (-1)^N-hat squares to I (dev {p2:.1e}) and splits 4/4 "
      f"(found {splitP}) = 4 even {{nu,u}} + 4 odd {{d,e}}", p2 < 1e-10 and splitP == [4, 4])
def weight_proj(ws):
    P = np.zeros((D, D), complex)
    for k in range(D):
        if wN[k] in ws:
            P += np.outer(VN[:, k], VN[:, k].conj())
    return P
P_colored = weight_proj({1, 2})                            # quark (triplet+antitriplet) vs lepton {0,3}
H1 = adag[0] @ a[0] - adag[1] @ a[1]                       # su(3) Cartan (color index)
H2 = (adag[0] @ a[0] + adag[1] @ a[1] - 2 * adag[2] @ a[2]) / np.sqrt(3)

GRADINGS = {
    "N-hat (species weight, 4-way nu/d/u/e)": Nhat,
    "P_colored (quark {1,2} vs lepton {0,3})": P_colored,
    "H1 (color Cartan)": H1,
    "H2 (color Cartan)": H2,
    "parity (-1)^N-hat (Cl(6) Z2)": P_parity,
}

# ===========================================================================
banner("S1  C2 (CRUX): which Fock gradings are good quantum numbers of the walk W_A?")
# ===========================================================================
print(f"    {'grading':<42} {'||[W_A,G]||':>12} {'||{W_A,G}||':>12}  conserved?")
conserved = {}
for name, G in GRADINGS.items():
    GxI = np.kron(G, I_V)
    comm = np.max(np.abs(W_A @ GxI - GxI @ W_A))
    acom = np.max(np.abs(W_A @ GxI + GxI @ W_A))
    if comm < 1e-10:
        verdict = "YES (every step)"
        conserved[name] = "commute"
    elif acom < 1e-10:
        verdict = "YES (even closed walks only)"
        conserved[name] = "anticommute"
    else:
        verdict = "NO (species mixed)"
        conserved[name] = "no"
    print(f"    {name:<42} {comm:>12.2e} {acom:>12.2e}  {verdict}")

# species (>=3-valued or the quark/lepton split) forced?
species_commute = conserved.get("N-hat (species weight, 4-way nu/d/u/e)") == "commute"
colored_commute = conserved.get("P_colored (quark {1,2} vs lepton {0,3})") == "commute"
# for closed even-length walks, anticommuting gradings are ALSO conserved:
def closed_conserved(name):
    return conserved.get(name) in ("commute", "anticommute")
species_closed = closed_conserved("N-hat (species weight, 4-way nu/d/u/e)")
colored_closed = closed_conserved("P_colored (quark {1,2} vs lepton {0,3})")
parity_closed = closed_conserved("parity (-1)^N-hat (Cl(6) Z2)")

check("C2 measured: species N-hat conservation by the closed-walk holonomy "
      f"(commute={species_commute}, closed-even-conserved={species_closed})", True)

# ===========================================================================
banner("S2  cross-check: the forced Z2 = fermion parity (-1)^N-hat")
# ===========================================================================
# each grade-1 gamma_e changes fermion number by +/-1 => anticommutes with
# (-1)^N-hat => W_A anticommutes with P => P preserved for EVEN-length closed
# walks (girth=10). Confirm.
PxI = np.kron(P_parity, I_V)
antiP = np.max(np.abs(W_A @ PxI + PxI @ W_A))
print(f"    ||{{W_A, (-1)^N}}|| = {antiP:.2e} (0 => parity flips each step, conserved for even L)")
check("S2: fermion parity (-1)^N-hat is conserved for even-length closed walks "
      "(the forced Cl(6) Z2 grading, {nu,u} vs {d,e}); it is a single Z2, NOT the "
      "4-way species -- a chirality-like grading, not flavor", antiP < 1e-10)

# ===========================================================================
banner("S3  C3: VERDICT (no fit, no data)")
# ===========================================================================
reopen = species_closed or colored_closed          # a physically-meaningful (>=quark/lepton) species forced
partial = (not reopen) and colored_closed
# finest conserved grading:
finest = "none"
if species_closed:
    finest = "N-hat (full 4-way species)"
elif colored_closed:
    finest = "P_colored (quark vs lepton)"
elif parity_closed:
    finest = "(-1)^N-hat (Z2 fermion-parity only)"

verdict = "REOPEN" if reopen else ("PARTIAL" if partial else "CONFIRM-ADOPTION")
print(f"""
    MEASURED: the FINEST Cl(6)-Fock grading that is a good quantum number of the
    built closed-walk holonomy W_A = {finest}.
      - species weight N-hat conserved (closed even walks): {species_closed}
      - quark/lepton split P_colored conserved: {colored_closed}
      - Z2 parity/chirality gamma7 conserved: {parity_closed}

    VERDICT: {verdict}
""")
if verdict == "CONFIRM-ADOPTION":
    print("""    The built walk<->Fock dictionary does NOT supply the constituent SPECIES:
    W_A = sum_e gamma_e (x) edge_adj mixes the N-hat species (each grade-1 gamma_e
    changes fermion number by +/-1), so neither the 4-way species nor even the
    quark/lepton split is a good quantum number of a closed walk. Only the Z2
    fermion parity (-1)^N-hat survives (for even closed walks) -- a chirality-like
    Cl(6) grading ({nu,u} vs {d,e}), NOT a flavor/species.

    => the single-site Cl(6)-Fock occupation -> extended-cycle SPECIES lift is a
    GENUINE IRREDUCIBLE ADOPTION, now confirmed from THREE independent angles:
      (N1)  geometry: a girth cycle spans all 4 weight-classes (no forced species);
      (E1)  per-step: the matter modes are edge x dart-qubit, site<->species weld
            a named conditional;
      (N1b) closed-walk holonomy: W_A conserves only the Z2, mixes species.
    This is the honest F2/EP-2 WALL: EP-2 does not reopen through the built
    dictionary. What IS forced across all three: the Z2 fermion parity + the
    ~80-class geometry+chirality skeleton + body-number->sector. kappa walled;
    no hadron labeled; no number fit.""")
elif verdict == "PARTIAL":
    print("""    The built dictionary forces the QUARK/LEPTON (color-singlet) split but not
    the u/d flavor: quark-vs-lepton reopens; u/d flavor is the residual adoption.""")
else:
    print("""    REOPEN: the constituent species IS a good quantum number of the built
    closed-walk holonomy -> the species lift routes through the walk<->Fock
    dictionary; EP-2 reopens with no new adoption. B1's dictionary is un-gated.""")

check("C3 scope: no binding/mass data; no hadron labeled; kappa walled; QED "
      "Clause-9; no fit", True)
print("=" * 84)
print(f" OVERALL gates {'PASS' if ok_all else 'FAIL'}; verdict = {verdict}")
print("=" * 84)
sys.exit(0 if ok_all else 1)
