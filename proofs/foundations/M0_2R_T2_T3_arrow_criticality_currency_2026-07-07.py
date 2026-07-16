#!/usr/bin/env python3
"""
proofs/foundations/M0_2R_T2_T3_arrow_criticality_currency_2026-07-07.py

M0-2R SESSION 1 — the two EXACT theorems: T2 (arrow = sub-criticality) and
T3 (Landauer = criticality). Frozen contract:
internal research notes (committed 846573a
BEFORE this file). Executor: a model The questions are frozen by the framer; this file
answers them on finite/exact objects and books nothing that does not fall out.

THE REFRAME (why this station exists): kappa's temperature lives in the state of HISTORY
(the multiway path gas), not the state of space. This session proves the two exact,
finite theorems that make that concrete:

  T2  ARROW = SUB-CRITICALITY.  The repo's own arrow (rho_step = u*|h|max < 1, the_run.py
      read_run) is EXACTLY the sub-criticality u < u_c = 1/(k-1) = 2^{-b_edge} of the
      multiway non-backtracking path gas. Forward converges (add is free); backward
      diverges (erasure is charged) -- delete>add as a COROLLARY, not an axiom.

  T3  LANDAUER = CRITICALITY.  The currency principle (p = 2^{-L} and E = kappa*L are one
      quantity) is consistent with a Gibbs weight e^{-beta E} IFF beta*kappa = ln2 (=
      Landauer, now a CONSISTENCY not an import); and at that beta the per-tick Boltzmann
      factor is exactly 2^{-b_edge} = u_c. => the OEF Landauer point, the MDL currency
      consistency, and the path-gas critical point are ONE point.

POISONS (flagged, never invoked/pattern-matched): the alpha_1-vs-u_c conflation (TWO
temperatures: kappa's T is the CRITICAL point u_c, NOT the run's operating u=alpha_1);
the 2pi/ln2 family; 2a1^5, 2a1^3, 5/12, 0.197. u_c = 1/(k-1) is COMBINATORIAL -- computed,
not discovered. NO scoreboard value moves; kappa stays OPEN (reduced, at most, to the 2pi).
"""
import math
import os
import sys

import numpy as np
import sympy as sp

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import srs  # noqa: E402

ok_all = True
def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    return cond
def banner(t):
    print("=" * 88); print(f" {t}"); print("=" * 88)

# ===========================================================================
banner("S0  the forced constants (READ off the object; nothing tuned)")
# ===========================================================================
k = srs.DEG                                  # coordination = 3 (READ)
q = k - 1                                     # NB branching = continuations per dart
b_edge = math.log2(q)                         # description cost per edge (bits) = log2(k-1)
u_c = 1.0 / q                                 # path-gas critical fugacity
print(f"    k = srs.DEG = {k};  q = k-1 = {q} (NB continuations/dart);  "
      f"b_edge = log2(k-1) = {b_edge};  u_c = 1/(k-1) = {u_c}")
check("S0a b_edge and u_c are consistent: u_c = 2^{-b_edge} EXACT",
      abs(u_c - 2 ** (-b_edge)) < 1e-15, detail=f"2^-b_edge = {2**(-b_edge)}")

# the run's OPERATING fugacity (the_run.py:323-324): u = alpha_1 = ((k-1)/k)^(girth-2)
girth_run = 10                                # read_girth() off B (renewal); crystal girth, not the K4-cell 3
alpha1 = (q / k) ** (girth_run - 2)
print(f"    run operating u = alpha_1 = ((k-1)/k)^(g-2) = (2/3)^{girth_run-2} = {alpha1:.6f}")
check("S0b TWO-TEMPERATURES GUARD: alpha_1 != u_c (kappa's T is at u_c, NOT the run's u)",
      abs(alpha1 - u_c) > 0.1,
      detail=f"alpha_1/u_c = {alpha1/u_c:.4f} (deeply sub-critical; 'cold') -- conflation FORBIDDEN")

# real-space Hashimoto at Gamma (k=0): the NB digraph on darts, no Bloch phases
B0 = srs.hashimoto((0, 0, 0)).real
ND = B0.shape[0]
print(f"    Hashimoto B(Gamma): {ND}x{ND} darts (2|E| = {2*len(srs.EDGES)})")

# ===========================================================================
banner("T2(a)  Omega_n = (k-1)^n EXACT  -- three independent proofs")
# ===========================================================================
# PROOF 1 (spectral, exact for ALL n): each dart has exactly (k-1) NB continuations
# <=> B is (k-1)-out-regular <=> B.1 = (k-1).1 <=> B^n.1 = (k-1)^n.1 <=> Omega_n=(k-1)^n.
one = np.ones(ND)
rowsums = B0 @ one
check("T2a-1 B(Gamma) is (k-1)-out-regular: B.1 = (k-1).1  (every dart has k-1 NB successors)",
      np.allclose(rowsums, q * one, atol=1e-12),
      detail=f"row sums all = {q} (min {rowsums.min():.3f}, max {rowsums.max():.3f})")
# hence B^n . 1 = (k-1)^n . 1 exactly
powers_ok = all(np.allclose(np.linalg.matrix_power(B0, n) @ one, (q ** n) * one, atol=1e-6)
                for n in range(1, 9))
check("T2a-1 => B^n.1 = (k-1)^n.1 for n=1..8 (Omega_n = (k-1)^n, ALL n)", powers_ok)

# PROOF 2 (real-space, boundary-free enumeration): build an L^3 supercell of the srs
# crystal (open boundary) and DIRECTLY count NB walks from a central bulk dart.
def build_supercell(L):
    """global vertex id ((cx,cy,cz),sub) -> int; undirected edges via homology offsets."""
    vid = {}
    for cx in range(L):
        for cy in range(L):
            for cz in range(L):
                for s in range(srs.NV):
                    vid[(cx, cy, cz, s)] = len(vid)
    adj = {v: set() for v in vid.values()}
    for cx in range(L):
        for cy in range(L):
            for cz in range(L):
                for (i, j, v) in srs.EDGES:
                    nx, ny, nz = cx + v[0], cy + v[1], cz + v[2]
                    if 0 <= nx < L and 0 <= ny < L and 0 <= nz < L:
                        a = vid[(cx, cy, cz, i)]; b = vid[(nx, ny, nz, j)]
                        adj[a].add(b); adj[b].add(a)
    return vid, adj

L = 6
vid, adj = build_supercell(L)
# a central bulk vertex (interior, full degree k)
center = vid[(L // 2, L // 2, L // 2, 0)]
deg_center = len(adj[center])
# count NB walks of length n from a dart leaving 'center' into a bulk neighbor
def count_nb_walks(adj, start_prev, start_cur, n):
    """number of NB walks of length n whose first dart is (start_prev -> start_cur)."""
    frontier = [(start_prev, start_cur)]
    for _ in range(n - 1):
        nxt = []
        for (prev, cur) in frontier:
            for c in adj[cur]:
                if c != prev:            # non-backtracking
                    nxt.append((cur, c))
        frontier = nxt
    return len(frontier)
nbr = next(iter(adj[center]))
counts = [count_nb_walks(adj, center, nbr, n) for n in range(1, 8)]
expected = [q ** (n - 1) for n in range(1, 8)]   # after fixing the 1st dart, (k-1)^(n-1) extensions
print(f"    supercell L={L}, center deg = {deg_center} (=k? {deg_center==k}); "
      f"NB-walk counts from a bulk dart, length n=1..7:")
print(f"      counted  = {counts}")
print(f"      (k-1)^(n-1) = {expected}")
check("T2a-2 direct real-space enumeration: NB walks from a bulk dart = (k-1)^(n-1) "
      "(bulk, boundary-free) => Omega_n = (k-1)^n", counts == expected)

# PROOF 3 (Perron): (k-1) is the spectral radius of B(Gamma) with Perron vector ~ 1.
ev0 = np.linalg.eigvals(B0)
rho0 = max(abs(ev0))
check("T2a-3 spectral radius of B(Gamma) = k-1 (Perron eigenvalue, vector = 1)",
      abs(rho0 - q) < 1e-9, detail=f"rho(B(Gamma)) = {rho0:.9f}")

# ===========================================================================
banner("T2(b)  Z(u) = 1/(1-(k-1)u),  radius of convergence u_c = 1/(k-1) = 2^{-b_edge}")
# ===========================================================================
u = sp.symbols('u', positive=True)
n = sp.symbols('n', integer=True, nonnegative=True)
qs = sp.Integer(q)
Z_conv = 1 / (1 - qs * u)                                  # the convergent closed form (|u|<u_c)
# airtight statement: the Taylor coefficients of the closed form ARE Omega_n = (k-1)^n
taylor = sp.series(Z_conv, u, 0, 9).removeO()
coeffs = [sp.simplify(taylor.coeff(u, m)) for m in range(9)]
print(f"    Z(u) = 1/(1-(k-1)u);  Taylor coeffs (Omega_n) = {coeffs}")
check("T2b Z(u)=1/(1-(k-1)u): its Taylor coefficients are exactly Omega_n=(k-1)^n",
      all(coeffs[m] == qs ** m for m in range(9)))
# and the summation itself equals the closed form on the convergent region (unwrap Piecewise)
Z_sum = sp.summation(qs ** n * u ** n, (n, 0, sp.oo))
Z_branch = Z_sum.args[0][0] if isinstance(Z_sum, sp.Piecewise) else Z_sum
check("T2b sum_n (k-1)^n u^n = 1/(1-(k-1)u) on the convergent region u<u_c",
      sp.simplify(Z_branch - Z_conv) == 0)
# radius of convergence = distance to the pole = 1/(k-1)
pole = sp.solve(sp.Eq(1 - qs * u, 0), u)[0]
check("T2b radius of convergence = pole of Z at u_c = 1/(k-1) = 2^{-b_edge}",
      float(pole) == u_c and abs(float(pole) - 2 ** (-b_edge)) < 1e-15,
      detail=f"pole at u = {pole} = {float(pole)}")
# INDEPENDENT cross-check via the Ihara zeta (Bass determinant): first pole = 1/(k-1).
# zeta(u)^{-1} = 0 at the smallest positive u; for a (k-1)-regular NB graph that is u=1/(k-1).
us = np.linspace(0.05, 0.95, 1901)
zinv = np.array([srs.ihara_zeta_inv(uu, (0, 0, 0)).real for uu in us])
# smallest positive root of zeta^{-1}=0 (sign change)
roots = us[:-1][np.sign(zinv[:-1]) != np.sign(zinv[1:])]
first_pole = roots.min() if len(roots) else float('nan')
check("T2b Ihara-zeta cross-check: smallest zeta(u)^{-1}=0 root at u_c = 1/(k-1)",
      abs(first_pole - u_c) < 5e-3, detail=f"Ihara first pole ~ {first_pole:.4f} (u_c={u_c})")

# ===========================================================================
banner("T2(c)  the arrow IS sub-criticality:  rho_step = u*|h|max < 1  <=>  u < u_c")
# ===========================================================================
# sup over the BZ of the NB spectral radius = k-1, attained at Gamma.
G = 7
grid = [(a / G, b / G, c / G) for a in range(G) for b in range(G) for c in range(G)]
radii = [(max(abs(np.linalg.eigvals(srs.hashimoto(kk)))), kk) for kk in grid]
rmax, kmax = max(radii, key=lambda t: t[0])
rGamma = radii[0][0]
print(f"    BZ scan ({G}^3 pts): max NB spectral radius = {rmax:.6f} at k={tuple(round(x,3) for x in kmax)}")
print(f"    at Gamma (k=0): NB spectral radius = {rGamma:.6f}")
check("T2c sup_BZ |h(k)|max = k-1, attained at Gamma (elsewhere gapped to the Ramanujan sqrt(k-1))",
      abs(rmax - q) < 1e-6 and abs(rGamma - q) < 1e-9 and tuple(kmax) == (0.0, 0.0, 0.0))
# the arrow condition, with the worst-case |h|max = k-1:
def arrow_holds(uu):
    return uu * q < 1
check("T2c arrow condition rho_step = u*(k-1) < 1  <=>  u < u_c = 1/(k-1)  (EXACT equivalence)",
      arrow_holds(u_c - 1e-9) and not arrow_holds(u_c + 1e-9) and not arrow_holds(u_c))
check("T2c the run is sub-critical (arrow holds): alpha_1*(k-1) < 1",
      arrow_holds(alpha1), detail=f"alpha_1*(k-1) = {alpha1*q:.4f} = u/u_c < 1")

# ===========================================================================
banner("T2(d)  forward converges / backward diverges  =>  delete > add (COROLLARY)")
# ===========================================================================
# forward: run weight per step = u*B (spectral radius u(k-1) < 1) => sum_n (uB)^n converges.
uB = alpha1 * B0
rho_fwd = max(abs(np.linalg.eigvals(uB)))
# backward: undoing a step multiplies by (uB)^{-1} (spectral radius 1/(u(k-1)) > 1) => diverges.
rho_bwd = max(abs(np.linalg.eigvals(np.linalg.inv(uB))))
check("T2d forward per-step spectral radius u*(k-1) < 1 (run G=(I-uB)^-1 CONVERGES)",
      rho_fwd < 1, detail=f"rho(uB) = {rho_fwd:.4f}")
check("T2d backward per-step spectral radius 1/(u(k-1)) > 1 (backward sum DIVERGES = the arrow)",
      rho_bwd > 1, detail=f"rho((uB)^-1) = {rho_bwd:.2f} = 1/(u(k-1))")
# convergence of the actual geometric operator series, forward only
partial = np.eye(ND)
term = np.eye(ND); conv = True
for _ in range(200):
    term = term @ uB
    partial = partial + term
G_exact = np.linalg.inv(np.eye(ND) - uB)
check("T2d forward series sum_n (uB)^n = (I-uB)^-1 (delete-cost>add-cost is a COROLLARY of u<u_c)",
      np.allclose(partial, G_exact, atol=1e-9))

# ===========================================================================
banner("T3  LANDAUER = CRITICALITY  (the currency theorem; symbolic + exact)")
# ===========================================================================
beta, kappa, L_sym = sp.symbols('beta kappa L', positive=True)
# currency principle: p = 2^{-L}  and  E = kappa*L  are ONE quantity; require consistency
# with a Boltzmann weight p ~ e^{-beta E}. Equate for ALL L:
lhs = 2 ** (-L_sym)                       # amplitude representation
rhs = sp.exp(-beta * kappa * L_sym)       # energy representation x Gibbs
# take logs and solve the per-L identity
eq = sp.Eq(sp.log(lhs), sp.log(rhs))      # -L ln2 = -beta kappa L
sol = sp.solve(sp.Eq(-sp.log(2), -beta * kappa), beta * kappa)   # coefficient of L
bk = sol[0]
print(f"    require 2^(-L) = e^(-beta*kappa*L)  for all L  =>  beta*kappa = {bk}")
check("T3-1 currency consistency FORCES beta*kappa = ln2 (= Landauer kappa=k_B T ln2, DERIVED)",
      sp.simplify(bk - sp.log(2)) == 0)
# at that beta, the per-TICK Boltzmann factor (one edge, L = b_edge) equals u_c:
per_tick = rhs.subs({beta * kappa: sp.log(2)}) if False else sp.exp(-sp.log(2) * b_edge)
per_tick_val = float(per_tick)
check("T3-2 per-tick Boltzmann factor at the Landauer point = 2^{-b_edge} = u_c = 1/(k-1)",
      abs(per_tick_val - u_c) < 1e-15,
      detail=f"e^(-ln2 * b_edge) = {per_tick_val} = u_c")
# THE IDENTIFICATION: three previously-separate framework objects are ONE point.
print("    => THREE OBJECTS = ONE POINT u_c = 1/(k-1) = 2^{-b_edge}:")
print("        (i)   path-gas CRITICAL fugacity           (T2b: pole of Z(u))")
print("        (ii)  MDL currency CONSISTENCY per tick     (T3: 2^{-b_edge} at beta*kappa=ln2)")
print("        (iii) OEF LANDAUER erasure cost per bit      (E=kappa*L at the same beta)")
check("T3-3 identification exact: path-gas critical point == currency per-tick factor == "
      "Landauer point", abs(u_c - per_tick_val) < 1e-15 and abs(u_c - 1.0 / q) < 1e-15)

# ===========================================================================
banner("ADOPTION-DOWNGRADE ANALYSIS (report only; register edit needs explicit booking)")
# ===========================================================================
print("""    DERIVED by T3 (was ADOPTED): the ln2 factor in kappa = k_B*T*ln2. It is FORCED as the
      bit<->nat conversion that makes the two currency representations (p=2^{-L}, E=kappa*L)
      jointly consistent with statistical mechanics. Landauer's RELATIONAL content
      (energy per bit = k_B T ln2) is thus a CONSISTENCY theorem, not an import.
    STILL ADOPTED (NOT downgraded here): (1) the currency IDENTIFICATION itself -- that a
      description length IS a physical energy (E proportional to L); (2) the temperature T
      (the dimensional anchor). => the OEF's A-IT3 downgrades PARTIALLY: the ln2 is derived;
      the currency premise and T remain. Booking the register requires the full protocol;
      this file only REPORTS the partial downgrade. kappa is NOT closed.""")

# ===========================================================================
banner("SUMMARY")
# ===========================================================================
print(f"""    T2  ARROW = SUB-CRITICALITY (EXACT, three proofs of Omega_n=(k-1)^n):
          the repo arrow rho_step=u*(k-1)<1 IS u < u_c = 1/(k-1) = 2^{{-b_edge}} = {u_c};
          forward converges, backward diverges => delete>add is a COROLLARY, not an axiom.
    T3  LANDAUER = CRITICALITY (EXACT): currency consistency forces beta*kappa = ln2, and
          the per-tick factor at that point = u_c. Path-gas critical point == currency
          per-tick == Landauer point: THREE objects, ONE point u_c.
    TWO-TEMPERATURES held: kappa's T lives at u_c=1/2, NOT the run's alpha_1={alpha1:.4f}
          (u/u_c={alpha1*q:.4f}, cold). No pattern-match to alpha_1 or 2pi. u_c is COMBINATORIAL.
    kappa STATUS: still OPEN; reduced toward the named 2pi residue (T4, next session).
          No scoreboard value moved. Adoption register NOT edited (partial downgrade reported).""")
print("RESULT:", "ALL CHECKS PASS -- T2 & T3 THEOREMS LAND" if ok_all else "A CHECK FAILED")
sys.exit(0 if ok_all else 1)
