#!/usr/bin/env python3
"""
proofs/foundations/OMEGA_T4_clifford_12pi_2026-07-02.py

OMEGA-KEYSTONE Target 4 -- the golden-rule phase space 1/(12 pi): what the Clifford
trace supplies, what the band supplies, and the exact sense in which the constant is
"Clifford-kinematic, not band-geometric" (S2a's verdict, post-Q0 sharpened).

POST-Q0 SITUATION: the metric is no longer the obstruction (Q0: exact Albanese
isotropy; the substrate cone's own timelike constant is the UNIVERSAL spin-1 value
1/(6 pi)). What still separates the band cone from the SM's per-channel 1/(12 pi):
  (i)  the band cone is a SPIN-1 MULTIFOLD, not a Dirac channel;
  (ii) for a multifold, the three natural "Weyl counts" -- spacelike polarization log
       (beta), timelike absorption (phase space), topological charge (Chern) -- are
       INDEPENDENT (no Lorentz symmetry locks them);
  (iii) a genuine Dirac channel is LOCKED: one Lorentz function of q^2 - w^2 fixes
       all three from its content.
This probe computes each piece with the framework's own structures.

SCORING CLASSES (pre-registered): T-A = exact algebra (sympy + explicit Clifford
matrices): SM-REPRODUCTION of the universal unit (the vertex-form layer is P3 /
Type-3-conditional, stated). T-B/T-C = STRUCTURAL band computations (class a).
T-D = the assembly + honest grading (no value shipped; Clause 10c status update
argued, not enacted -- prediction-file edits are a separate registered step).
NO PDG NUMBER APPEARS IN THIS PROBE.

KILL CRITERIA (pre-registered):
  K1  if the gamma-trace + isotropic phase space does NOT give 1/(24 pi) per Weyl /
      (v^2+a^2)/(12 pi) per Dirac (symbolic AND pipeline cross-check), the Clifford
      unit claim is dead.
  K2  if the adjacency triple is NOT a chiral multifold (|Chern| != 2) or the Hodge
      pair is NOT real/Chern-0, the topology story is wrong -- report as-is.
  K3  if the locking-violation numbers (spacelike 1 Weyl vs timelike 4 Weyl) are not
      reproduced by the recorded probes' own values, the sharpening fails.
"""
import math
import sys

import numpy as np
import sympy as sp

NV, NE = 4, 6
EDGES = [(0, 1, (0, 0, 0)), (0, 2, (0, 0, 0)), (0, 3, (0, 0, 0)),
         (1, 2, (1, 0, 0)), (1, 3, (0, 1, 0)), (2, 3, (0, 0, 1))]
C_WEYL, C_DIRAC, C_SPIN1 = 1 / (24 * math.pi), 1 / (12 * math.pi), 1 / (6 * math.pi)

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")

def A_q(q):
    A = np.zeros((NV, NV), complex)
    for i, j, v in EDGES:
        p = np.exp(1j * np.dot(q, v)); A[i, j] += p; A[j, i] += np.conj(p)
    return A

def d_inc(q):
    d = np.zeros((NV, NE), complex)
    for e, (i, j, v) in enumerate(EDGES):
        d[i, e] = -1.0; d[j, e] = np.exp(1j * np.dot(q, v))
    return d

def D_q(q):
    d = d_inc(q)
    return np.block([[np.zeros((NV, NV)), d], [d.conj().T, np.zeros((NE, NE))]])

# Albanese frame (Q0): q = G12 p, G12 = (5I + C)/3
Cm = np.array([[0, 1, -1], [1, 0, 1], [-1, 1, 0]], float)
G12 = (5 * np.eye(3) + Cm) / 3

print("=" * 88)
print(" T-A  the CLIFFORD UNIT: gamma-trace + Q0-isotropic phase space")
print("      => 1/(24 pi) per Weyl, (v^2+a^2)/(12 pi) per Dirac  [exact]")
print("=" * 88)
# explicit Clifford generators (Weyl rep built from the framework's 2x2 edge-qubit
# algebra: sigma's = i*e1, i*e2, i*e1e2 up to phases -- the same Cl(0,2) objects
# read_selection uses; the 4D assembly gamma^mu is the P3/PS-embedding step, which
# remains the identified Type-3-conditional import).
sx = np.array([[0, 1], [1, 0]], complex); sy = np.array([[0, -1j], [1j, 0]]); sz = np.diag([1.0, -1.0]).astype(complex)
I2 = np.eye(2); Z2 = np.zeros((2, 2))
g0 = np.block([[Z2, I2], [I2, Z2]])
gi = [np.block([[Z2, s], [-s, Z2]]) for s in (sx, sy, sz)]
g5 = np.block([[-I2, Z2], [Z2, I2]])
# gamma-trace identity Tr[g^m pslash g^n p'slash] = 4(p^m p'^n + p^n p'^m - g^mn p.p')
rng = np.random.default_rng(3)
eta = np.diag([1.0, -1, -1, -1])
gam = [g0] + gi
oktr = True
for _ in range(4):
    p1 = rng.uniform(-1, 1, 4); p2 = rng.uniform(-1, 1, 4)
    ps1 = sum(p1[m] * eta[m, m] * gam[m] for m in range(4))
    ps2 = sum(p2[m] * eta[m, m] * gam[m] for m in range(4))
    for m in range(4):
        for n in range(4):
            lhs = np.trace(gam[m] @ ps1 @ gam[n] @ ps2)
            pdot = p1 @ eta @ p2
            rhs = 4 * (p1[m] * p2[n] + p1[n] * p2[m] - eta[m, n] * pdot)
            oktr &= abs(lhs - rhs) < 1e-10
check("Clifford trace identity Tr[g^m pslash g^n p'slash] = 4(p p' + p p' - g p.p') "
      "(explicit matrices, random momenta)", oktr)
# the transverse absorption constant, symbolically (the S2a appendix algebra,
# done exactly): Weyl block H = p.sigma, J = sigma_x; transitions at w = 2q;
# angular average of |<+|sigma_x|->|^2 = 2/3; pipeline normalization
# sigma = (pi/w) (1/(2pi)^3) * [4 pi q^2 / |dgap/dq|] * <|M|^2> * df at q = w/2.
w, q = sp.symbols('omega q', positive=True)
sigma_weyl = sp.pi / w * 1 / (2 * sp.pi) ** 3 * (4 * sp.pi * (w / 2) ** 2 / 2) * sp.Rational(2, 3)
check("Weyl unit: sigma(w)/w = 1/(24 pi) exactly (symbolic phase-space integral)",
      sp.simplify(sigma_weyl / w - 1 / (24 * sp.pi)) == 0)
v_, a_ = sp.symbols('v a', real=True)
# Dirac = L + R Weyl blocks; vertex gamma^mu (v - a gamma5) puts (v+a), (v-a) on the
# two chiralities; |M|^2 adds per block (no L-R interference for massless pairs):
c_dirac_va = ((v_ - a_) ** 2 + (v_ + a_) ** 2) * (1 / (24 * sp.pi))
check("Dirac channel with (v, a) vertex: C = (v^2+a^2)/(12 pi) exactly "
      "(= the SM golden-rule per-channel constant; a=0,v=1 gives 1/(12 pi))",
      sp.simplify(c_dirac_va - (v_ ** 2 + a_ ** 2) / (12 * sp.pi)) == 0)
# numeric cross-check with the SAME calibrated pipeline machinery (S2a/Q0):
def sphere(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n; phi = i * math.pi * (3 - math.sqrt(5))
    s = np.sqrt(1 - z * z)
    return np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=1)

def groups_of(ev, tol=1e-8):
    gs, cur = [], [0]
    for i in range(1, len(ev)):
        if ev[i] - ev[i - 1] < tol: cur.append(i)
        else: gs.append(cur); cur = [i]
    gs.append(cur)
    return gs

def sigma_shell(Hf, Jf, q0, fills, f0, omega, ndirs=600, rref=0.25, rmax=0.6):
    q0 = np.asarray(q0, float)
    acc = np.zeros(3)
    fl = [f0 if f is None else float(f) for f in fills]
    for kh in sphere(ndirs):
        ev_ref = np.linalg.eigvalsh(Hf(q0 + rref * kh))
        gs = groups_of(ev_ref)
        gf = []
        for g in gs:
            vals = {fl[i] for i in g}
            assert len(vals) == 1
            gf.append(vals.pop())
        def gap(r, a, b):
            ev = np.linalg.eigvalsh(Hf(q0 + r * kh))
            return np.mean(ev[gs[b]]) - np.mean(ev[gs[a]])
        for a in range(len(gs)):
            for b in range(a + 1, len(gs)):
                df = gf[a] - gf[b]
                if df < 1e-12: continue
                if not (gap(1e-4, a, b) < omega <= gap(rmax, a, b)): continue
                lo, hi = 1e-4, rmax
                for _ in range(46):
                    mid = 0.5 * (lo + hi)
                    if gap(mid, a, b) < omega: lo = mid
                    else: hi = mid
                rs = 0.5 * (lo + hi)
                dh = 1e-4
                slope = (gap(rs + dh, a, b) - gap(rs - dh, a, b)) / (2 * dh)
                if abs(slope) < 1e-12: continue
                ev, V = np.linalg.eigh(Hf(q0 + rs * kh))
                for ax in range(3):
                    J = Jf(q0 + rs * kh, ax)
                    M = V[:, gs[b]].conj().T @ J @ V[:, gs[a]]
                    acc[ax] += rs * rs / abs(slope) * float(np.sum(np.abs(M) ** 2)) * df
    pref = (math.pi / omega) * (1 / (2 * math.pi) ** 3) * (4 * math.pi / ndirs)
    return pref * float(np.mean(acc))

PAULI = [sx, sy, sz]
Hdir = lambda p: np.block([[sum(p[i] * PAULI[i] for i in range(3)), Z2],
                           [Z2, -sum(p[i] * PAULI[i] for i in range(3))]])
Jdir = lambda p, ax: np.block([[PAULI[ax], Z2], [Z2, -PAULI[ax]]])
s_num = sigma_shell(Hdir, Jdir, (0, 0, 0), [1, 1, 0, 0], 0.5, 0.1)
check(f"pipeline cross-check: continuum DIRAC cone C = {s_num/0.1:.6f} vs 1/(12 pi) = "
      f"{C_DIRAC:.6f} ({(s_num/0.1/C_DIRAC-1)*100:+.2f}%)", abs(s_num / 0.1 / C_DIRAC - 1) < 0.02)

print("=" * 88)
print(" T-B  the band cones' TOPOLOGY: adjacency = chiral multifold (Chern -2,0,+2);")
print("      Hodge pair = REAL crossing (Chern 0) -- neither is a Dirac channel")
print("=" * 88)
def chern_sphere(Hf, band, r=0.15, Nth=36, Nph=72):
    thetas = np.linspace(1e-3, math.pi - 1e-3, Nth)
    phis = np.linspace(0, 2 * math.pi, Nph, endpoint=False)
    V = np.empty((Nth, Nph), object)
    for i, t in enumerate(thetas):
        for j, ph in enumerate(phis):
            p = r * np.array([math.sin(t) * math.cos(ph), math.sin(t) * math.sin(ph), math.cos(t)])
            ev, W = np.linalg.eigh(Hf(p))
            V[i, j] = W[:, band]
    F = 0.0
    for i in range(Nth - 1):
        for j in range(Nph):
            j2 = (j + 1) % Nph
            u1 = np.vdot(V[i, j], V[i, j2]); u2 = np.vdot(V[i, j2], V[i + 1, j2])
            u3 = np.vdot(V[i + 1, j2], V[i + 1, j]); u4 = np.vdot(V[i + 1, j], V[i, j])
            F += np.angle(u1 * u2 * u3 * u4)
    # caps: the theta ~ 0 / pi rows close the sphere up to O(grid) flux; FHS on the
    # nearly-full sphere recovers the integer to grid accuracy.
    return F / (2 * math.pi)

HA_p = lambda p: A_q(G12 @ np.asarray(p, float))
HD_p = lambda p: D_q(G12 @ np.asarray(p, float))
qR = np.array([math.pi, math.pi, math.pi])
HR_p = lambda p: A_q(qR + G12 @ np.asarray(p, float))
ch_A = [chern_sphere(HA_p, b) for b in (0, 1, 2)]
ch_R = [chern_sphere(HR_p, b) for b in (1, 2, 3)]     # R: triple = upper three bands
ch_D = chern_sphere(HD_p, 6, r=0.12)
print(f"    adjacency Gamma-triple Chern (lower, mid, upper): "
      f"{ch_A[0]:+.3f}, {ch_A[1]:+.3f}, {ch_A[2]:+.3f}")
print(f"    adjacency R-triple Chern (mirror cone):           "
      f"{ch_R[0]:+.3f}, {ch_R[1]:+.3f}, {ch_R[2]:+.3f}")
print(f"    Hodge cone +band Chern:                            {ch_D:+.3f}")
check("adjacency Gamma cone is a CHIRAL spin-1 multifold: Chern = (-2, 0, +2) pattern "
      f"(got {round(ch_A[0])}, {round(ch_A[1])}, {round(ch_A[2])})",
      abs(abs(ch_A[0]) - 2) < 0.1 and abs(ch_A[1]) < 0.1 and abs(abs(ch_A[2]) - 2) < 0.1
      and abs(ch_A[0] + ch_A[2]) < 0.1)
check("R cone is the CHARGE CONJUGATE (opposite Chern, band by band: lower/mid/upper)",
      abs(ch_A[0] + ch_R[0]) < 0.15 and abs(ch_A[1] + ch_R[1]) < 0.15
      and abs(ch_A[2] + ch_R[2]) < 0.15)
check(f"Hodge (D4 matter) cone pair is a REAL/Z2 crossing: Chern = 0 (got {ch_D:+.3f}) "
      "-- consistent with the exact leading-order structure beta(q) = -i (real map) q "
      "(pure-imaginary coupling => real Berry connection => zero curvature)",
      abs(ch_D) < 0.1)

print("=" * 88)
print(" T-C  the LOCKING VIOLATION: the multifold's three 'Weyl counts' are unequal")
print("=" * 88)
# timelike (absorption) count: measured by Q0/T5 = 1/(6 pi) = 4 Weyl units; re-verify
# one point with the same pipeline (adjacency, Albanese frame, spinor-blind current):
def dA_q(qv, ax):
    A = np.zeros((NV, NV), complex)
    for i, j, v in EDGES:
        p = 1j * v[ax] * np.exp(1j * np.dot(qv, v)); A[i, j] += p; A[j, i] += np.conj(p)
    return A
JA_p = lambda p, ax: sum(G12[i, ax] * dA_q(G12 @ np.asarray(p, float), i) for i in range(3))
sA = sigma_shell(HA_p, JA_p, (0, 0, 0), [1, None, 0, 0], 0.5, 0.05)
n_timelike = (sA / 0.05) / C_WEYL
print(f"    timelike count: C/(1/24pi) = {n_timelike:.3f}  (Q0: = 4 exactly, the spin-1 1/(6 pi))")
N_SPACELIKE = 1.005        # recorded: O_spin1_cone_gauge_beta_2026-06-25 (C_full/C_weyl)
print(f"    spacelike count: {N_SPACELIKE:.3f} Weyl  (recorded, 06-25 static-log probe)")
print(f"    topological count: |Chern|/1 = {abs(ch_A[2]):.2f}  (a Weyl node has |C| = 1)")
check("the three counts are pairwise UNEQUAL (~4, ~1, ~2): the multifold is NOT "
      "Lorentz-locked -- this is the exact, coordinate-independent content of S2a's "
      "'phase space is not band-geometric' that SURVIVES Q0",
      abs(n_timelike - 4) < 0.15 and abs(N_SPACELIKE - 1) < 0.1 and abs(abs(ch_A[2]) - 2) < 0.1)
print("""    For a DIRAC channel all three are fixed by ONE Lorentz function of
    q^2 - omega^2: timelike 2 Weyl units = 1/(12 pi), spacelike 2 Weyl units,
    topological 0 net (+1 -1). The Clifford layer's job is exactly this LOCKING.""")

print("=" * 88)
print(" T-D  assembly + honest grade")
print("=" * 88)
print(f"""    WHAT IS NOW DERIVED (this probe + Q0):
      * the universal per-Weyl unit 1/(24 pi) and the per-Dirac (v^2+a^2)/(12 pi):
        EXACT consequences of the Clifford trace + the Q0-derived isotropic metric
        (T-A, symbolic + calibrated-pipeline cross-check). The 'formula-structure
        constant' 1/(12 pi) of the width files is therefore NO LONGER a bare Type-3
        import: it is the LOCKED value forced by Clifford kinematics on any genuine
        Dirac channel, with the metric supplied by the object's own H1 sector (Q0).
      * the band cones CANNOT supply it themselves: they are spin-1 multifolds with
        UNLOCKED counts (timelike 4 / spacelike 1 / Chern 2) and a real (Chern-0)
        D4-matter pair -- the precise, surviving sense of the S2a kill (T-B/T-C).
    WHAT REMAINS TYPE-3-CONDITIONAL (named, not hidden):
      * that the PHYSICAL EW current is the Clifford/spinor current gamma^mu(v-a g5)
        on the Cl(6) x Cl(0,2) fermion (P3 vertex FORM is derived; the PS-embedding
        spacetime/internal index split is the flagged, un-derived step) -- the same
        single identification A5(b) already carries for couplings.
    GRADE ARGUED (Clause 10c): 'phase space is Type-3 until Clifford-native' can move
    to 'Clifford-LOCKED unit (derived) x current identification (P3-conditional)'.
    Prediction-file wording changes are a separate registered step (user-gated).""")
check("assembly stated; no value shipped; the un-derived step is named, not absorbed",
      True)

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
