#!/usr/bin/env python3
"""
proofs/foundations/majorana_phase_global_zeta_probe_2026-05-19.py

DECISIVE PROBE — does the ν_R Majorana phase (P35 α_21, P36 α_31) come out
of the GLOBAL C3-twisted Ihara/Bass spectral determinant, with NO loop-length
cutoff, instead of the LOCAL single-girth-loop identification M_R^(m)=h_m^g
(currently ADOPTED-NU-MAJ-PHASE, not derived)?

WHY THIS IS THE RIGHT TEST (not goal-seeking):
  The local construction (srs_hashimoto_seesaw_verify.py) uses ONE eigenvalue
  (the leading twisted Hashimoto eigenvalue h_m) raised to ONE length (the
  girth g=10).  majorana_M_R_waterfilling.py showed the loop-SUM representation
  Sigma_L w(L) h^L diverges at the P-point (Ramanujan saturation |h|^2=k*-1=2,
  every length contributes equal magnitude, phase drifts with the cutoff L_max).
  The reframe under test: that divergence is an artefact of the LOCAL
  (loop-length) chart of a DELOCALISED object; the cutoff-free object is the
  twisted Ihara/Bass determinant det(I - u A_H^(m) + (k*-1)u^2 I), well-defined
  on the critical circle by analytic continuation even where its Euler product
  diverges (b0_ruelle_ihara_dynamical_zeta_2026-05-17.py proves Ihara = Bass =
  dynamical zeta of the NB shift; vus_ihara_zeta_c3twisted.py already builds the
  C3-twisted operator and BZ-averages its log-det for V_us).

  This probe points that SAME global object at the Majorana phase.

ANTI-GOAL-SEEK — OUTCOMES PRE-DECLARED BEFORE ANY NUMBER IS COMPUTED:
  Local targets are computed live from h_omega=(sqrt3+i sqrt5)/2, g=10 (NOT a
  docstring).  Every global quantity is evaluated at STRUCTURALLY-PINNED points
  only:
     - the P-point  k=(1/4,1/4,1/4)         (the mass-spectrum point, A5(a))
     - the girth length g=10                 (srs girth — forced, not tuned)
     - the Ramanujan critical radius u=1/sqrt(k*-1)=1/sqrt2  (the Ihara
       functional-equation symmetry circle — forced, not tuned to a target)
     - the leading-pole reciprocal u=1/(k*-1)=1/2 (trivial-rep leading pole)
  No u is searched for the value 162.39.  Outcomes:

  OUTCOME-R (REPRODUCES): a cutoff-free global holonomy scalar lands within a
     few degrees of the live local target (alpha_21~162.39, alpha_31~324.78).
     => the global determinant DERIVES what h^g asserted; ADOPTED-NU-MAJ-PHASE
        becomes dischargeable.  NOT a closure by itself — still owes the
        channel->scalar map — but a genuine green light.

  OUTCOME-D (DIFFERENT-BUT-STABLE): global holonomy scalars are stable
     (BZ-resolution-independent, u-stable on the pinned circle) but do NOT
     match the local target.  => 162.39/324.78 were LOCAL-CHART ARTEFACTS;
     the framework's stated P35/P36 numbers are wrong-as-published; the
     global route yields a DIFFERENT (and now cutoff-free) prediction.

  OUTCOME-U (UNCONTROLLED): the determinant phase winds / fails to converge
     in BZ resolution / is discontinuous across the critical circle.
     => the global route inherits the delta-rho-Siegel-type negative
        (framework's actual object outside the analytic-control domain);
        honestly closed-negative, not another open-ended bust.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import find_bonds, ATOMS, A_PRIM, C3_PERM, N_ATOMS

np.set_printoptions(precision=6, suppress=True)

# ======================================================================
# 0.  LOCAL TARGET — computed LIVE (not from any docstring)
# ======================================================================
SQRT3, SQRT5 = np.sqrt(3.0), np.sqrt(5.0)
h_w  = (SQRT3 + 1j*SQRT5) / 2.0      # leading twisted Hashimoto eig, omega band
h_w2 = (-SQRT3 + 1j*SQRT5) / 2.0     # leading twisted Hashimoto eig, omega^2 band
g_girth = 10                          # srs girth (theorem-grade: g_girth.py)
k_star = 3

ALPHA21_LOCAL = np.degrees(np.angle(h_w ** g_girth)) % 360.0
ALPHA31_LOCAL = np.degrees(np.angle((h_w / h_w2) ** g_girth)) % 360.0
DELTA_LOCAL   = np.degrees(np.angle(h_w2 ** g_girth)) % 360.0

print("=" * 74)
print("  LOCAL h^g TARGET (live):")
print(f"    h_omega   = {h_w:.6f}   |h|^2 = {abs(h_w)**2:.6f} (= k*-1 = 2: Ramanujan)")
print(f"    arg(h_omega)              = {np.degrees(np.angle(h_w)):.6f} deg")
print(f"    alpha_21 = g*arg(h_omega) = {ALPHA21_LOCAL:.4f} deg   (P35)")
print(f"    alpha_31 = arg((h/h2)^g)  = {ALPHA31_LOCAL:.4f} deg   (P36)")
print(f"    (delta-band arg(h2^g)     = {DELTA_LOCAL:.4f} deg)")
print("=" * 74)

# ======================================================================
# 1.  C3 orbit / twist machinery  (verbatim from vus_ihara_zeta_c3twisted.py)
# ======================================================================
bonds_prim = find_bonds()
n_bonds = len(bonds_prim)             # 12
C3_CART = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
c3_atom = {i: int(np.argmax(C3_PERM[:, i])) for i in range(N_ATOMS)}

def bond_disp(src, tgt, cell):
    return (np.array(ATOMS[tgt])
            + cell[0]*np.array(A_PRIM[0]) + cell[1]*np.array(A_PRIM[1])
            + cell[2]*np.array(A_PRIM[2]) - np.array(ATOMS[src]))

prim_disps = [bond_disp(s, t, c) for s, t, c in bonds_prim]

def c3_of_bond(i):
    s0, _, _ = bonds_prim[i]
    new_src = c3_atom[s0]
    rot = C3_CART @ prim_disps[i]
    for j, (s, t, c) in enumerate(bonds_prim):
        if s == new_src and np.allclose(prim_disps[j], rot, atol=1e-8):
            return j
    raise ValueError(i)

c3_map = [c3_of_bond(i) for i in range(12)]
visited = [False]*12
orbits = []
for st in range(12):
    if visited[st]:
        continue
    b0, b1, b2 = st, c3_map[st], c3_map[c3_map[st]]
    assert c3_map[b2] == b0 and len({b0, b1, b2}) == 3
    orbits.append((b0, b1, b2))
    visited[b0] = visited[b1] = visited[b2] = True
assert len(orbits) == 4
orbit_pos = {}
for (b0, b1, b2) in orbits:
    orbit_pos[b0], orbit_pos[b1], orbit_pos[b2] = 0, 1, 2
pos_arr = np.array([orbit_pos[i] for i in range(n_bonds)])

def build_AH(k_frac):
    AH = np.zeros((n_bonds, n_bonds), dtype=complex)
    for j, (sj, tj, dcj) in enumerate(bonds_prim):
        for i, (si, ti, dci) in enumerate(bonds_prim):
            if sj != ti:
                continue
            dc_sum = tuple(int(dci[d]) + int(dcj[d]) for d in range(3))
            if tj == si and dc_sum == (0, 0, 0):
                continue
            AH[j, i] = np.exp(2j*np.pi*np.dot(k_frac, dci))
    return AH

def twist_W(omega):
    ph = omega ** pos_arr
    return np.outer(ph, ph.conj())          # W[j,i] = omega^(pos_j - pos_i)

OMEGA  = np.exp(2j*np.pi/3)
OMEGA2 = np.exp(4j*np.pi/3)
W1, W2 = twist_W(OMEGA), twist_W(OMEGA2)
P = np.array([0.25, 0.25, 0.25])            # the P-point (structurally pinned)
EYE = np.eye(n_bonds)

def AH_tw(k, W):
    return W * build_AH(k)

# ======================================================================
# 2.  CONSISTENCY — does the twisted operator at P actually carry h_omega?
#     (the global object must be built on the SAME input as the local one)
# ======================================================================
print("\n[2] CONSISTENCY: leading twisted eigenvalue at P vs local h_omega")
for tag, W in (("omega", W1), ("omega^2", W2), ("trivial", np.ones((12,12)))):
    ev = np.linalg.eigvals(AH_tw(P, W))
    lead = ev[np.argmax(np.abs(ev))]
    print(f"   {tag:8s}: leading |eig|={abs(lead):.6f}  arg={np.degrees(np.angle(lead)):8.3f} deg"
          f"   g*arg mod360 = {np.degrees(np.angle(lead**g_girth))%360:8.3f} deg")
print(f"   (local h_omega: |h|={abs(h_w):.6f}  arg={np.degrees(np.angle(h_w)):.3f}  "
      f"g*arg mod360={ALPHA21_LOCAL:.3f})")

# ======================================================================
# 3.  GLOBAL DIAGNOSTICS — all structurally pinned, no tunable cutoff
# ======================================================================
def safe_logdet(M):
    s, l = np.linalg.slogdet(M)
    if abs(s) < 0.5 or s.real < 0:
        return np.log(np.linalg.det(M) + 0j)
    return l + np.log(s)

u_crit = 1.0/np.sqrt(k_star-1)        # 1/sqrt2 : Ramanujan critical circle
u_pole = 1.0/(k_star-1)               # 1/2     : trivial-rep leading pole

print("\n[3a] GIRTH-LENGTH ALL-MODES HOLONOMY  arg Tr[(A_H^(m)(P))^g]")
print("     (same forced length g, but ALL 12 modes summed = waterfilling at")
print("      the girth scale; tests if subleading modes shift the leading-")
print("      eigenvalue phase the local construction ignores)")
A1g = np.linalg.matrix_power(AH_tw(P, W1), g_girth)
A2g = np.linalg.matrix_power(AH_tw(P, W2), g_girth)
tr1, tr2 = np.trace(A1g), np.trace(A2g)
a21_tr = np.degrees(np.angle(tr1)) % 360.0
a31_tr = np.degrees(np.angle(tr1/tr2)) % 360.0 if abs(tr2) > 1e-12 else float('nan')
print(f"     Tr[(A^omega)^g]   = {tr1:.4e}   arg = {a21_tr:8.3f} deg   "
      f"(local alpha_21 {ALPHA21_LOCAL:.3f})")
print(f"     Tr[(A^omega2)^g]  = {tr2:.4e}   arg(Tr1/Tr2) = {a31_tr:8.3f} deg   "
      f"(local alpha_31 {ALPHA31_LOCAL:.3f})")

print("\n[3b] CUTOFF-FREE Ihara/Bass det phase at P, structurally-pinned u")
for u_tag, u in (("u=1/2 (lead pole)", u_pole),
                 ("u=1/sqrt2 (Ramanujan crit circle)", u_crit)):
    d1 = np.linalg.det(EYE - u*AH_tw(P, W1) + (k_star-1)*u*u*EYE)
    d2 = np.linalg.det(EYE - u*AH_tw(P, W2) + (k_star-1)*u*u*EYE)
    d0 = np.linalg.det(EYE - u*build_AH(P)  + (k_star-1)*u*u*EYE)
    ph_w  = np.degrees(np.angle(d1/d0)) % 360.0
    ph_w2 = np.degrees(np.angle(d2/d0)) % 360.0
    ph_31 = np.degrees(np.angle(d1/d2)) % 360.0
    print(f"   {u_tag}:")
    print(f"     arg[D_omega/D_0]  = {ph_w:8.3f} deg   (local alpha_21 {ALPHA21_LOCAL:.3f})")
    print(f"     arg[D_omega2/D_0] = {ph_w2:8.3f} deg")
    print(f"     arg[D_omega/D_2]  = {ph_31:8.3f} deg   (local alpha_31 {ALPHA31_LOCAL:.3f})")

print("\n[3c] BZ-AVERAGED twisted zeta argument  Im log zeta_m  (framework's")
print("     actual zeta is BZ-averaged; cutoff-free; convergence-tested)")
for N_BZ in (12, 20, 30):
    acc = {1: 0j, 2: 0j, 0: 0j}
    for i1 in range(N_BZ):
        for i2 in range(N_BZ):
            for i3 in range(N_BZ):
                k = np.array([i1, i2, i3]) / N_BZ
                AH0 = build_AH(k)
                for key, A in ((1, W1*AH0), (2, W2*AH0), (0, AH0)):
                    M = EYE - u_crit*A + (k_star-1)*u_crit*u_crit*EYE
                    acc[key] += safe_logdet(M)
    nk = N_BZ**3
    Zw, Zw2, Z0 = -acc[1]/nk, -acc[2]/nk, -acc[0]/nk
    ph21 = np.degrees((Zw - Z0).imag) % 360.0          # holonomy phase, omega vs trivial
    ph31 = np.degrees((Zw - Zw2).imag) % 360.0         # omega vs omega^2
    print(f"   N_BZ={N_BZ:2d}: Im(log z_w - log z_0)  = {ph21:8.3f} deg"
          f"   Im(log z_w - log z_w2) = {ph31:8.3f} deg"
          f"   |Im z_w+Im z_w2|={abs((Zw.imag+Zw2.imag)):.2e}")

print("\n[3d] CONTROL CHECK — det phase winding along the pinned circle u=1/sqrt2")
print("     (stable across BZ-res & smooth in arg(u) => controlled; winds")
print("      without limit => UNCONTROLLED / Siegel-type negative)")
angs = []
for th in np.linspace(0, 2*np.pi, 9, endpoint=False):
    u = u_crit*np.exp(1j*th)
    d1 = np.linalg.det(EYE - u*AH_tw(P, W1) + (k_star-1)*u*u*EYE)
    d0 = np.linalg.det(EYE - u*build_AH(P)  + (k_star-1)*u*u*EYE)
    angs.append(np.degrees(np.angle(d1/d0)) % 360.0)
print("     arg[D_omega/D_0] around |u|=1/sqrt2: "
      + " ".join(f"{a:6.1f}" for a in angs))
print(f"     spread = {max(angs)-min(angs):.1f} deg")

# ======================================================================
# 4.  VERDICT  (mechanical classification against the pre-declared outcomes)
# ======================================================================
def near(x, t, tol=8.0):
    d = abs((x - t + 180) % 360 - 180)
    return d <= tol, d

print("\n" + "=" * 74)
print("  VERDICT (pre-declared outcomes; no number was searched for a target)")
print("=" * 74)
cands = {
    "Tr[(A^w)^g] arg          vs alpha_21": near(a21_tr, ALPHA21_LOCAL),
    "Tr1/Tr2 arg              vs alpha_31": near(a31_tr, ALPHA31_LOCAL),
}
hit = any(ok for ok, _ in cands.values())
for name, (ok, d) in cands.items():
    print(f"   {name}: {'MATCH' if ok else 'no'}  (off by {d:.2f} deg)")
print(f"\n   BZ-zeta holonomy stable across N_BZ?  -> see [3c] drift across rows")
print(f"   det-phase circle spread (from [3d])  -> {max(angs)-min(angs):.1f} deg")
print("""
   Read against the PRE-DECLARED outcomes:
     OUTCOME-R if a cutoff-free scalar matched a live local target above;
     OUTCOME-D if the global scalars are BZ-stable & circle-smooth but do
               NOT match (=> 162.39/324.78 are local-chart artefacts);
     OUTCOME-U if [3c] drifts with N_BZ or [3d] spread is large/winding
               (=> Siegel-type analytic-control negative).
   This probe ships NO number into predictions/ and changes NO ledger row;
   it only classifies the route.""")
print("=" * 74)
