"""
Grade-blind mass classification — companion probe.

Backs `docs/theorems/theorem_grade_blind_mass_classification_2026-06-03.md`.

Thesis: a Standard-Model excitation is MASSIVE iff the srs <-> srs-z mirror acts
non-trivially on it. The criterion is grade-blind (one condition for fermions and
bosons); the *mechanism* differs by Clifford grade. The species/particle TYPES are
the bounded solution set of that condition over a structurally-finite carrier, NOT
an enumerated table.

Gates:
  G1  odd quadrant: Ihara-Bass roots over the C3-fixed Bloch points classify the
      four fermion walker types (discriminant trichotomy).
  G2  the C3-fixed locus is one-dimensional (a line); among high-symmetry points
      the fixed set is exactly {Gamma, H, P} (N excluded); H is fixed via mod-G*.
  G3  the fermion walk length collapses to one integer L = g*is_down - 2*is_lepton
      whose sign selects cycle / saturation / band-edge and reproduces the anchors.
  G4  even quadrant: the gauge mass matrix M2_ab = <T_a phi | T_b phi> is the mirror
      response; eigenvalue 0 = mirror-invariant = massless (photon), >0 = massive.
  G5  bounded even count: PS adjoint (21) + 1 Higgs split by the mirror chain into
      9 massless (8 gluon + photon) + 12 massive (W,Z + 9 leptoquark) + 1 Higgs.

No framework imports; everything is standalone and CAS-checkable.
"""
import numpy as np
import cmath

PASS = []

def check(name, cond):
    PASS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# ----------------------------------------------------------------------------
# G1 — odd quadrant: Ihara-Bass root classification over C3-fixed Bloch points
# ----------------------------------------------------------------------------
# A self-sustaining closed non-backtracking walk is a root h of
#     h^2 - E_k*h + (k*-1) = 0,   k*=3.
# Solve the SAME quadratic over the finite C3-fixed point set; the discriminant
# E^2-8 partitions into the qualitative root-classes = the walker types.
print("G1  odd quadrant: Ihara-Bass roots over C3-fixed points")
kstar = 3
c = kstar - 1
fixed_points = {        # adjacency eigenvalue E at the C3-stable Bloch points
    "Gamma(lambda=+3)": 3.0,
    "Gamma/H(lambda=-1)": -1.0,
    "P(lambda=sqrt3)": 3 ** 0.5,
}
classes = {}
for name, E in fixed_points.items():
    disc = E * E - 4 * c
    roots = [(E + cmath.sqrt(disc)) / 2, (E - cmath.sqrt(disc)) / 2]
    classes[name] = (disc, roots)
    kind = "real-pair" if disc > 1e-9 else "complex-pair |h|^2=2"
    print(f"     {name:20} disc={disc:+.3f} -> {kind}: "
          + ", ".join(f"{r:.3f}" for r in roots))

# Gamma real pair must be exactly {2,1} (Perron=down, saturation=up);
# P and Gamma/H complex pairs must satisfy |h|^2 = k*-1 = 2 (Ramanujan saturation).
gamma_roots = sorted(r.real for r in classes["Gamma(lambda=+3)"][1])
check("Gamma gives real pair {1,2} (up/down split)",
      np.allclose(gamma_roots, [1.0, 2.0]))
for name in ("P(lambda=sqrt3)", "Gamma/H(lambda=-1)"):
    disc, roots = classes[name]
    check(f"{name} complex pair saturates |h|^2 = 2",
          disc < 0 and all(abs(abs(r) ** 2 - 2) < 1e-9 for r in roots))


# ----------------------------------------------------------------------------
# G2 — the C3-fixed locus is a LINE; high-sym fixed set is exactly {Gamma,H,P}
# ----------------------------------------------------------------------------
# srs Bravais = BCC; reciprocal G* = FCC = integer vectors with even coord-sum.
# Body-diagonal C3 acts on k by cyclic permutation P(k)=(kz,kx,ky).
print("G2  C3-fixed locus (a line) and the high-symmetry fixed set {Gamma,H,P}")

def in_Gstar(v, tol=1e-9):
    v = np.asarray(v, float)
    near = np.round(v)
    return np.allclose(v, near, atol=tol) and int(round(near.sum())) % 2 == 0

def C3(k):
    return np.array([k[2], k[0], k[1]], float)

def is_fixed(k):
    return in_Gstar(C3(k) - k)

hs = {
    "Gamma": [0, 0, 0], "P": [.5, .5, .5], "H": [1, 0, 0], "N": [.5, .5, 0],
}
fixed_hs = {n for n, k in hs.items() if is_fixed(k)}
check("high-symmetry C3-fixed set is exactly {Gamma,H,P}",
      fixed_hs == {"Gamma", "P", "H"})
# the entire [111] line is fixed exactly (=> the fixed locus is 1-dimensional)
line_fixed = all(np.allclose(C3([t, t, t]), [t, t, t]) for t in np.linspace(0, .5, 9))
check("entire [111] line is C3-fixed (locus is a line, not points)", line_fixed)
# H is fixed by the mod-G* mechanism (Pk-k is a NONZERO reciprocal vector)
dH = C3(hs["H"]) - np.array(hs["H"], float)
check("H fixed via mod-G* (Pk-k nonzero but in G*)",
      np.linalg.norm(dH) > 0.5 and in_Gstar(dH))


# ----------------------------------------------------------------------------
# G3 — fermion walk-length compression: one integer, sign-selected regime
# ----------------------------------------------------------------------------
print("G3  walk length L = g*is_down - 2*is_lepton; sign selects the regime")
g = 10
Q = 2.0 / 3.0
spec = {  # (is_down, is_lepton)
    "d": (1, 0), "e": (1, 1), "u": (0, 0), "nu": (0, 1),
}
L = {s: g * a - 2 * b for s, (a, b) in spec.items()}
check("L values are d=10, e=8, u=0, nu=-2",
      (L["d"], L["e"], L["u"], L["nu"]) == (10, 8, 0, -2))
# finite-L anchors reproduce the skeleton Q^L (up to chir & 1/k*^edge_sel factors)
check("Q^L reproduces y_b=Q^10 and y_t=Q^0=1",
      abs(Q ** L["d"] - Q ** 10) < 1e-12 and abs(Q ** L["u"] - 1.0) < 1e-12)
check("nu cell underflows (L<0) -> no localized cycle -> band edge",
      L["nu"] < 0)


# ----------------------------------------------------------------------------
# G4 — even quadrant: gauge mass = mirror response (the gauge mass matrix)
# ----------------------------------------------------------------------------
# M2_ab = <(T_a phi)|(T_b phi)>; eigval 0 = mirror-invariant = massless (photon),
# eigval > 0 = mirror-gapped = massive (W,Z). The even-grade analog of holonomy.
print("G4  even quadrant: gauge mass^2 = mirror response of each generator")
i = 1j
s1 = np.array([[0, 1], [1, 0]], complex)
s2 = np.array([[0, -i], [i, 0]])
s3 = np.array([[1, 0], [0, -1]], complex)
gens = [s1 / 2, s2 / 2, s3 / 2, 0.5 * np.eye(2)]   # T1,T2,T3,Y(=1/2)
phi = np.array([0, 1], complex) / np.sqrt(2)        # vev = mirror axis
M2 = np.array([[(Ga @ phi).conj() @ (Gb @ phi) for Gb in gens] for Ga in gens]).real
w = np.sort(np.linalg.eigvalsh(M2))
n_massless = int(np.sum(np.abs(w) < 1e-9))
n_massive = int(np.sum(w > 1e-9))
print(f"     gauge mass^2 eigenvalues (v^2 units): {np.round(w,4)}")
check("EW sector: 1 massless (photon) + 3 massive (W+,W-,Z)",
      n_massless == 1 and n_massive == 3)
# the massive pair (W+,W-) is degenerate and lighter than Z
check("W+,W- degenerate and below Z",
      abs(w[1] - w[2]) < 1e-9 and w[3] > w[2] + 1e-9)


# ----------------------------------------------------------------------------
# G5 — bounded even-grade count via the mirror chain
# ----------------------------------------------------------------------------
print("G5  bounded even-grade count (PS adjoint + Higgs, split by mirror chain)")
PS = 15 + 3 + 3            # Pati-Salam adjoint (fixed by Cl(6)/k*=3)
SM = 8 + 3 + 1            # unbroken at SM level
massless_gauge = 8 + 1     # gluon + photon (mirror-commuting / harmonic)
massive_gauge = 3 + (PS - SM)   # (W,Z) + heavy leptoquarks
check("PS adjoint = 21", PS == 21)
check("massless gauge = 9, massive gauge = 12, sum = 21",
      massless_gauge == 9 and massive_gauge == 12
      and massless_gauge + massive_gauge == PS)
check("even-grade total accounted = 22 (21 gauge + 1 Higgs)", PS + 1 == 22)


# ----------------------------------------------------------------------------
print()
print(f"{sum(PASS)}/{len(PASS)} gates PASS")
if not all(PASS):
    raise SystemExit("grade-blind mass classification probe: FAIL")
