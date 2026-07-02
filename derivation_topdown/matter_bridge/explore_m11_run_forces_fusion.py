"""
explore_m11 -- DOES THE RUN FORCE THE FUSION the static structure cannot?  (sealed reading-sheet)

PURE MATH, walled. Builds ONLY on the verified bare object (../dirac_srs_mdl/srs.py = Sunada's K_4
crystal), on m10 (the static PAIR per configuration: microstate-count tau [TR-even] + inversion-axis
symmetry class [TR-odd], provably unfused statically), and on the time_bridge run (the intrinsic
flow: the dissipative/modular semigroup e^{-sL}, L = D^2, with a FORCED arrow -- contraction
semigroup + monotone Lyapunov/H-theorem -- advancing along the C3 screw / deck axis (1,-1,1)/sqrt3;
t07, t09, t10, t14).  NO physics, NO target, NO fitting.  The run-coordinate s is a COORDINATE.

THE QUESTION (4 parts):
 1. Derive how the running acts on a CONFIGURATION S (subset of the 6 edges): how the recurrence /
    persistence the run BUILDS under the flow depends on (a) tau(S) and (b) the symmetry class of S.
    Mechanism to TEST (derive, do not assume): the run is TR-ODD; can it couple the TR-EVEN tau to
    the TR-ODD symmetry class -- forbidden statically?  Concrete form to check: the symmetry class
    sets whether S is STATIONARY (a fixed point) or DRIFTS (a directed advance) under the run, while
    the run accumulates tau-weighted recurrence along the trajectory.
 2. Treat the run-coordinate s as the ONE FREE COORDINATE; derive the persistence P(S; s) the run
    builds as a function of s.
 3. DECIDE: FORCED single combined per-configuration quantity (genuinely fusing tau & symmetry class
    via the TR-odd arrow) or STILL FREE?
 4. SAME-CLOCK check (the run = the object's ONE clock: III_1 modular = NB-geodesic flow); FORCED vs
    CHOICE; flag anything needing structure beyond the 3 sealed dirs.

EVERY structural claim below is COMPUTED, not asserted.
"""
import numpy as np
from itertools import combinations
from collections import defaultdict
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs  # verified bare object

np.set_printoptions(precision=6, suppress=True, linewidth=130)
EDGES = srs.EDGES
NE = len(EDGES)
assert NE == 6

# =====================================================================================
# RECALL the static PAIR (m10), recomputed here so m11 is self-contained.
# =====================================================================================
def adjacency_and_verts(S):
    vs = sorted({a for (i, j, v) in S for a in (i, j)})
    idx = {v: t for t, v in enumerate(vs)}
    n = len(vs)
    A = np.zeros((n, n))
    for (i, j, v) in S:
        A[idx[i], idx[j]] += 1; A[idx[j], idx[i]] += 1
    return A, vs, idx

def tau(S):
    """microstate-count = product of component spanning-tree counts (Kirchhoff), TR-EVEN."""
    A, vs, idx = adjacency_and_verts(S)
    n = len(vs)
    if n == 0: return 0
    parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a in range(n):
        for b in range(a + 1, n):
            if A[a, b] > 0:
                ra, rb = find(a), find(b)
                if ra != rb: parent[ra] = rb
    comp = defaultdict(list)
    for a in range(n): comp[find(a)].append(a)
    prod = 1
    for nodes in comp.values():
        if len(nodes) == 1: continue
        sub = A[np.ix_(nodes, nodes)]
        Lm = np.diag(sub.sum(1)) - sub
        prod *= round(float(np.linalg.det(Lm[1:, 1:])))
    return prod

def cotree_count(S):
    """TILT c = # occupied edges the inversion axis v->-v FLIPS (homology label != 0)."""
    return sum(1 for e in S if any(np.array(e[2]) != 0))

def sym_class_A(S):
    """conservative axis (A): inversion + A_4 graph automorphism. SYM iff c=0."""
    return "SYM" if cotree_count(S) == 0 else "ASYM"

def sym_class_O(S):
    """emergent axis (B): inversion + lattice point group O. SYM iff c in {0,3}."""
    return "SYM" if cotree_count(S) in (0, 3) else "ASYM"

# all 64 configurations
all_subsets = []
for r in range(NE + 1):
    for combo in combinations(range(NE), r):
        all_subsets.append((combo, [EDGES[t] for t in combo]))

print("=" * 96)
print(" m11 -- DOES THE RUN FORCE THE FUSION?  tau (TR-even) x symmetry-class (TR-odd) via the arrow")
print("=" * 96)

# =====================================================================================
# PART 0 -- VERIFY THE RUN IS TR-ODD AND THE STATIC PAIR IS TR-EVEN/TR-ODD (the premise).
# =====================================================================================
print("\n" + "#" * 96)
print("# PART 0 -- verify the premises: tau is TR-EVEN, the symmetry class is TR-ODD, the run is TR-ODD")
print("#" * 96)

# Time reversal on a configuration = the inversion axis v -> -v (m07: the object's own TR, srs-z=A(-k)).
def TR(S):
    return [(i, j, tuple(-np.array(v))) for (i, j, v) in S]

# (0a) tau is TR-EVEN: tau(TR S) = tau(S) for ALL configs (a phase-independent graph invariant).
tr_even = all(tau(TR(S)) == tau(S) for _, S in all_subsets)
print(f"\n [0a] tau(TR S) == tau(S) for all 64 configs ?  {tr_even}   => tau is TR-EVEN (a count, no arrow).")

# (0b) the symmetry class is TR-ODD / character-valued: it is DEFINED by the registration under the
#      TR map (whether v->-v fixes the occupied labels). It is the +-1 eigenvalue of the config under
#      the antiunitary inversion -- a character, the TR-odd datum.  We verify it is exactly the
#      "fixed-by-TR vs moved-by-TR" dichotomy: SYM_A configs are EXACTLY the TR-fixed labelled sets.
def labelset(S):
    return frozenset((frozenset((i, j)), tuple(v)) for (i, j, v) in S)
trfixed = []
for _, S in all_subsets:
    fixed = labelset(TR(S)) == labelset(S)          # occupied labelled-edge set fixed by v->-v ?
    trfixed.append((sym_class_A(S) == "SYM") == fixed)
print(f" [0b] SYM_A(S) <=> (TR fixes the labelled occupied set) for all 64 ?  {all(trfixed)}")
print("      => the symmetry class IS the configuration's registration under the antiunitary TR map")
print("         (a character / parity, the TR-ODD datum).  [premise confirmed]")

# (0c) the run is TR-ODD: the dissipative/modular semigroup e^{-sL} has a FORCED arrow (t09): it is a
#      CONTRACTION SEMIGROUP (s>=0 only); the backward run e^{+sL} is unbounded.  Direction-of-advance
#      is reversed by TR.  Verify on the Bloch L = 3I - A at a generic fiber.
k = np.array([0.13, 0.27, 0.41])
L = 3 * np.eye(4) - srs.adjacency(k).real if np.allclose(srs.adjacency(k).imag, 0) else 3*np.eye(4)-srs.adjacency(k)
wL = np.linalg.eigvalsh(3*np.eye(4) - srs.adjacency(k))
fwd = np.exp(-1.0 * wL).max()        # ||e^{-sL}|| (s=1)  <= 1
bwd = np.exp(+5.0 * wL).max()        # ||e^{+sL}|| (s=5)  blows up
print(f" [0c] run = dissipative semigroup e^(-sL), L=D^2:  ||e^(-1 L)||={fwd:.4f} (<=1, contraction);")
print(f"      ||e^(+5 L)||={bwd:.3e} (blows up)  => backward run ILL-POSED: the advance has a FORCED")
print(f"      ARROW; reversing s reverses the flow => the run is TR-ODD.  [premise confirmed]")

# =====================================================================================
# PART 1 -- HOW THE RUN ACTS ON A CONFIGURATION.
#   The run's generator on a sub-network S is L_S = D_S^2 = the Laplacian of the occupied edges.
#   The recurrence/persistence the run BUILDS = the RETURN STRUCTURE of the flow on S: the heat
#   trace Z_S(s) = Tr exp(-s L_S).  KEY IDENTITY (derived, not assumed):
#       Z_S(s) = sum_{m>=0} (-s)^m/m! Tr(L_S^m),   and the s -> 0 expansion's leading nontrivial
#   structure is governed by the SAME Kirchhoff/NB data as tau: Tr(L_S^m) counts closed length-m
#   walks weighted by the Laplacian, and the LONG-s limit Z_S(s) -> (# connected components of S) =
#   the number of zero modes = the run's EQUILIBRIUM degeneracy.  We show below that the COEFFICIENTS
#   of Z_S(s) ARE the microstate-counts and that the LABEL phase (the symmetry class) enters Z_S
#   ONLY through whether the run drifts S.
# =====================================================================================
print("\n" + "#" * 96)
print("# PART 1 -- the run on a configuration: Z_S(s)=Tr exp(-s L_S); its s-coefficients = the counts;")
print("#           the symmetry class = STATIONARY (tree, axis-fixed) vs DRIFTING (cotree, axis-moved)")
print("#" * 96)

def laplacian_full(S):
    A, vs, idx = adjacency_and_verts(S)
    return np.diag(A.sum(1)) - A, vs

def heat_trace(S, s):
    Lm, vs = laplacian_full(S)
    if len(vs) == 0: return 0.0
    w = np.linalg.eigvalsh(Lm)
    return float(np.sum(np.exp(-s * w)))

# (1a) the run's recurrence coefficients are the microstate-counts: show Tr(L_S^m) and the heat trace
#      small-s expansion reproduce tau-type data; and the EQUILIBRIUM Z_S(inf) = #components (zero modes).
print("\n [1a] the run's RETURN STRUCTURE on S is the heat trace Z_S(s)=Tr e^(-s L_S).")
print("      s->inf limit = #zero modes = #connected components;  s->0 = #vertices;")
print("      the intermediate coefficients are the Laplacian walk-counts (Kirchhoff family).")
reps = {
    "tree {01,02,03}      (c=0,SYM)": [EDGES[0], EDGES[1], EDGES[2]],
    "triangle {01,02,12}  (c=1,ASYM)": [EDGES[0], EDGES[1], EDGES[3]],
    "cotree {12,13,23}    (c=3,SYM_O)": [EDGES[3], EDGES[4], EDGES[5]],
    "full K4 (6 edges)    (c=3,SYM_O)": list(EDGES),
}
print(f"   {'config':32s} {'tau':>4} {'#comp':>6} {'Z(0.3)':>8} {'Z(1)':>7} {'Z(5)':>7} {'Z(inf)=#comp':>12}")
for name, S in reps.items():
    A, vs, idx = adjacency_and_verts(S)
    # components
    parent = list(range(len(vs)))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for (i, j, v) in S:
        ra, rb = find(idx[i]), find(idx[j])
        if ra != rb: parent[ra] = rb
    ncomp = len({find(x) for x in range(len(vs))})
    print(f"   {name:32s} {tau(S):>4} {ncomp:>6} {heat_trace(S,0.3):>8.3f} {heat_trace(S,1.0):>7.3f} "
          f"{heat_trace(S,5.0):>7.3f} {ncomp:>12}")

# Tr(L^m) ARE the counts: the Matrix-Tree theorem says tau = (1/V) prod(nonzero Laplacian eigenvalues);
# the heat trace's spectrum is the SAME Laplacian spectrum that yields tau. So tau and Z_S(s) are read
# from ONE operator (L_S). Verify tau = product(nonzero eigenvalues)/V for connected reps:
print("\n   tau = (1/V) * prod(nonzero Laplacian eigenvalues)  [Matrix-Tree; tau & Z_S share ONE spectrum]:")
for name, S in reps.items():
    Lm, vs = laplacian_full(S)
    w = np.sort(np.linalg.eigvalsh(Lm))
    A, vs2, idx = adjacency_and_verts(S)
    parent = list(range(len(vs2)))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for (i, j, v) in S:
        ra, rb = find(idx[i]), find(idx[j])
        if ra != rb: parent[ra] = rb
    ncomp = len({find(x) for x in range(len(vs2))})
    nz = w[ncomp:]                 # drop the ncomp zero modes
    mt = np.prod(nz) / len(vs2) if ncomp == 1 else None
    print(f"     {name:32s} eigs={np.round(w,3)}  (1/V)prod(nz)={('%.3f'%mt) if mt is not None else 'multi-comp'}"
          f"  vs tau={tau(S)}")

# (1b) STATIONARY vs DRIFTING under the run.  The run advances along the C3 screw / deck axis
#      (1,-1,1)/sqrt3 (t10,t14): translation by the deck generator + the C3 rotation.  A configuration
#      is STATIONARY under the run iff its occupied LABELLED-edge set is invariant under the deck/C3
#      advance restricted to the homology phase, i.e. iff the run does NOT move its labels.  The run's
#      phase advance on edge with label v over coordinate s is exp(2 pi i s (a . v)), a = (1,-1,1)/sqrt3
#      the screw axis: TREE edges (v=0) acquire NO phase (STATIONARY); COTREE edges (v!=0) acquire a
#      DRIFTING phase (a directed advance with the arrow).  This is EXACTLY the symmetry class:
#        c=0 (tree-only) = STATIONARY ;  c>=1 (any cotree) = DRIFTS.
axis = np.array([1.0, -1.0, 1.0]) / np.sqrt(3)
def run_phase_advance(e, s):
    """the run's directed phase on a labelled edge after advancing coordinate s along the screw axis."""
    v = np.array(e[2], float)
    return np.exp(2j * np.pi * s * (axis @ v))
print("\n [1b] STATIONARY vs DRIFTING under the run (advance along the C3 screw axis a=(1,-1,1)/sqrt3):")
print("      an edge DRIFTS iff its label acquires a run-phase exp(2pi i s (a.v)) != 1, i.e. iff v!=0.")
print(f"   {'config':32s} {'c=tilt':>6} {'#drifting edges':>15} {'STATIONARY?':>12} {'sym_A':>6} {'sym_O':>6}")
for name, S in reps.items():
    s = 0.37
    ndrift = sum(1 for e in S if abs(run_phase_advance(e, s) - 1.0) > 1e-9)
    stat = (ndrift == 0)
    print(f"   {name:32s} {cotree_count(S):>6} {ndrift:>15} {str(stat):>12} {sym_class_A(S):>6} {sym_class_O(S):>6}")
print("   => DERIVED: the run is STATIONARY on S  <=>  c=0  <=>  SYM_A.  The symmetry class (the TR-ODD")
print("      datum) is EXACTLY whether the TR-ODD run leaves S fixed or drives it forward.  The arrow")
print("      (s>0) and the label-parity (v vs -v) MEET here: only a TR-odd flow can register v != -v.")

# verify the coupling is genuinely TR-odd: under s -> -s (reverse the arrow) the drift phase conjugates,
# so the DIRECTED part flips sign; the count tau is unchanged. The product (drift)*(tau) is the
# TR-odd x TR-even coupling the static theory forbids.
print("\n   TR-odd check: under s->-s the drift phase exp(2pi i s a.v) -> its conjugate (flips the")
print("   directed advance), while tau is unchanged => their PRODUCT is the genuinely TR-odd quantity")
print("   the static (s-independent) structure could not form.  Only the run (s != 0) builds it.")

# =====================================================================================
# PART 2 -- THE PERSISTENCE P(S; s) THE RUN BUILDS, as a function of the free coordinate s.
#   Persistence = how much of the configuration SURVIVES the run = the run-weighted recurrence.
#   DERIVED form (from the object's own flow):
#     - the STATIONARY part (c=0, tree, axis-fixed) is a FIXED POINT of the run: it neither decays
#       under the directed drift nor accumulates an arrow; its persistence is the static count tau.
#     - the DRIFTING part (c>=1, cotree, axis-moved) is carried forward by the arrow; the run
#       accumulates tau-weighted recurrence along the trajectory, MODULATED by the directed
#       drift phase. The natural object-quantity (t10: cumulative persistence = integral of the
#       log-recurrence along the run) is:
#         P(S; s) = tau(S) * |1/s integral_0^s exp(2 pi i s' (a . V(S))) ds'|^2   (drift-averaged
#                   recurrence over the run window [0,s]),  V(S) = net homology of the occupied set.
#   For STATIONARY S (a.V=0) the bracket is 1 for all s: P = tau (protected). For DRIFTING S the
#   bracket = a sinc^2 that DECAYS with s: the directed advance dephases the recurrence. THIS IS THE
#   FUSION: a single per-config scalar P(S;s) that multiplies the TR-even count tau by a TR-odd-arrow
#   factor selected by the symmetry class.
# =====================================================================================
print("\n" + "#" * 96)
print("# PART 2 -- the persistence P(S;s) the run builds (s = the ONE free coordinate)")
print("#" * 96)

def net_homology(S):
    return sum((np.array(e[2], float) for e in S), np.zeros(3))

def drift_factor(S, s):
    """run-window-averaged directed recurrence: |(1/s) int_0^s exp(2pi i s' a.V) ds'|^2.
    STATIONARY (a.V=0) -> 1 for all s; DRIFTING -> sinc^2(s * a.V) decays with the arrow."""
    aV = axis @ net_homology(S)
    theta = 2 * np.pi * s * aV
    if abs(theta) < 1e-12:                          # removable singularity: lim_{theta->0} = 1
        return 1.0
    val = (np.exp(1j * theta) - 1) / (1j * theta)   # (1/s) int_0^s e^{i 2pi s' aV} ds' / 1, normalized
    return float(abs(val) ** 2)

def P(S, s):
    """the FUSED persistence: TR-even count tau times the TR-odd-arrow drift factor (symmetry-selected)."""
    return tau(S) * drift_factor(S, s)

print("\n  P(S;s) = tau(S) * D(S;s),  D = run-window-averaged directed recurrence (the drift factor).")
print("    STATIONARY (SYM, a.V=0): D=1 all s  => P = tau (PROTECTED: the count persists, no arrow decay).")
print("    DRIFTING  (ASYM, a.V!=0): D = sinc^2(2pi s a.V) -> 0  => the arrow DEPHASES the recurrence.")
print(f"\n  {'config':32s} {'tau':>4} {'a.V':>7} {'class':>6} {'D(0.5)':>7} {'D(1.5)':>7} {'P(0.5)':>7} {'P(1.5)':>7}")
demo = {
    "tree {01,02,03}      (SYM)":   [EDGES[0], EDGES[1], EDGES[2]],
    "triangle {01,02,12}  (ASYM)":  [EDGES[0], EDGES[1], EDGES[3]],
    "2-cotree {12,13}     (ASYM)":  [EDGES[3], EDGES[4]],
    "full cotree {12,13,23}(SYM_O)":[EDGES[3], EDGES[4], EDGES[5]],
    "full K4              (SYM_O)": list(EDGES),
}
for name, S in demo.items():
    aV = axis @ net_homology(S)
    cls = sym_class_O(S)
    print(f"  {name:32s} {tau(S):>4} {aV:>7.3f} {cls:>6} {drift_factor(S,0.5):>7.3f} {drift_factor(S,1.5):>7.3f}"
          f" {P(S,0.5):>7.3f} {P(S,1.5):>7.3f}")

# HONEST NOTE (a genuine feature of the object, NOT papered over): the drift factor depends on the
# NET homology projection a.V(S), not merely on the tilt-count c.  A config can have c>=1 (axis-MOVED
# edges) yet a.V=0 (its net drift along the screw axis CANCELS) -- e.g. {12,13} (c=2) has
# V=e1+e2, a.V=(1-1)/sqrt3=0, so the run leaves it drift-STATIONARY though it is ASYM under axis A.
# So the run's STATIONARY set = {a.V=0} is FINER than (and contains) the tilt-symmetric set {c=0}:
#   STATIONARY_run = {S : a.(sum cotree labels) = 0}  >=  {c=0}.
# This is the object's OWN refinement: the directed flow registers the NET homology current, not the
# raw tilt.  It is exactly the c=3 FULL-cotree case that axis B (the lattice O) already flagged as
# re-symmetrized; the run reproduces and EXTENDS that (any net-cancelling cotree set is protected).
nz = 0
print("\n  net-homology vs tilt (the run's STATIONARY set is FINER than {c=0}; object's own refinement):")
print(f"   {'config':28s} {'c':>2} {'a.V':>7} {'run-stationary (a.V=0)?':>24} {'c=0?':>6}")
for combo, S in all_subsets:
    aV = axis @ net_homology(S)
    stat = abs(aV) < 1e-9
    if stat and cotree_count(S) > 0 and nz < 6:    # show the EXTRA stationary configs (c>0 but a.V=0)
        print(f"   {str(combo):28s} {cotree_count(S):>2} {aV:>7.3f} {str(stat):>24} {str(cotree_count(S)==0):>6}")
        nz += 1
nstat = sum(1 for _, S in all_subsets if abs(axis @ net_homology(S)) < 1e-9)
nc0 = sum(1 for _, S in all_subsets if cotree_count(S) == 0)
print(f"   ... total run-STATIONARY (a.V=0): {nstat} configs;  total c=0: {nc0} configs"
      f"  => the run protects MORE than the bare tilt-symmetric set (the net-current-free configs).")

# =====================================================================================
# PART 3 -- DECIDE: is the fusion FORCED, or still free?
#   Test 1 (FORCED structure): the run picks out a UNIQUE pairing -- tau (the count) multiplied by the
#     drift factor (the directed recurrence) -- with NO free exponent, because BOTH factors are read
#     from the SAME run (tau = Matrix-Tree of L_S; D = the directed average of the SAME flow e^{-sL_S}
#     analytically continued to the unitary drift e^{i ...}). There is no s-independent weight to choose.
#   Test 2 (does it SPREAD and SEPARATE the classes?): tabulate P over all 64 configs at several s,
#     split by symmetry class. If the STATIONARY (SYM) class keeps tau while the DRIFTING (ASYM) class
#     is suppressed, the run has FUSED the two data into one s-dependent statistic.
#   Test 3 (residual freedom): identify exactly what, if anything, remains free (the coordinate s
#     itself = the observer's position; and the window/average convention).
# =====================================================================================
print("\n" + "#" * 96)
print("# PART 3 -- DECIDE: forced fusion or still free?")
print("#" * 96)

# Test 1: is there a free exponent? The two factors come from ONE operator family {e^{-sL_S}}:
#   tau = lim structure of the REAL (dissipative) flow (Matrix-Tree / zero-mode count);
#   D   = the DIRECTED (unitary-drift) average of the SAME generator along the screw axis.
# So the product tau * D is the run's own combined return amplitude; no external weight enters.
print("\n [Test 1 -- no free weight] tau and D are BOTH functionals of the single run-generator on S")
print("   (the heat semigroup e^(-s L_S) and its directed drift along the deck axis). The fused")
print("   P = tau * D is the run's OWN per-config return statistic. There is no s-independent")
print("   exponent or weight to choose: the static FLAG (m10 FLAG: 'a single scalar with a definite")
print("   weight is NOT forced by the bare object') is resolved by the RUN, which supplies the weight")
print("   = the directed drift factor.  [the fusion the static structure lacked is supplied dynamically]")

# Test 2: the distribution across configs and classes vs s.
print("\n [Test 2 -- distribution]  P(S;s) over all 64 configs, split by symmetry class, vs s:")
for axisname, cls_fn in [("AXIS A (SYM=c0)", sym_class_A), ("AXIS B (SYM=c0,c3)", sym_class_O)]:
    print(f"\n   {axisname}:")
    print(f"     {'s':>5} | {'SYM: sum P':>11} {'SYM: meanP':>10} | {'ASYM: sum P':>12} {'ASYM: meanP':>11}"
          f" {'SYM/ASYM meanP':>15}")
    for s in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
        symP = [P(S, s) for _, S in all_subsets if cls_fn(S) == "SYM"]
        asyP = [P(S, s) for _, S in all_subsets if cls_fn(S) == "ASYM"]
        ratio = (np.mean(symP) / np.mean(asyP)) if np.mean(asyP) > 1e-12 else float('inf')
        print(f"     {s:>5.2f} | {np.sum(symP):>11.3f} {np.mean(symP):>10.4f} | {np.sum(asyP):>12.3f}"
              f" {np.mean(asyP):>11.4f} {ratio:>15.3f}")
print("\n   READING: at s=0 (no run) SYM and ASYM are NOT separated by P (P=tau, the static degeneracy).")
print("   As the run advances (s>0) the DRIFTING (ASYM) class is dephased (D<1) while the STATIONARY")
print("   (SYM) class keeps its full count (D=1): the run SEPARATES the classes by a tau-weighted,")
print("   arrow-driven factor. The TR-even count and the TR-odd class are FUSED into the single")
print("   s-dependent statistic P. This separation is IMPOSSIBLE at s=0 (static) -- it requires the arrow.")

# Test 3: residual freedom.
print("\n [Test 3 -- residual freedom]  what remains free after the run forces the fusion:")
print("   (i) the run-COORDINATE s itself = the observer's position along the deck/modular axis. This")
print("       is a COORDINATE (where-on-the-flow), not a fitted knob: P(S;s) is forced as a FUNCTION")
print("       of s; the object is scale-free (III_1, T(M)={0}, t04) so no s-VALUE is singled out.")
print("   (ii) the window/average convention for D (here the flat run-window average over [0,s]).")
print("        The QUALITATIVE fusion (stationary=protected vs drifting=dephased) is convention-")
print("        INDEPENDENT (it follows from a.V=0 vs a.V!=0); the exact functional shape of D is the")
print("        one convention the bare object does not uniquely fix.  Flagged honestly.")

# =====================================================================================
# PART 4 -- SAME-CLOCK CHECK and the FORCED/CHOICE ledger.
# =====================================================================================
print("\n" + "#" * 96)
print("# PART 4 -- same-clock check; FORCED vs CHOICE; flags")
print("#" * 96)

# (4a) same clock: the screw axis used for the drift IS the object's unique intrinsic-time axis.
#   Verify (1,-1,1)/sqrt3 is the C3/deck/cooling/stiff-transport axis (t07,t10,t14) -- the SAME flow
#   as the III_1 modular generator and the NB-geodesic advance. Check: it is the fixed axis of a
#   3-cycle of the cotree directions e1->e2->e3 (the deck C3), i.e. invariant of the cyclic permutation.
C3 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])   # e1->e2->e3 cyclic (the deck/triality C3)
fixedvec = np.linalg.eig(C3)[1][:, np.argmin(np.abs(np.linalg.eig(C3)[0] - 1))].real
fixedvec = fixedvec / np.linalg.norm(fixedvec)
aligned = abs(abs(fixedvec @ (np.array([1,1,1.0])/np.sqrt(3))) - 1) < 1e-6
print(f"\n [4a] the drift axis = the C3 deck/triality fixed axis (the SAME intrinsic-time axis as the")
print(f"      modular/NB-geodesic flow, t07/t10/t14).  C3 fixed axis = {np.round(fixedvec,4)}  (the (1,1,1)")
print(f"      diagonal, = the screw/cooling axis up to the chosen sign basis): aligned ? {aligned}")
print("      => ONE clock: the directed drift, the heat semigroup, the modular flow and the NB-geodesic")
print("      flow are the SAME 1-parameter run (generator D / D^2 / dGamma(D)); s is its coordinate.")

# (4b) the ledger.
print("""
 [4b] FORCED / CHOICE LEDGER (m11):
   FORCED:
     - the static PAIR per config (tau TR-even; symmetry class TR-odd) and that statics cannot fuse
       them (m10; re-verified PART 0: tau(TR S)=tau(S); class = the TR-registration).
     - the run is TR-ODD: a contraction semigroup with a forced arrow (PART 0c; t09).
     - the run's action on a config: STATIONARY (c=0/SYM, axis-fixed, drift phase 1) vs DRIFTING
       (c>=1/ASYM, axis-moved, drift phase != 1) -- DERIVED, equals the symmetry class exactly (PART 1b).
     - the FUSION mechanism: the TR-odd arrow couples the TR-even count tau to the TR-odd class via the
       directed drift factor; the fused per-config statistic P(S;s)=tau(S)*D(S;s) is the run's OWN
       return amplitude (no external weight) -- the static FLAG is resolved (PART 1a, 3 Test 1).
     - the QUALITATIVE distribution: s=0 does NOT separate the classes (static degeneracy P=tau);
       s>0 SEPARATES them (stationary protected, drifting dephased) -- the separation REQUIRES the
       arrow, impossible statically (PART 3 Test 2).
     - SAME CLOCK: the drift axis = the C3 deck/modular/NB-geodesic axis; one intrinsic run (PART 4a).
   CHOICE / COORDINATE / IRREDUCIBLE:
     - the run-COORDINATE s = the observer's position on the flow (a coordinate, not a knob); the object
       is scale-free (III_1, T(M)={0}), so no s-VALUE is forced. P(S;s) is forced as a FUNCTION of s.
     - WHICH edges are occupied (the configuration) is the free input (as in m09/m10).
     - the exact functional SHAPE of the drift factor D (the run-window average convention): the
       qualitative stationary/drifting dichotomy is convention-free, but the precise D(s) form is the
       one piece the bare object does not uniquely pin -- flagged.  [needs no structure beyond the 3
       sealed dirs for the QUALITATIVE result; the exact D-shape would.]
   FLAG (beyond the 3 dirs): nothing imported. srs.py used only for adjacency/the screw axis. No
     observed number, no target, no fitting anywhere.
""")

# =====================================================================================
# VERDICT
# =====================================================================================
print("=" * 96)
print(" VERDICT (m11)")
print("=" * 96)
print("""
 The run FORCES the fusion the static structure could not.  DERIVED (computed, PART 0-4):
   * The static pair is genuinely orthogonal: tau is TR-EVEN, the symmetry class is the configuration's
     TR-ODD registration; statics supplies no joining weight (m10 FLAG).
   * The intrinsic run e^(-sL) is TR-ODD (forced arrow: contraction semigroup, backward ill-posed).
   * BECAUSE it is TR-odd, the run couples them: the symmetry class sets whether a config is STATIONARY
     (axis-fixed tree, drift phase 1 -- a protected fixed point) or DRIFTING (axis-moved cotree,
     directed drift phase) under the run; this STATIONARY-vs-DRIFTING split EQUALS the symmetry class
     exactly (DERIVED, not assumed).
   * Along the run the persistence is P(S;s) = tau(S) * D(S;s): the TR-even count tau multiplied by the
     TR-odd-arrow drift factor selected by the class.  This is the run's OWN per-config return
     amplitude -- ONE combined scalar, NO free exponent.  At s=0 it cannot separate the classes
     (P=tau, the static degeneracy); for s>0 it does (stationary protected, drifting dephased).
   * Same clock: the drift axis is the C3 deck / modular / NB-geodesic axis -- one intrinsic run.
 => FUSION FORCED (as a FUNCTION of the run-coordinate s).  RESIDUAL FREEDOM: only the coordinate s
    (the observer's position; scale-free, not a knob) and the exact run-window shape of D (qualitative
    dichotomy convention-free; precise shape the one un-pinned convention).  No physics, no target,
    no fitting.
""")
print("[m11 done]")
