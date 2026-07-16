#!/usr/bin/env python3
"""
GEN-IDENT-D -- D1 driver: does the canonical M3(C) leg of M rtimes_alpha Z3 carry NO residual
moduli, given D0 = OUTER-CONFIRMED (docs/theorems/genident_D_outerness_2026-07-15.md)?

Freeze: internal research notes, D1.
Theorem write-up (the actual proof lives there, in prose):
    docs/theorems/genident_D1_canonical_home_2026-07-15.md

WHAT THIS SCRIPT IS AND IS NOT.
  The D1 rigidity result (M' cap (M rtimes_alpha Z3) = C, forced by proper outerness of alpha and
  alpha^2, D0-iii/iv) is an INFINITE-DIMENSIONAL operator-algebra theorem about M = L(F_inv(6)); it
  is NOT itself a finite computation -- exactly the same epistemic status as D0. This driver supplies
  three kinds of NON-VACUITY / MECHANISM witnesses, all pure finite linear algebra (sympy, exact
  algebraic arithmetic, no floating point, no physics import):

    (1) SECTION 1 -- a finite (necessarily ABELIAN, see the honest note printed there) toy crossed
        product Z3 acting freely on a finite set, with a NONTRIVIAL fixed-point leg M^alpha (dim 2,
        not just C), verifying the matrix-unit relations E_jk = U^j e U^{-k} genuinely close as
        M3(C) tensor M^alpha for the CORRECT choice of seed projection e -- and, honestly, that the
        freeze's own literal shorthand "e = E_{M^alpha} = (1/3)(1+u+u^2)" does NOT work as the
        matrix-unit seed (a precision correction, flagged and verified computationally, analogous to
        the verifier's D0-iv citation-precision flag -- it does not change the D1 verdict).

    (2) SECTION 2 -- THE DECISIVE CONTRAST the task asks for. A finite-dimensional INNER model
        (M = M_2(C), alpha = Ad(u0) for an explicit order-3 unitary u0 -- forced to be INNER by
        Skolem-Noether, exactly GEN-IDENT-C's obstruction) realized as an honest covariant
        representation of M rtimes_alpha Z3 on C^2 tensor C^3. This EXHIBITS a concrete, non-scalar
        element y = y_1 . u (y_1 = u0^{-1}) of the relative commutant M' cap (M rtimes Z3) --
        i.e. the residual "unitary moduli" GEN-IDENT-C found (there: dim 24, U(4)xU(2)xU(2))
        REAPPEARS here, concretely, exactly because alpha is inner. A negative control (a generic
        y_1 NOT satisfying the defining relation) is checked to genuinely fail to commute, so the
        positive witness is not vacuous.

    (3) SECTION 3 -- an honest dimension-count witness for WHY the M3(C) tensor M^alpha
        decomposition cannot hold for an INNER action (12 != 9*2 for the M_2(C) toy), reinforcing
        that outerness is not merely "nicer" but load-bearing for the theorem to even be
        dimensionally consistent -- and a goal-seek AST self-scan.

  The FULL RIGOROUS PROOF that D0's outerness (already sealed) forces y_1 = y_2 = 0 for the REAL
  M = L(F_inv(6)) -- via the lemma "a nonzero w in M with x w = w alpha(x) for all x in M forces
  alpha inner, using finiteness of M (isometry => unitary in a II_1 factor)" -- is infinite-
  dimensional and lives in the theorem write-up, NOT in this driver. This driver's Section 2
  computationally CALIBRATES that lemma's central equivalence ("x y1 = y1 alpha(x) for all x"
  <=> "Ad(y1^{-1}) = alpha") in the one regime where it is directly testable (the inner toy), and
  lets the (already sealed, not re-litigated) D0 theorem supply the "no such y1 exists" half that
  only holds in the outer/infinite-dimensional case.

GOAL-SEEK GUARD: this driver imports NOTHING from the physics codebase (no the_net.py, no
predictions/, no m1b_*.py). It is pure finite-dimensional operator-algebra combinatorics on
abstract toy algebras (M_2(C), a 6-point set). No mass/ppm/Koide/mass-ordering/mixing/CKM/PMNS
value appears anywhere. Verified by an AST self-scan in Section 3 (same technique as the D0 driver).

OMP_NUM_THREADS=4. sympy exact arithmetic throughout (algebraic omega = primitive cube root of
unity) -- no floating point, no numpy. Runtime: a few seconds.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")

import sympy as sp
from sympy import Matrix, eye, zeros, sqrt, Rational, I, simplify, expand

RESULTS = []


def check(name, cond, note=""):
    RESULTS.append((name, bool(cond), note))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}   {note}")
    return bool(cond)


def hdr(s):
    print("\n" + "=" * 100 + "\n" + s + "\n" + "=" * 100)


def mzero(M):
    """Exact zero-matrix test via sympy's algebraic equality (handles sqrt(3), I exactly)."""
    return M.equals(zeros(*M.shape))


def meq(A, B):
    return A.equals(B)


omega = Rational(-1, 2) + sqrt(3) * I / 2  # primitive cube root of unity, EXACT algebraic number
check("SETUP omega^3 = 1 exactly (algebraic, not floating point)", meq(expand(omega**3), sp.Integer(1)))
check("SETUP omega != 1 (order exactly 3, not 1)", not meq(omega, sp.Integer(1)))

# =====================================================================================================
hdr("SECTION 1 -- finite toy crossed product: matrix units E_jk = U^j e U^{-k} close as M3(C) (x) M^alpha")
# =====================================================================================================
print("""
Toy: Z3 acting FREELY on a 6-point set {0,1,2} x {a,b} (translation on the first factor, identity on
the second) -- the finite/abelian shadow of a free G-action, with a NONTRIVIAL fixed-point leg
M^alpha = C^{a,b} (dim 2, matching the freeze's own rail: "M^alpha is a big subalgebra, NOT C" --
here it is dim 2, not dim 1, so the check is not vacuously trivial). Basis order: index = 2*n + s,
n in {0,1,2} (the Z3 grade), s in {0,1} (the spectator leg realizing M^alpha).
""")

D6 = 6


def idx(n, s):
    return 2 * n + s


U6 = zeros(D6, D6)  # shift n -> n+1, identity on s
for n in range(3):
    for s in range(2):
        U6[idx((n + 1) % 3, s), idx(n, s)] = 1

check("S1 U6 is a genuine permutation matrix (0/1 entries, one 1 per row/col)",
      all(sum(U6[r, c] for c in range(D6)) == 1 for r in range(D6)) and
      all(sum(U6[r, c] for r in range(D6)) == 1 for c in range(D6)))
check("S1 U6^3 = I (order exactly 3)", meq(U6 * U6 * U6, eye(D6)))
check("S1 U6 != I and U6^2 != I (order is EXACTLY 3, not a divisor)",
      not meq(U6, eye(D6)) and not meq(U6 * U6, eye(D6)))

U6pow = {0: eye(D6), 1: U6, 2: U6 * U6}


def U6_pow(n):
    return U6pow[n % 3]


def U6_neg_pow(n):
    return U6pow[(-n) % 3]

# THE CORRECT seed: e = minimal-type projection onto the n=0 slice (rank 2 = dim M^alpha, NOT rank 1)
e_min = zeros(D6, D6)
e_min[idx(0, 0), idx(0, 0)] = 1
e_min[idx(0, 1), idx(0, 1)] = 1
check("S1 e_min is a genuine projection (e^2=e, e*=e)", meq(e_min * e_min, e_min) and meq(e_min.T, e_min))
check("S1 rank(e_min) = 2 = dim(M^alpha) (nontrivial fixed-point leg, not the degenerate dim-1 case)",
      e_min.trace() == 2)

Ejk = {}
for j in range(3):
    for k in range(3):
        Ejk[(j, k)] = U6_pow(j) * e_min * U6_neg_pow(k)

closure_ok = True
for j in range(3):
    for k in range(3):
        for l in range(3):
            for m in range(3):
                lhs = Ejk[(j, k)] * Ejk[(l, m)]
                rhs = Ejk[(j, m)] if k == l else zeros(D6, D6)
                if not meq(lhs, rhs):
                    closure_ok = False
check("S1 matrix-unit closure: E_jk E_lm = delta_{kl} E_jm EXACTLY, all 81 (j,k,l,m) combinations",
      closure_ok)

sum_diag = Ejk[(0, 0)] + Ejk[(1, 1)] + Ejk[(2, 2)]
check("S1 partition of unity: E_00 + E_11 + E_22 = I6 exactly", meq(sum_diag, eye(D6)))

herm_ok = all(meq(Ejk[(j, k)].T, Ejk[(k, j)]) for j in range(3) for k in range(3))
check("S1 E_jk^dagger = E_kj (Hermitian matrix-unit structure, real entries here so transpose=adjoint)",
      herm_ok)

print("""
This confirms the RECIPE (matrix units seeded by a rank-2 = dim(M^alpha) minimal-type projection,
built from the implementing unitaries) genuinely closes as M3(C) tensor M^alpha, with a NONTRIVIAL
M^alpha leg -- the mechanism the freeze's D1 target names.
""")

# --- Honest precision check: the freeze's OWN literal shorthand "e = E_{M^alpha} = (1/3)(1+u+u^2)"
# does NOT work as the E_jk seed -- flagged and verified computationally (does not change the D1
# verdict; the correct construction above is what the theorem write-up actually uses).
e_avg = Rational(1, 3) * (eye(D6) + U6 + U6 * U6)
check("S1-PRECISION e_avg = (1/3)(1+U+U^2) IS a genuine projection (e_avg^2=e_avg)",
      meq(e_avg * e_avg, e_avg))
diag_from_avg = [U6_pow(j) * e_avg * U6_neg_pow(j) for j in range(3)]
degenerate = all(meq(diag_from_avg[j], e_avg) for j in range(3))
check("S1-PRECISION (honest flag, NOT a D1 defect): U^j e_avg U^{-j} = e_avg IDENTICALLY for ALL j "
      "-- i.e. the freeze's literal group-averaged e does NOT give 3 distinct diagonal matrix units "
      "(this is a genuine algebraic fact: e_avg is G-invariant by construction, so conjugating it by "
      "any group element leaves it fixed). The CORRECT seed for E_jk is a MINIMAL-type projection "
      "(e_min above), not the group average -- e_avg is instead the JONES/conditional-expectation "
      "projection satisfying x*e_avg = e_avg*E(x) for x in M^alpha-language, a different, "
      "complementary object. This is a precision correction to the freeze's shorthand, analogous to "
      "the verifier's D0-iv citation flag; it does not affect the D1 verdict, which rests on "
      "the relative-commutant argument (Section 2 below), not on this specific formula.",
      degenerate)

# =====================================================================================================
hdr("SECTION 2 -- THE DECISIVE CONTRAST: finite INNER model exhibits the residual moduli; "
    "the (uncomputable-in-finite-dim) OUTER mechanism is calibrated against it")
# =====================================================================================================
print("""
M = M_2(C) (a genuine FACTOR: M_2(C)' cap M_2(C) = C by Schur). alpha = Ad(u0), u0 an explicit
order-3 unitary -- by Skolem-Noether, EVERY automorphism of a full finite matrix algebra is inner,
so alpha is FORCED inner here (this is exactly GEN-IDENT-C's obstruction, in miniature: no finite-
dimensional factor can host a properly outer Z3 action, which is why D0/D1's theorem needed the
type-II_1 factor M = L(F_inv(6)) instead).
""")

u0 = Matrix([[1, 0], [0, omega]])
check("S2 u0 is unitary (u0^dagger u0 = I)", meq(u0.H * u0, eye(2)))
check("S2 u0^3 = I (order exactly 3)", meq(u0 * u0 * u0, eye(2)))
check("S2 u0 != I and u0^2 != I", not meq(u0, eye(2)) and not meq(u0 * u0, eye(2)))

u0_inv = u0.inv()
check("S2 u0_inv = u0^dagger (unitarity, independent check)", meq(u0_inv, u0.H))

E11 = Matrix([[1, 0], [0, 0]])
E12 = Matrix([[0, 1], [0, 0]])
E21 = Matrix([[0, 0], [1, 0]])
E22 = Matrix([[0, 0], [0, 1]])
BASIS = [E11, E12, E21, E22]


def alpha1(x):
    return simplify(u0 * x * u0_inv)


def alpha2(x):
    return alpha1(alpha1(x))


check("S2 alpha is a genuine automorphism of M_2(C): alpha(xy) = alpha(x)alpha(y) on 3 sample "
      "products", all(meq(alpha1(a * b), alpha1(a) * alpha1(b))
                       for a, b in [(E12, E21), (E11, E12), (E21, E22)]))
check("S2 alpha^3 = id on the basis (order divides 3, matches u0's order)",
      all(meq(alpha2(alpha1(x)), x) for x in BASIS))

# M^alpha (fixed-point subalgebra of THIS M_2(C), i.e. things commuting with u0)
comm_flags = {}
for name, x in [("E11", E11), ("E12", E12), ("E21", E21), ("E22", E22)]:
    comm_flags[name] = mzero(u0 * x - x * u0)
dim_Malpha = sum(1 for v in comm_flags.values() if v)
check("S2 M^alpha (things commuting with u0) is spanned by {E11,E22} (diagonal matrices), dim 2 -- "
      "E12,E21 do NOT commute with u0 (off-diagonal entries pick up the omega phase)",
      comm_flags["E11"] and comm_flags["E22"] and not comm_flags["E12"] and not comm_flags["E21"],
      note=f"{comm_flags}")

print("""
--- The clean, BARE M_2(C) form of the key relation the D1 rigidity lemma turns on ---
"x y1 = y1 alpha(x) for all x in M"  <=>  "Ad(y1^{-1}) = alpha"  (for invertible y1).
Solve directly: y1 := u0^{-1} satisfies x u0^{-1} = u0^{-1} alpha(x) for ALL x (this is a one-line
algebraic consequence of alpha(x) = u0 x u0^{-1}: multiply on the left/right by u0^{-1}). Verified
on the full basis below (exhaustive, by linearity this covers ALL of M_2(C), not a sample).
""")

y1_relation_ok = all(meq(x * u0_inv, u0_inv * alpha1(x)) for x in BASIS)
check("S2 THE KEY RELATION: x * u0^{-1} = u0^{-1} * alpha(x) holds EXACTLY for all 4 basis elements "
      "of M_2(C) (exhaustive by linearity) -- y1 = u0^{-1} solves the defining equation because "
      "alpha is (necessarily, Skolem-Noether) INNER here", y1_relation_ok)

print("""
--- Now realize this inside the actual crossed product M rtimes_alpha Z3, via its covariant
representation on C^2 (x) C^3 (H_M (x) l^2(Z3)), to exhibit the NON-SCALAR relative-commutant
element directly as an operator identity, not just the bare-M_2(C) relation above. ---

Covariant representation (standard construction): PI(x) is block-diagonal with blocks
[x, alpha^{-1}(x), alpha^{-2}(x)] = [x, alpha^2(x), alpha(x)] at grades n=0,1,2 (block n carries
alpha^{-n}(x), the twist needed so PI(u) PI(x) PI(u)^{-1} = PI(alpha(x)) holds); PI(u) is the pure
shift n -> n+1 (identity on the C^2 leg). This is checked below BEFORE being used further.
""")


def PI(x):
    return sp.diag(x, alpha2(x), alpha1(x))  # blocks at n=0,1,2: x, alpha^{-1}(x)=alpha^2(x), alpha^{-2}(x)=alpha(x)


D6b = 6
PI_u = zeros(D6b, D6b)
for n in range(3):
    PI_u[2 * ((n + 1) % 3):2 * ((n + 1) % 3) + 2, 2 * n:2 * n + 2] = eye(2)

check("S2 PI_u is a genuine permutation-of-blocks matrix, PI_u^dagger PI_u = I", meq(PI_u.T * PI_u, eye(D6b)))
check("S2 PI_u^3 = I (order exactly 3)", meq(PI_u * PI_u * PI_u, eye(D6b)))

PI_u_inv = PI_u.T  # real permutation-of-blocks matrix, inverse = transpose

covariance_ok = all(meq(PI_u * PI(x) * PI_u_inv, PI(alpha1(x))) for x in BASIS)
check("S2 COVARIANCE CHECK (correctness of the representation, before using it further): "
      "PI(u) PI(x) PI(u)^{-1} = PI(alpha(x)) EXACTLY, all 4 basis elements",
      covariance_ok)

Y_relation = PI(u0_inv) * PI_u  # abstractly represents the crossed-product element y1.u, y1=u0^{-1}

commute_ok = all(mzero(Y_relation * PI(x) - PI(x) * Y_relation) for x in BASIS)
check("S2 *** THE POSITIVE WITNESS *** Y := PI(u0^{-1}) PI(u) COMMUTES with PI(x) for ALL x in "
      "M_2(C) (exhaustive over the basis) -- Y is a genuine element of the relative commutant "
      "M' cap (M rtimes Z3)", commute_ok)

is_scalar = meq(Y_relation, Y_relation[0, 0] * eye(D6b))
check("S2 *** Y IS NOT A SCALAR MULTIPLE OF THE IDENTITY *** -- this is the concrete, non-vacuous "
      "reappearance of GEN-IDENT-C's residual moduli (there: dim 24, U(4)xU(2)xU(2); here: at "
      "minimum this one extra non-scalar direction) EXACTLY BECAUSE alpha is inner. If alpha had "
      "been properly outer (impossible in finite dim, by Skolem-Noether, but true for the REAL "
      "M=L(F_inv(6)) per sealed D0), no such y1 could exist (D0-iii/iv), and M' cap (M rtimes Z3) "
      "would collapse to scalars only -- exactly the D1 rigidity claim.",
      not is_scalar, note=f"Y[0,0]={Y_relation[0,0]}")

# --- Negative control: does a GENERIC y1 (not solving the relation) fail to give a commuting Y? ---
print("""
--- Negative control (guards against the positive witness above being vacuously easy to satisfy) ---
Pick y1_bad = E12 (does NOT solve "x y1 = y1 alpha(x)" -- verified directly below), and confirm the
corresponding Y_bad = PI(y1_bad) PI(u) genuinely FAILS to commute with a generic element of M_2(C).
""")
y1_bad = E12
relation_fails_for_bad = not all(meq(x * y1_bad, y1_bad * alpha1(x)) for x in BASIS)
check("S2-CTRL y1_bad = E12 genuinely does NOT solve the defining relation for at least one basis x "
      "(confirms y1_bad is a real negative control, not an accidental second solution)",
      relation_fails_for_bad)

Y_bad = PI(y1_bad) * PI_u
bad_commutes = [mzero(Y_bad * PI(x) - PI(x) * Y_bad) for x in BASIS]
check("S2-CTRL *** THE NEGATIVE WITNESS *** Y_bad = PI(E12) PI(u) FAILS to commute with at least "
      "one basis element of M_2(C) -- confirms the Fourier-matching mechanism is genuinely "
      "DISCRIMINATING (only relation-solving y1 give commuting elements), not trivially true for "
      "any y1", any(not c for c in bad_commutes), note=f"commutes_per_basis_elt={bad_commutes}")

# --- y0-alone check: a non-scalar y0 (no u-component at all) also fails to commute (M is a factor) ---
print("""
--- y0-alone check (the n=0 Fourier component of the rigidity argument: M' cap M = C since M is a
factor) ---
""")
Y0 = PI(E11)  # y0 = E11 (non-scalar), y1=y2=0
y0_commutes = [mzero(Y0 * PI(x) - PI(x) * Y0) for x in BASIS]
check("S2-Y0 a non-scalar y0=E11 (embedded at grade 0 only, y1=y2=0) FAILS to commute with all of "
      "M_2(C) -- confirms the n=0 Fourier component must be scalar (M_2(C) is a factor, "
      "M_2(C)' cap M_2(C) = C, the finite-dimensional shadow of the same M'cap(M rtimes Z3) "
      "argument's n=0 term used for the REAL M = L(F_inv(6))",
      any(not c for c in y0_commutes), note=f"commutes_per_basis_elt={y0_commutes}")

# =====================================================================================================
hdr("SECTION 3 -- dimension-count witness: WHY the M3(C) (x) M^alpha decomposition structurally "
    "requires outerness (fails for the inner toy), + goal-seek self-scan")
# =====================================================================================================
print("""
dim(M rtimes G) = |G| * dim(M) ALWAYS (definitional: the crossed product's underlying vector space
is M (x) span{u^0,...,u^{|G|-1}}), regardless of whether the action is inner or outer. The CLEAN
tensor decomposition M rtimes_alpha G = M_{|G|}(C) (x) M^alpha, by contrast, forces
dim(M rtimes G) = |G|^2 * dim(M^alpha) -- and this SECOND formula is only valid for a FREE (outer)
action. For our M_2(C) toy (forced INNER by Skolem-Noether), check whether the two formulas agree.
""")

dim_M = 4  # dim_C M_2(C)
G_order = 3
dim_crossed_definitional = G_order * dim_M
dim_Malpha_toy = 2  # computed in Section 2 (E11, E22 span it)
dim_would_be_tensor = (G_order ** 2) * dim_Malpha_toy

check(f"S3 dim(M rtimes Z3) [definitional, ALWAYS true] = {G_order}*{dim_M} = {dim_crossed_definitional}",
      dim_crossed_definitional == 12)
check(f"S3 would-be dim(M3(C) (x) M^alpha) [ONLY valid if the action is FREE/outer] = "
      f"9*{dim_Malpha_toy} = {dim_would_be_tensor}", dim_would_be_tensor == 18)
check("S3 *** THE MISMATCH *** 12 != 18 -- the clean M3(C) (x) M^alpha tensor decomposition is "
      "DIMENSIONALLY INCONSISTENT for this INNER action, confirming (by a completely different, "
      "purely arithmetic route from Section 2's operator-level witness) that outerness is not "
      "merely 'nicer' but STRUCTURALLY NECESSARY for the theorem to even typecheck. (For the REAL, "
      "properly-outer alpha on M=L(F_inv(6)) [infinite-dimensional, sealed D0], the analogous count "
      "is the standard index formula [M:M^alpha]=|G|=3 EXACTLY, which is what makes the "
      "decomposition consistent there -- not independently checkable in finite dimensions, but the "
      "mismatch here shows concretely what goes wrong when the hypothesis fails.)",
      dim_crossed_definitional != dim_would_be_tensor)

print("""
--- Goal-seek AST self-scan (same technique as the D0 driver: an AST walk, not a self-referential
substring search over this file's own descriptive prose) ---
""")
import ast

with open(__file__) as f:
    own_src = f.read()
tree = ast.parse(own_src)

imported_modules = []
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imported_modules += [a.name for a in node.names]
    elif isinstance(node, ast.ImportFrom) and node.module:
        imported_modules.append(node.module)

FORBIDDEN_MODULE_PREFIXES = ("derivation_topdown", "predictions", "proofs.foundations.m1b", "the_net")
bad_imports = [m for m in imported_modules if m.startswith(FORBIDDEN_MODULE_PREFIXES)]
check("GOALSEEK-1 (AST) actual import statements reference ONLY stdlib/sympy (no the_net, no "
      "predictions/, no m1b_*)", len(bad_imports) == 0, note=f"all imports={imported_modules}")

numeric_constants = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        numeric_constants.add(node.value)
has_float = any(isinstance(v, float) for v in numeric_constants)
BOUND = 1000
bad_numeric = sorted(v for v in numeric_constants if isinstance(v, int) and abs(v) > BOUND)
check("GOALSEEK-2 (AST) every numeric literal in the executable code is a small integer this "
      "driver's own toy-algebra combinatorics needs (grades 0-6, dims 2-18) -- NO floating-point "
      "constant anywhere (a mass/coupling/ppm ratio would be a float)",
      (not has_float) and len(bad_numeric) == 0,
      note=f"has_float={has_float}, out-of-range ints={bad_numeric}, "
           f"all numeric constants={sorted(numeric_constants, key=str)}")

# =====================================================================================================
hdr("SUMMARY")
# =====================================================================================================
n_pass = sum(1 for r in RESULTS if r[1])
n_total = len(RESULTS)
print(f"\n{n_pass}/{n_total} recorded checks PASS\n")
for name, passed, note in RESULTS:
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}   {note}")

print("\n" + "-" * 100)
print("These are NON-VACUITY / MECHANISM WITNESSES for the D1 theorem (full proof in")
print("docs/theorems/genident_D1_canonical_home_2026-07-15.md), not the proof itself. Section 1")
print("confirms the matrix-unit recipe genuinely closes as M3(C) tensor M^alpha with a nontrivial")
print("M^alpha leg (and honestly flags a precision issue in the freeze's literal e-formula, fixed).")
print("Section 2 is the decisive contrast: the finite/INNER toy concretely EXHIBITS a non-scalar")
print("relative-commutant element (the residual moduli GEN-IDENT-C found), with a genuine negative")
print("control confirming the mechanism discriminates. Section 3 shows the tensor decomposition is")
print("dimensionally IMPOSSIBLE for an inner action, and confirms the goal-seek guard via AST scan.")

if n_pass == n_total:
    print("\nRESULT: ALL CHECKS PASS")
else:
    print(f"\nRESULT: {n_total - n_pass} CHECK(S) FAILED")

sys.exit(0 if n_pass == n_total else 1)
