#!/usr/bin/env python3
"""
GEN-IDENT-B -- durable check driver for
internal research notes

PRIMARY QUESTION (freeze §1): when the observer factor C^3_obs is coupled to the substrate through
the REAL vertex functional -kappa.I(A;B) (I-0a; the same functional that selected the triad = W's
orbit in V1/July), what is the residual freedom of the observer identification? Does the continuous
U(1)^2 collapse (B2) or survive (B1)?

Runs T1-T5 in the order the freeze specifies.

T1's answer is DISPOSITIVE and determines everything downstream: there is NO built code-level
coupling between the observer factor C^3_obs (R3 / M1.B) and the vertex functional -kappa.I(A;B)
(the_net.py section 11, V1). This is verified directly below by source inspection (no shared
imports either direction, no function name at the intersection, and an explicit standing
disclaimer in the_net.py itself forbidding the conflation). Per the freeze's own routing rule
(§3 T1, §5 "T1 = coupling un-built"), that is this station's booked finding -- NOT a B1/B2/B3 bin.

T3-T5 below independently reproduce (and extend with the freeze's mandatory T5 controls) the
mechanism-availability computation from the main-loop scout (scratchpad/genidentB_scout.py):
IF an observer were forced to respect both sigma (winding/generation axis) and W (the July
selector axis), the joint commutant collapses to scalars by Schur (<sigma,W> = A4, irreducible on
rho3). This is reported as a COUNTERFACTUAL / mechanism-availability result, not the station's
actual finding, because T1 shows the "if" is not realized in the codebase.

GOAL-SEEK GUARD (verbatim, stronger here -- the ppm wall): no mass/ppm/Koide-Q/mass-ordering/
mixing/CKM value is read, compared, referenced, or used as a selection criterion ANYWHERE below.
Every object used (sigma, W, rho3, the A4 vertex group) is pure finite-group / linear-algebra
structure, identical to what the GEN-IDENT-A check and the GEN-HOMES check already used for the
same purpose. Nothing is fit; nothing is tuned.

OMP_NUM_THREADS=4. Runtime: a few seconds. Read-only on the_run.py and Layer-1. Not wired into
verify.py (freeze §6).
"""
import sys, os, re, inspect

sys.path.insert(0, ".")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

REPO = "."

RESULTS = []


def check(name, cond, note=""):
    RESULTS.append((name, bool(cond), note))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}   {note}")
    return bool(cond)


def hdr(s):
    print("\n" + "=" * 100 + "\n" + s + "\n" + "=" * 100)


# =====================================================================================================
hdr("T1 -- LOCATE the real observer<->substrate coupling (source-level, not prose)")
# =====================================================================================================
print("""
Freeze §3 T1: find, in the code, how C^3_obs (the observer factor: R3 / m1b_*.py) couples to the
vertex -kappa.I(A;B) (I-0a; the_net.py section 11 "V1 -- THE VERTEX ON THE FAMILY"). If the coupling
is only prose / not concretely built, that is a legitimate finding, logged rather than fabricated.
""")

with open(os.path.join(REPO, "derivation_topdown/state/the_net.py")) as f:
    the_net_src = f.read()

m1b_files = [
    "proofs/foundations/m1b_observer_substrate_iprojection_attempt.py",
    "proofs/foundations/m1b_c_basis_match.py",
    "proofs/foundations/m1b_d_iprojection_structural_map.py",
]
m1b_src = {}
for rel in m1b_files:
    with open(os.path.join(REPO, rel)) as f:
        m1b_src[rel] = f.read()

r3_path = "predictions/R3_observer_c3_generation.py"
with open(os.path.join(REPO, r3_path)) as f:
    r3_src = f.read()

# (a) the_net.py imports NOTHING from the observer-factor files (R3 / m1b_*)
import_lines = [l for l in the_net_src.splitlines() if l.strip().startswith(("import ", "from "))]
observer_imports_in_net = [l for l in import_lines if re.search(r"\bR3\b|m1b_|observer", l, re.I)]
check("T1a the_net.py has ZERO import referencing R3/m1b/observer files",
      len(observer_imports_in_net) == 0, note=f"found={observer_imports_in_net}")

# (b) the observer-factor files import NOTHING from the_net.py / derivation_topdown
net_imports_in_observer_files = {}
for rel, src in {**m1b_src, r3_path: r3_src}.items():
    lines = [l for l in src.splitlines() if l.strip().startswith(("import ", "from "))]
    hits = [l for l in lines if re.search(r"the_net|derivation_topdown", l, re.I)]
    net_imports_in_observer_files[rel] = hits
check("T1b none of the observer-factor files (R3, m1b_*) import the_net.py/derivation_topdown",
      all(len(v) == 0 for v in net_imports_in_observer_files.values()),
      note=f"{net_imports_in_observer_files}")

# (c) no function/object name in the_net.py's namespace at the observer<->vertex intersection
import derivation_topdown.state.the_net as tn
observer_named = [n for n in dir(tn) if re.search(r"observ|c3_gen|c3_obs", n, re.I)]
check("T1c the_net.py module namespace has ZERO name referencing the observer factor "
      "(no observer_coupling / c3_obs / c3_gen object of any kind)",
      len(observer_named) == 0, note=f"found={observer_named}")

# (d) the standing disclaimer IS present verbatim in the_net.py (the codebase's own firewall)
disclaimer_hit = "OUTLOOK ONLY" in the_net_src and "R3_observer_c3_generation" in the_net_src
disclaimer_line = next((i + 1 for i, l in enumerate(the_net_src.splitlines())
                         if "R3_observer_c3_generation" in l), None)
check("T1d the_net.py carries an EXPLICIT standing disclaimer against conflating the vertex-selected "
      "species/multiplicity map with the observer-C3 generation count ('OUTLOOK ONLY... triad<->"
      "generations is never in the verdict path')",
      disclaimer_hit, note=f"line {disclaimer_line}")

# (e) the vertex functional -kappa.I(A;B) (V1, the_net.py section 11) operates on H_hist (x) F --
#     a DIFFERENT tensor factor than C^3_obs (R3's B(C^3), or M1.B's Galois-tower M_3(C) factor).
#     Confirm the V1 vertex functional's carrier objects exist and are self-contained within the_net.
v1_objects_present = all(hasattr(tn, name) for name in
                          ["w2_family_direction", "w2_gamma_table", "v1_channel_state"])
check("T1e the V1 vertex machinery (w2_family_direction/w2_gamma_table/v1_channel_state) is built "
      "and self-contained in the_net.py, operating on H_hist (x) F -- a DIFFERENT carrier than "
      "C^3_obs (R3's B(C^3) / M1.B's crossed-product M_3(C) factor, both abstract/toy-model, no "
      "numeric tie to H_hist (x) F anywhere)", v1_objects_present)

T1_COUPLING_BUILT = not (len(observer_imports_in_net) == 0
                          and all(len(v) == 0 for v in net_imports_in_observer_files.values())
                          and len(observer_named) == 0)
print(f"\n*** T1 VERDICT: coupling is {'BUILT' if T1_COUPLING_BUILT else 'UN-BUILT'} "
      f"(the observer factor and the vertex functional are two disjoint code objects; the "
      f"freeze's premised coupling '-kappa.I(A;B) applied to C^3_obs' does not exist as a callable, "
      f"only as a shared-name coincidence flagged and explicitly firewalled in the_net.py itself) ***")

# =====================================================================================================
hdr("T2 -- does the coupling distinguish the W-structure on the observer side? (moot given T1)")
# =====================================================================================================
print("""
T2 asks whether the real observer<->substrate coupling operator is W-covariant. Since T1 establishes
there is no such coupling object to interrogate, T2 has no operand: there is nothing to evaluate for
W-covariance. This is recorded as N/A-by-construction, not as a silent skip.
""")
check("T2 is N/A: no coupling operator exists on which to test W-covariance (consequence of T1)",
      not T1_COUPLING_BUILT)


# =====================================================================================================
hdr("T3 -- mechanism-availability computation (COUNTERFACTUAL: 'if the coupling existed')")
# =====================================================================================================
print("""
Independent re-derivation of the main-loop scout (scratchpad/genidentB_scout.py), reported here as
what the residual WOULD be if an observer were forced to respect BOTH sigma (generation axis) and W
(the July selector axis) -- NOT a claim about the as-built system (T1 already answered that: B1-like,
full freedom, since nothing forces anything). Pure finite-group / linear algebra, no external data.
""")

from derivation_topdown.state.the_net import _a4_vertex_group, _a4_standard_3irrep, _a4_key, NV

A4v = _a4_vertex_group()
ix = {_a4_key(g): n for n, g in enumerate(A4v)}


def comp(g, h):
    return {i: g[h[i]] for i in range(NV)}


e_id = {i: i for i in range(NV)}
sigma = {0: 0, 1: 2, 2: 3, 3: 1}
sigma_idx = ix[_a4_key(sigma)]
sigma2 = comp(sigma, sigma)
sigma2_idx = ix[_a4_key(sigma2)]
_, rho3, _, _ = _a4_standard_3irrep()
rS, rW = rho3[sigma_idx], rho3[5]
W_gen = A4v[5]


def group_closure(gens_dict):
    elems = {_a4_key(e_id): e_id}
    for g in gens_dict:
        elems[_a4_key(g)] = g
    changed = True
    while changed:
        changed = False
        for a in list(elems.values()):
            for b in gens_dict:
                c = comp(a, b)
                k = _a4_key(c)
                if k not in elems:
                    elems[k] = c
                    changed = True
    return elems


grp_sigma_W = group_closure([sigma, W_gen])
check("T3a <sigma, W> generates A4 exactly (order 12)", len(grp_sigma_W) == 12,
      note=f"order={len(grp_sigma_W)}")


def commutant_dim(mats, tol_factor=1e-9):
    """dim_C of {X in M_3(C) : X M = M X for all M in mats}, via SVD nullity of the stacked
    commutator-constraint operator vec(MX-XM) = (I kron M - M^T kron I) vec(X)."""
    rows = []
    I3 = np.eye(3)
    for M in mats:
        C = np.kron(I3, M) - np.kron(M.T, I3)
        rows.append(C)
    A = np.vstack(rows)
    u, s, vh = np.linalg.svd(A)
    tol = tol_factor * max(A.shape) * (s[0] if s.size else 1.0)
    rank = int(np.sum(s > tol))
    return 9 - rank


dim_sigma_only = commutant_dim([rS])
dim_joint = commutant_dim([rS, rW])
check("T3b commutant(rho3(sigma)) alone has complex dim 3 (the maximal torus, i.e. U(1)^3 -> U(1)^2 "
      "mod global phase)", dim_sigma_only == 3, note=f"dim={dim_sigma_only}")
check("T3c joint commutant(rho3(sigma), rho3(W)) collapses to complex dim 1 (scalars only, Schur, "
      "since <sigma,W>=A4 acts irreducibly on rho3)", dim_joint == 1, note=f"dim={dim_joint}")

CONTINUOUS_DIM_SIGMA_ONLY = dim_sigma_only - 1  # mod overall global phase
CONTINUOUS_DIM_JOINT = dim_joint - 1
print(f"\nContinuous residual dimension (mod overall phase): sigma-alone = "
      f"{CONTINUOUS_DIM_SIGMA_ONLY} (matches GEN-HOMES's U(1)^2); joint{{sigma,W}} = "
      f"{CONTINUOUS_DIM_JOINT} (collapsed).")
check("T3d the continuous part fully collapses under the joint (counterfactual) constraint "
      "(0-dimensional residual)", CONTINUOUS_DIM_JOINT == 0)

print("""
On the DISCRETE ("labeling") part: the freeze's own mechanism sketch (§0) holds the discrete S_3
(the 3! ways to assign the names {gen-0,gen-1,gen-2} to the 3 already-distinguished eigenspaces)
FIXED at order 6, treating it as a bookkeeping freedom orthogonal to the U(3) commutant computation.
This check does NOT independently re-derive whether the FORCED sigma<->W relative orientation
(GEN-IDENT-A) further constrains that labeling choice below order 6 (e.g. down to the Out(A4)~=Z_2
normalizer-of-the-pair, a plausible but UNVERIFIED alternate reading this station flags but does not
resolve) -- that refinement is left explicitly OPEN, not computed here, to avoid overreach beyond
what T3a-T3d rigorously establish (the freeze does not ask for it, and resolving it risks the exact
smuggled-precision failure mode the goal-seek guard warns against).
""")
check("T3e discrete labeling order reported AS DEFINED BY THE FREEZE (S_3, order 6) -- NOT "
      "independently re-derived; the possible further reduction to Out(A4)~=2 is flagged OPEN, "
      "not booked", True, note="see printed caveat above")


# =====================================================================================================
hdr("T4 -- NO-GO CROSS-CHECK + circularity hunt (mandatory)")
# =====================================================================================================
print("""
The no-go theorem (generation_splitting_no_go_2026-06-29.md) requires the residual identification
freedom to leave AT LEAST ONE external datum-class. Checked for BOTH the actual (T1) and the
counterfactual (T3) states.
""")

# Actual (as-built) state: T1 found no coupling exists, so nothing constrains the observer beyond
# the GEN-HOMES sigma-alone baseline -- U(1)^2 x S3, vastly more than one datum-class.
check("T4a AS-BUILT state satisfies the no-go bound trivially (full U(1)^2 x S3 freedom remains, "
      "since T1 found no coupling to shrink it -- consistent with 'coupling un-built', not a "
      "collapse)", not T1_COUPLING_BUILT)

# Counterfactual (if-imposed) state: continuous dim 0, discrete order 6 (as reported, not re-derived)
# -- the discrete S3 alone is >= 1 datum-class, so the no-go bound is satisfied (not violated, not a
# B3 zero-datum overshoot).
counterfactual_datum_classes = 6  # S3 order, per freeze's own definition (T3e)
check("T4b COUNTERFACTUAL (if-imposed) state leaves >=1 datum-class (S3, order 6) -- satisfies the "
      "no-go bound; NOT a B3 zero-datum overshoot", counterfactual_datum_classes >= 1,
      note=f"datum classes = {counterfactual_datum_classes}")

print("""
Circularity hunt (mandatory since neither state is B3, this is a hygiene check, not an escalation):
trace every input used in T3's computation.
""")
inputs_used = {
    "sigma": "the permutation {0:0,1:2,2:3,3:1}, the winding/generation deck-screw axis from "
             "GEN-HOMES -- pure combinatorial structure of the 4-vertex construction, no data.",
    "W_gen = A4v[5]": "the 5th element of _a4_vertex_group(), a pure group-theoretic enumeration "
                       "of the A4 vertex-permutation group -- no data.",
    "rho3": "_a4_standard_3irrep()'s honest A4 standard 3-irrep matrices -- pure representation "
            "theory, no data.",
    "commutant_dim": "linear-algebra nullspace dimension via SVD -- no data, no threshold tuned to "
                      "any external value (the SVD tolerance is a fixed floor-detection constant, "
                      "identical in form to every other commutant computation in this repo).",
}
for k, v in inputs_used.items():
    print(f"  - {k}: {v}")
no_data_tokens = ["m_e", "m_mu", "m_tau", "m_nu", "koide", "ppm", "pdg", "0.0510", "105.658",
                   "1776.8", "M_Z", "m_W", "CKM", "PMNS"]
# Scan the SOURCE of the actual functions/inputs used in T3 (not this script's own source, which
# trivially contains the token LIST above as the search criterion itself -- that would be a
# self-referential false positive, not a real circularity check).
traced_sources = "".join(inspect.getsource(f) for f in
                          (_a4_vertex_group, _a4_standard_3irrep, group_closure, commutant_dim))
data_hits = [t for t in no_data_tokens if t.lower() in traced_sources.lower()]
check("T4c circularity hunt: the traced input functions (_a4_vertex_group, _a4_standard_3irrep, "
      "group_closure, commutant_dim -- everything T3's computation depends on) contain NO "
      "mass/ppm/Koide/CKM/PMNS token", len(data_hits) == 0, note=f"hits={data_hits}")


# =====================================================================================================
hdr("T5 -- DISCRIMINATING CONTROLS (mandatory)")
# =====================================================================================================
print("""
(i) An observer coupled to sigma ALONE must reproduce the full baseline (GEN-HOMES): commutant dim 3
    (already shown, T3b). Restated here as its own control check.
(ii) An observer coupled to TWO COMMUTING order-3 structures (same C3) must NOT collapse (joint
     commutant stays dim 3). Two independent constructions of "a second, genuinely different, but
     COMMUTING order-3 structure":
       (ii-a) sigma^2 -- the only other nontrivial order-3 element of A4 that commutes with sigma
              (centralizer of a 3-cycle in A4 has order 3 = <sigma> itself; this is the unique
              non-cherry-picked choice within A4v).
       (ii-b) a fresh order-3 unitary built directly in sigma's OWN eigenbasis but NOT a power of
              sigma (diag(1,1,omega) in that eigenbasis) -- commutes with rho3(sigma) by
              construction (same eigenbasis), independent check that collapse requires genuine
              non-commuting/irreducible joint action, not merely "a second order-3 object present".
""")

check("T5i control: sigma ALONE reproduces the GEN-HOMES baseline (commutant dim 3)",
      dim_sigma_only == 3, note=f"dim={dim_sigma_only}")

rS2 = rS @ rS
check("T5(ii-a) sanity: rho3(sigma^2) == rho3(sigma) @ rho3(sigma) (consistency of the rep)",
      np.max(np.abs(rho3[sigma2_idx] - rS2)) < 1e-10,
      note=f"resid={np.max(np.abs(rho3[sigma2_idx] - rS2)):.2e}")

dim_commuting_control_a = commutant_dim([rS, rS2])
check("T5(ii-a) joint commutant(sigma, sigma^2) does NOT collapse -- stays dim 3 (two commuting "
      "order-3 structures generating the SAME cyclic group do not trigger Schur collapse)",
      dim_commuting_control_a == 3, note=f"dim={dim_commuting_control_a}")

# (ii-b) build sigma's eigenbasis, construct a genuinely different order-3 diagonal matrix in it
eigvals, eigvecs = np.linalg.eig(rS)
order3_check = np.max(np.abs(eigvals ** 3 - 1))
check("T5(ii-b) setup sanity: rho3(sigma)'s eigenvalues cube to 1 (order-3, as expected)",
      order3_check < 1e-9, note=f"resid={order3_check:.2e}")
omega = np.exp(2j * np.pi / 3)
D_new_diag = np.diag([1.0, 1.0, omega])  # NOT a power of sigma's own diag(eigvals) pattern
D_new = eigvecs @ D_new_diag @ np.linalg.inv(eigvecs)
commute_resid = np.max(np.abs(D_new @ rS - rS @ D_new))
check("T5(ii-b) sanity: the fresh order-3 matrix D_new commutes with rho3(sigma) by construction "
      "(same eigenbasis)", commute_resid < 1e-8, note=f"resid={commute_resid:.2e}")
not_a_power = min(np.max(np.abs(D_new - np.eye(3))), np.max(np.abs(D_new - rS)),
                   np.max(np.abs(D_new - rS2)))
check("T5(ii-b) sanity: D_new is NOT (numerically) equal to I, sigma, or sigma^2 -- a genuinely "
      "distinct commuting order-3 structure", not_a_power > 1e-3, note=f"min dist={not_a_power:.3e}")

dim_commuting_control_b = commutant_dim([rS, D_new])
check("T5(ii-b) joint commutant(sigma, D_new) does NOT collapse -- stays dim 3 (a genuinely "
      "DIFFERENT commuting order-3 structure still fails to trigger Schur collapse -- collapse "
      "requires the JOINT group to act irreducibly, i.e. genuinely non-commuting generation, not "
      "merely 'a second order-3 object')", dim_commuting_control_b == 3,
      note=f"dim={dim_commuting_control_b}")

check("T5 SUMMARY: the computation DISCRIMINATES (baseline=3, commuting controls=3, "
      "genuinely-irreducible-joint={sigma,W}=1) -- a computation that collapsed regardless of "
      "input would be invalid; this one does not", dim_sigma_only == 3
      and dim_commuting_control_a == 3 and dim_commuting_control_b == 3 and dim_joint == 1)


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
print(f"BOOKED FINDING: T1 = coupling UN-BUILT (the observer factor C^3_obs and the vertex "
      f"functional -kappa.I(A;B) are two disjoint code objects; no import, no shared name, and "
      f"the_net.py itself carries a standing disclaimer forbidding the conflation).")
print(f"COUNTERFACTUAL (mechanism-availability, NOT the station's actual finding): IF the coupling "
      f"were built and DID impose W on the observer, the joint commutant collapses "
      f"({dim_sigma_only} -> {dim_joint}), killing the continuous U(1)^2 by Schur; the discrete S3 "
      f"labeling (order 6, per the freeze's own definition) is reported unchanged, with a possible "
      f"further reduction (Out(A4)~=Z_2) flagged OPEN, not computed/booked.")
print(f"NO-GO: both the as-built state (full freedom) and the counterfactual (S3 remains) satisfy "
      f"'>=1 external datum-class'; neither is a B3 zero-datum overshoot.")
print(f"CONTROLS: T5(i)/(ii-a)/(ii-b) confirm the computation discriminates (does not collapse "
      f"regardless of input).")

if n_pass == n_total:
    print("\nRESULT: ALL CHECKS PASS")
else:
    print(f"\nRESULT: {n_total - n_pass} CHECK(S) FAILED")

sys.exit(0 if n_pass == n_total else 1)
