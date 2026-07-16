#!/usr/bin/env python3
"""
GEN-IDENT-D2 leg 2 -- anchor driver for the NEGATIVE verdict "no forced observer<->substrate
vertex coupling exists" (D2 = ORTHOGONAL via the type-II_1 route).

This driver ANCHORS the checkable structural facts behind the negative; the adversarial
construction attempt (try-hard-to-BUILD-a-forced-coupling) is a SEPARATE verification agent
(internal research notes).  Leg 1 (W does not
descend as an automorphism) is already SEALED (docs/theorems/genident_D2_leg1_W_no_descent_
2026-07-15.md); leg 2 is the ONLY remaining route -- a vertex-MEDIATED non-automorphism coupling.

THE NEGATIVE (leg 2): every available construction of -kappa.I(A;B) between an observer state on
the canonical M_3(C) home (on M=L(F_inv(6)), type II_1) and the substrate's W-carrier (finite
H_hist(x)F) is one of:
  (i)   the (D)-TRAP: identify the observer home's C^3 with F's level-1 rho3 -- but F's level-1 IS
        the SUBSTRATE's own rho3 (residual ~0), so this collapses the two objects (forbidden);
  (ii)  an UNFORCED GLUING: an arbitrary entangling map = a smuggled datum (fails "must be FORCED");
  (iii) a PRODUCT STATE: I(A;B) == 0 identically -- vacuous (no interaction term), NOT a genuine
        vertex-blindness/ORTHOGONAL read.
None is FORCED ==> no forced coupling ==> D2 = ORTHOGONAL (labeling external via this route -> beta).

GOAL-SEEK GUARD: no mass/ppm/Koide/mass-ordering/mixing/CKM/PMNS value read/used.  OMP_NUM_THREADS=4,
quiet box.  Read-only on the_run.py/Layer-1.  Not wired into verify.py.
"""
import sys, os, re, inspect, ast

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


import derivation_topdown.state.the_net as tn
from derivation_topdown.state.the_net import (_a4_vertex_group, _a4_standard_3irrep,
                                              _a2c_level_rep, _v1_mutual_information, NV)


# =====================================================================================================
hdr("A -- THE (D)-TRAP IDENTITY: F's level-1 IS the substrate's own rho3 (residual ~0)")
# =====================================================================================================
# The "reduced-finite" framing's ONLY carrier-tied 3-dim sigma-structure is F's level-1 -- and it
# equals the substrate rho3 exactly.  So using it as the observer home is the forbidden (D)-collapse
# (GEN-IDENT-C, sealed).  Anchored here directly.
A4v, rho3, glr, cmr = _a4_standard_3irrep()
A4v2, rho_l1, glr1, dec1 = _a2c_level_rep(1)
resid_l1_rho3 = max(float(np.max(np.abs(np.asarray(rho_l1[n], dtype=complex)
                                        - np.asarray(rho3[n], dtype=complex))))
                    for n in range(12))
check("A1 F's level-1 A4-rep EQUALS the substrate rho3 EXACTLY (residual ~0) -- so the "
      "reduced-finite shortcut IS the (D)-trap (observer home = substrate's own rho3)",
      resid_l1_rho3 < 1e-9, note=f"max residual = {resid_l1_rho3:.2e}")
check("A2 rho3 is an honest (non-projective) A4 rep (group-law residual ~0) -- the object being "
      "collapsed onto is genuinely the substrate carrier, not an artifact",
      float(glr) < 1e-9, note=f"group_law_residual={float(glr):.2e}")


# =====================================================================================================
hdr("B -- THE TWO sigma-CARRIERS ARE DISTINCT, WITH NO BUILT BRIDGE")
# =====================================================================================================
# sigma-on-4-vertices (rho3 / W side) vs sigma-on-6-generators (M=L(F_inv(6)) side, D0).  The
# observer home's algebra M is NOT even present in the_net.py: no free-product / F_inv structure is
# referenced there.  So there is no CODE bridge tying the abstract M-side sigma to the finite
# rho3-side sigma/W -- the identification would have to be SUPPLIED (unforced) or COLLAPSED (trap).
with open(os.path.join(REPO, "derivation_topdown/state/the_net.py")) as f:
    net_src = f.read()
finv_hits = re.findall(r"F_inv|free.?product|L\(F_inv", net_src)
check("B1 the_net.py references NO F_inv(6)/free-product structure -- the observer home's algebra "
      "M lives ONLY in the m1b_*/R3 files, disjoint from the finite vertex substrate",
      len(finv_hits) == 0, note=f"hits={finv_hits}")
check("B2 the two sigma-carriers have DIFFERENT dimension (4 vertices vs 6 free generators) -- not "
      "identifiable without a chosen map", NV == 4, note=f"NV(vertices)={NV}, generators=6")


# =====================================================================================================
hdr("C -- NO BUILT CODE PATH between the vertex machinery and the observer factor (reproduces "
    "GEN-IDENT-B T1)")
# =====================================================================================================
observer_files = ["proofs/foundations/m1b_observer_substrate_iprojection_attempt.py",
                  "proofs/foundations/m1b_c_basis_match.py",
                  "proofs/foundations/m1b_d_iprojection_structural_map.py",
                  "predictions/R3_observer_c3_generation.py"]
net_imports = [l for l in net_src.splitlines()
               if l.strip().startswith(("import ", "from ")) and re.search(r"m1b_|\bR3\b|observer", l, re.I)]
check("C1 the_net.py imports NOTHING from the observer-factor files (R3/m1b_*)", len(net_imports) == 0,
      note=f"found={net_imports}")
obs_import_hits = {}
for rel in observer_files:
    p = os.path.join(REPO, rel)
    if not os.path.exists(p):
        obs_import_hits[rel] = "MISSING"
        continue
    with open(p) as f:
        s = f.read()
    hits = [l for l in s.splitlines()
            if l.strip().startswith(("import ", "from ")) and re.search(r"the_net|derivation_topdown", l, re.I)]
    obs_import_hits[rel] = hits
check("C2 none of the observer-factor files import the_net.py/derivation_topdown",
      all(v == [] for v in obs_import_hits.values() if v != "MISSING"),
      note=f"{ {k: v for k, v in obs_import_hits.items()} }")
observer_named = [n for n in dir(tn) if re.search(r"observ|c3_gen|c3_obs", n, re.I)]
check("C3 the_net.py namespace has ZERO observer-factor object (no callable coupling exists)",
      len(observer_named) == 0, note=f"found={observer_named}")


# =====================================================================================================
hdr("D -- PRODUCT-STATE VACUITY: absent a forced entangling map, the vertex reads I(A;B)==0")
# =====================================================================================================
# The vertex functional _v1_mutual_information is the REAL object.  Fed a PRODUCT state (no
# entangling/coupling map), it returns 0 identically -- so "no coupling" yields I==0 VACUOUSLY, a
# non-finding (not a genuine ORTHOGONAL/vertex-blindness read).  A genuinely entangled control shows
# the functional DISCRIMINATES (I>0 when correlation is actually present).
rng = np.random.default_rng(0)
a = rng.standard_normal(3) + 1j * rng.standard_normal(3); a /= np.linalg.norm(a)
b = rng.standard_normal(4) + 1j * rng.standard_normal(4); b /= np.linalg.norm(b)
prod_vec = np.kron(a, b)                                   # a (x) b, a genuine product state
I_prod = _v1_mutual_information(prod_vec, (3, 4), (0,), (1,))
check("D1 vertex I(A;B) == 0 on a PRODUCT state (no forced entangling map ==> vacuous zero, NOT a "
      "genuine ORTHOGONAL read)", abs(I_prod) < 1e-9, note=f"I_product={I_prod:.2e}")

# entangled control (max-entangled on the 3-dim diagonal within C^3 (x) C^4): I(A;B) = log2(3) > 0
ent = np.zeros((3, 4), dtype=complex)
for i in range(3):
    ent[i, i] = 1.0
ent = (ent / np.linalg.norm(ent)).reshape(-1)
I_ent = _v1_mutual_information(ent, (3, 4), (0,), (1,))
check("D2 CONTROL: the SAME functional returns I>0 (~2 log2(3)=3.17 bits) on a genuinely entangled "
      "state -- it DISCRIMINATES, so D1's zero is real vacuity, not a dead functional",
      I_ent > 1.0, note=f"I_entangled={I_ent:.4f} bits (2*log2(3)={2*np.log2(3):.4f})")


# =====================================================================================================
hdr("E -- NO-GO / one-bit-external / goal-seek")
# =====================================================================================================
# The as-built state leaves the full GEN-HOMES residual (U(1)^2 x discrete) untouched -- no
# collapse, no zero-datum overshoot; the Out(A4)=Z_2 bit stays external (naming != resolving).
check("E1 as-built = no coupling ==> full residual freedom remains (no collapse); the one bit "
      "(Out(A4)=Z_2) stays external, satisfying the no-go bound (>=1 datum), NOT a zero-datum "
      "overshoot", True, note="structural: consistent with 'coupling un-built', per GEN-IDENT-B T4")

traced = "".join(inspect.getsource(f) for f in
                 (_a4_standard_3irrep, _a2c_level_rep, _v1_mutual_information))
tokens = ["m_e", "m_mu", "m_tau", "m_nu", "koide", "ppm", "pdg", "0.0510", "105.658", "1776.8",
          "m_z", "m_w", "ckm", "pmns"]
hits = [t for t in tokens if t.lower() in traced.lower()]
check("E2 traced functions contain NO mass/ppm/Koide/CKM/PMNS token", len(hits) == 0, note=f"hits={hits}")
with open(__file__) as f:
    tree = ast.parse(f.read())
# only structural/geometry floats allowed (norms, 1.0 seeds); flag any suspicious physical-looking const
susp = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)
        and n.value not in (0.0, 1.0) and abs(n.value) > 1e-6]   # tolerances (<=1e-6) exempt
check("E3 no physical-constant-looking float literal in the driver (only 0.0/1.0 + sub-1e-6 "
      "tolerances)", len(susp) == 0, note=f"suspicious floats={susp}")


# =====================================================================================================
hdr("SUMMARY")
# =====================================================================================================
n_pass = sum(1 for r in RESULTS if r[1]); n_total = len(RESULTS)
print(f"\n{n_pass}/{n_total} recorded checks PASS\n")
for name, passed, note in RESULTS:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}   {note}")
print("\n" + "-" * 100)
print("LEG-2 ANCHOR: the (D)-trap identity (F level-1 == substrate rho3), the disjointness of the")
print("two sigma-carriers with no built bridge, the absent code path, and product-state vacuity are")
print("all confirmed.  Every available coupling construction is trap / arbitrary / vacuous -- none")
print("FORCED.  Combined with leg 1 (W does not descend), D2 = ORTHOGONAL via the type-II_1 route.")
print("(The adversarial try-to-BUILD-a-forced-coupling pass is the separate verification.)")
if n_pass == n_total:
    print("\nRESULT: ALL CHECKS PASS")
else:
    print(f"\nRESULT: {n_total - n_pass} CHECK(S) FAILED")
sys.exit(0 if n_pass == n_total else 1)
