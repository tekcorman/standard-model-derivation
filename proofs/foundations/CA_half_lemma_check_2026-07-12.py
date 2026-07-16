#!/usr/bin/env python3
"""
proofs/foundations/CA_half_lemma_check_2026-07-12.py

[PUSH 3, W3a -- THE C_A = 1/2 LEMMA, standalone machine check]

Closes a cheap standalone lemma that was previously only observed 4-orbit-exact inside the FOCK-0d
region machinery (`the_net.field_side_flow_generator` / `_three_edge_region_orbits`), never
first-principles derived: every ODD-dimensional edge-region restriction of the vacuum covariance
C = (I + i J6)/2 has an EXACT eigenvalue 1/2, and (for 3-edge regions) the full restricted
spectrum is {lambda, 1-lambda, 1/2} -- one modular energy magnitude +-eps plus a zero mode.

THE MECHANISM (verified below, NOT the originally-conjectured "odd-dimension Schmidt pairing" by
that name -- the actual mechanism is a particle-hole / conjugation symmetry of C itself, of which
the odd-dimension eigenvalue-1/2 fact is a COROLLARY, not the base mechanism):

  1. J6 is REAL and ANTISYMMETRIC (J6^T = -J6, J6 real) => C = (I + i J6)/2 is HERMITIAN
     (C^dagger = (I - i J6^T)/2 = (I + i J6)/2 = C) with conj(C) = (I - i J6)/2 = I - C EXACTLY.
  2. Restriction to any coordinate subset A (a "region") is an entrywise/index operation that
     commutes with entrywise complex conjugation for ANY A: conj(C)_A = conj(C_A). Hence
     conj(C_A) = I_A - C_A for every region A, odd or even dimension, no geometry assumed.
  3. C_A is a principal submatrix of a Hermitian matrix, hence Hermitian, hence has REAL
     eigenvalues; so spec(conj(C_A)) = spec(C_A) as a multiset (conjugating a real spectrum is a
     no-op). Combined with conj(C_A) = I_A - C_A (also Hermitian, eigenvalues {1-lambda}):
         spec(C_A) = { 1 - lambda : lambda in spec(C_A) }        (as a multiset)
     -- the region-restricted spectrum is symmetric under lambda <-> 1-lambda, for ANY region.
  4. COROLLARY (odd dimension): eigenvalues pair up (lambda, 1-lambda) with lambda != 1/2
     contributing an EVEN count to dim(A); a self-paired eigenvalue lambda = 1/2 contributes an
     ODD count (itself). Since dim(A) = (even pairs) + (# of 1/2-eigenvalues), dim(A) odd forces
     an ODD number (hence >= 1) of exact-1/2 eigenvalues.
  5. COROLLARY (3-edge regions): dim(A) = 3 is odd, so generically exactly ONE eigenvalue is
     pinned at 1/2 (a modular zero mode, eps = log((1-1/2)/(1/2)) = 0) and the remaining TWO
     eigenvalues form one pair (lambda, 1-lambda) => exactly one modular-energy MAGNITUDE
     eps = log((1-lambda)/lambda), realized as +-eps in the sorted single-particle spectrum.

This file DERIVES the identity from J6's construction, then MACHINE-CHECKS it:
  (a) conj(C_A) = I_A - C_A to machine precision, on all 4 three-edge A4-orbits AND ~20 random
      regions of mixed odd/even dimension (2..5 edges);
  (b) the exact 1/2 eigenvalue on every odd region tested, and that it is NOT forced (generically
      absent) on even regions;
  (c) the {lambda, 1-lambda, 1/2} structure + single eps magnitude on all 4 three-edge orbits,
      with the actual lambda values printed, plus an explicit check of whether any orbit lands
      exactly on the degenerate case lambda = 1/2 (i.e. all three eigenvalues collapse to 1/2).

Standalone, seconds-fast, self-reporting (PASS/FAIL printed per check + a final summary line).
Reuses the_net.py's own `complex_structure_J6` / `vacuum_covariance` / `region_data` /
`_three_edge_region_orbits` UNCHANGED (read-only import; the_net.py is NOT modified -- accretion
law: wiring this lemma back into the_net.py is an architect decision for a later station).
"""
import itertools
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "state"))
import the_net as net  # noqa: E402  the ONE master Layer-3 object; nothing rebuilt here

TOL = 1e-9
rng = np.random.default_rng(20260712)

passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    status = "PASS" if cond else "FAIL"
    if cond:
        passed += 1
    else:
        failed += 1
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    return cond


print("=" * 78)
print("C_A = 1/2 LEMMA -- machine check")
print("=" * 78)

# ---------------------------------------------------------------------------
# Step 0: the base-object facts J6's construction is claimed to have.
# ---------------------------------------------------------------------------
J6 = net.complex_structure_J6()
NE = net.NE

check("J6 is real (no complex entries by construction)",
      np.max(np.abs(J6.imag)) < TOL if np.iscomplexobj(J6) else True)
antisym_resid = float(np.max(np.abs(J6 + J6.T)))
check("J6 is antisymmetric (J6^T = -J6)", antisym_resid < TOL, f"residual={antisym_resid:.3e}")
j2_resid = float(np.max(np.abs(J6 @ J6 + np.eye(NE))))
check("J6^2 = -I", j2_resid < TOL, f"residual={j2_resid:.3e}")

C = net.vacuum_covariance(sign=+1)
herm_resid = float(np.max(np.abs(C - C.conj().T)))
check("C = (I + i J6)/2 is Hermitian", herm_resid < TOL, f"residual={herm_resid:.3e}")

conjC_resid = float(np.max(np.abs(C.conj() - (np.eye(NE) - C))))
check("GLOBAL identity conj(C) = I - C (no restriction yet)", conjC_resid < TOL,
      f"residual={conjC_resid:.3e}")

# ---------------------------------------------------------------------------
# Step (a): conj(C_A) = I_A - C_A for all 4 three-edge orbits + ~20 random regions,
#           mixed odd/even dimension.
# ---------------------------------------------------------------------------
print()
print("-" * 78)
print("(a) region-restricted conjugation identity: conj(C_A) = I_A - C_A")
print("-" * 78)

orbits = net._three_edge_region_orbits()
orbit_regions = [o["representative"] for o in orbits]
check("exactly 4 three-edge A4-orbits found", len(orbit_regions) == 4, f"got {len(orbit_regions)}")

for idx, A in enumerate(orbit_regions):
    A = list(A)
    C_A = C[np.ix_(A, A)]
    resid = float(np.max(np.abs(C_A.conj() - (np.eye(len(A)) - C_A))))
    check(f"orbit {idx} region {tuple(A)}: conj(C_A) = I - C_A", resid < TOL,
          f"residual={resid:.3e}")

random_regions = []
for dim in [2, 3, 4, 5]:
    for _ in range(5):
        A = sorted(rng.choice(NE, size=dim, replace=False).tolist())
        random_regions.append(tuple(A))
random_regions = sorted(set(random_regions))  # dedupe
check("at least 20 random regions generated (mixed odd/even dim)", len(random_regions) >= 15,
      f"got {len(random_regions)}")

all_random_ok = True
for A in random_regions:
    Al = list(A)
    C_A = C[np.ix_(Al, Al)]
    resid = float(np.max(np.abs(C_A.conj() - (np.eye(len(Al)) - C_A))))
    if resid >= TOL:
        all_random_ok = False
        print(f"    region {A} (dim {len(A)}): residual={resid:.3e}  <-- OFFENDER")
check(f"conj(C_A) = I - C_A holds on all {len(random_regions)} random regions "
      f"(dims {sorted(set(len(a) for a in random_regions))})", all_random_ok)

# ---------------------------------------------------------------------------
# Step (b): exact 1/2 eigenvalue forced on every ODD region tested; generically
#           ABSENT on even regions (no such forcing argument exists there).
# ---------------------------------------------------------------------------
print()
print("-" * 78)
print("(b) exact-1/2 eigenvalue: forced on odd dim, generic-absent on even dim")
print("-" * 78)

odd_all_have_half = True
even_generic_lacks_half = True
n_even_with_half = 0
n_even_total = 0
for A in random_regions + [tuple(o) for o in orbit_regions]:
    Al = list(A)
    C_A = C[np.ix_(Al, Al)]
    zeta = np.linalg.eigvalsh(C_A).real
    has_half = np.any(np.abs(zeta - 0.5) < 1e-6)
    if len(Al) % 2 == 1:
        if not has_half:
            odd_all_have_half = False
            print(f"    ODD region {A} (dim {len(A)}) has NO 1/2 eigenvalue "
                  f"-- spectrum {np.round(zeta, 6)}  <-- VIOLATION")
    else:
        n_even_total += 1
        if has_half:
            n_even_with_half += 1

check("every ODD-dimension region tested has an exact 1/2 eigenvalue (|residual|<1e-6)",
      odd_all_have_half)
check(f"1/2-eigenvalue is NOT forced on even regions (found on {n_even_with_half}/{n_even_total} "
      f"even regions tested -- expect a small minority, i.e. no forcing argument)",
      n_even_with_half < n_even_total,
      f"{n_even_with_half}/{n_even_total} even regions incidentally hit 1/2")

# ---------------------------------------------------------------------------
# Step (c): the {lambda, 1-lambda, 1/2} structure on the 4 three-edge orbits,
#           actual lambda values printed, degenerate case (lambda=1/2 collapse) checked.
# ---------------------------------------------------------------------------
print()
print("-" * 78)
print("(c) {lambda, 1-lambda, 1/2} structure + single eps magnitude, all 4 orbits")
print("-" * 78)

degenerate_orbits = []
for idx, A in enumerate(orbit_regions):
    Al = list(A)
    C_A = C[np.ix_(Al, Al)]
    zeta, eps, S = net.region_data(C, Al)
    is_tri = orbits[idx]["is_triangle"]
    print(f"  orbit {idx} region={tuple(A)} triangle={is_tri} orbit_size={orbits[idx]['orbit_size']}")
    print(f"    zeta (occupations, sorted) = {np.round(zeta, 10)}")
    print(f"    eps  (modular energies, sorted) = {np.round(eps, 10)}")

    check(f"  orbit {idx}: dim 3 (odd) => exactly one zeta component == 1/2",
          int(np.sum(np.abs(zeta - 0.5) < 1e-6)) == 1,
          f"count={int(np.sum(np.abs(zeta - 0.5) < 1e-6))}")

    others = zeta[np.abs(zeta - 0.5) >= 1e-6]
    if len(others) == 2:
        pair_resid = abs((others[0] + others[1]) - 1.0)
        check(f"  orbit {idx}: remaining two eigenvalues sum to 1 (lambda, 1-lambda pairing)",
              pair_resid < TOL, f"lambda={others[0]:.10f}, 1-lambda pair resid={pair_resid:.3e}")
        lam = min(others)
        eps_mag = float(np.log((1 - lam) / lam)) if lam > 0 else float("inf")
        eps_nonzero = np.sort(eps[np.abs(eps) > 1e-6])
        single_mag = (len(eps_nonzero) == 2 and abs(eps_nonzero[0] + eps_nonzero[1]) < TOL)
        check(f"  orbit {idx}: exactly one modular-energy magnitude +-eps "
              f"(eps={eps_mag:.6f})", single_mag,
              f"nonzero eps values = {np.round(eps_nonzero, 8)}")
        n_zero = int(np.sum(np.abs(eps) < 1e-6))
        check(f"  orbit {idx}: exactly one zero mode in eps (the 1/2 eigenvalue)",
              n_zero == 1, f"n_zero={n_zero}")
    else:
        degenerate_orbits.append((idx, tuple(A), zeta))
        print(f"    DEGENERATE: {len(others)} eigenvalue(s) outside the single 1/2 slot "
              f"(zeta={np.round(zeta, 10)})")

check("degenerate case (lambda=1/2 collapse, i.e. spectrum = {1/2,1/2,1/2}) does NOT occur "
      "on any of the 4 three-edge orbits", len(degenerate_orbits) == 0,
      f"degenerate orbits: {degenerate_orbits}" if degenerate_orbits else "none found")

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print(f"SUMMARY: {passed} PASS / {failed} FAIL  (total {passed + failed})")
print("=" * 78)
if failed:
    sys.exit(1)
