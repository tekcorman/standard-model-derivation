#!/usr/bin/env python3
"""
Verification runner for the standard-model-derivation framework.

Runs each backbone proof script as a subprocess and reports pass/fail.
All scripts are self-contained computational proofs that exit 0 on success.

Usage:
    python3 verify.py           # run all backbone proofs
    python3 verify.py --quick   # run only the 5 fastest proofs
"""

import subprocess
import sys
import time

# Backbone proofs: (category, script_path, short_description)
BACKBONE = [
    # Foundations
    ("foundations", "proofs/foundations/toggle_arity.py",
     "k*=3 optimal coordination number"),
    ("foundations", "proofs/foundations/dl_comparison.py",
     "srs unique by description length"),
    ("foundations", "proofs/foundations/srs_generation_c3.py",
     "Generation = C3 at P point"),
    ("foundations", "proofs/foundations/srs_p_point_algebra.py",
     "H^2 = k*I at P, Hashimoto eigenvalues"),
    ("foundations", "proofs/foundations/srs_ramanujan_theorem.py",
     "Ramanujan bound saturation"),
    ("foundations", "proofs/foundations/srs_foundation_closure.py",
     "All foundation theorems verified"),

    # Gauge
    ("gauge", "proofs/gauge/cl8_verification.py",
     "Cl(6) = Cl(4) x Cl(2), gauge group"),

    # Flavor
    ("flavor", "proofs/flavor/srs_unified_mixing.py",
     "All PMNS angles from h"),
    ("flavor", "proofs/flavor/srs_final_pmns_theorem.py",
     "Self-consistent PMNS, chi2/dof = 0.22 (4 obs)"),
    ("flavor", "proofs/flavor/srs_hashimoto_seesaw_proof.py",
     "CP phases from Hashimoto eigenvalue"),
    ("flavor", "proofs/flavor/srs_ckm_tree_derivation.py",
     "CKM from tree approximation at z*=17/6"),
    ("flavor", "proofs/flavor/vus_feshbach_derivation.py",
     "V_us Feshbach dark correction (unified with m_nu)"),

    # Masses
    ("masses", "proofs/masses/koide_scale_proof.py",
     "Lepton masses from Koide formula"),
    ("masses", "proofs/masses/srs_mdl_meanfield_theorem.py",
     "MDL mean-field uniquely optimal"),

    # Cosmology
    ("cosmology", "proofs/cosmology/srs_eta_b_exact.py",
     "Baryon asymmetry eta_B"),
    ("cosmology", "proofs/cosmology/dm_hierarchy_derivation.py",
     "Dark matter fraction and n_s"),

    # P2 parity sector
    ("parity", "proofs/cosmology/A_dilution_derivation.py",
     "A = 1/15 hemispherical + cubic moment (Theorems 1, 2)"),
    ("parity", "proofs/cosmology/path_c_beta_verify.py",
     "beta = sin(arg h) * alpha_EM (A-)"),
    ("parity", "proofs/cosmology/srs_photon_bloch_primitive.py",
     "B(P) doubly-degenerate h (Theorem 3)"),
]

# Quick subset: fastest-running proofs for rapid checks
QUICK = [
    "proofs/foundations/toggle_arity.py",
    "proofs/gauge/cl8_verification.py",
    "proofs/cosmology/dm_hierarchy_derivation.py",
    "proofs/foundations/srs_foundation_closure.py",
    "proofs/masses/koide_scale_proof.py",
]


def run_proof(path, description, timeout_sec=120):
    """Run a proof script, return (pass, elapsed, output)."""
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True,
            timeout=timeout_sec
        )
        elapsed = time.time() - t0
        passed = result.returncode == 0
        output = result.stdout + result.stderr
        return passed, elapsed, output
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return False, elapsed, f"TIMEOUT after {timeout_sec}s"
    except Exception as e:
        elapsed = time.time() - t0
        return False, elapsed, str(e)


def main():
    quick = '--quick' in sys.argv
    proofs = [(c, p, d) for c, p, d in BACKBONE
              if not quick or p in QUICK]

    print("=" * 72)
    print("  Standard Model Derivation -- Verification Suite")
    print("=" * 72)
    print(f"  Running {'quick' if quick else 'full'} suite: "
          f"{len(proofs)} proofs\n")

    results = []
    total_time = 0

    for category, path, description in proofs:
        sys.stdout.write(f"  [{category:12s}] {description:45s} ... ")
        sys.stdout.flush()

        passed, elapsed, output = run_proof(path, description)
        total_time += elapsed
        results.append((category, path, description, passed, elapsed))

        status = "PASS" if passed else "FAIL"
        print(f"{status}  ({elapsed:.1f}s)")

        if not passed:
            # Show last 5 lines of output on failure
            lines = output.strip().split('\n')
            for line in lines[-5:]:
                print(f"         {line}")

    # Summary
    n_pass = sum(1 for _, _, _, p, _ in results if p)
    n_fail = len(results) - n_pass

    print()
    print("=" * 72)
    print(f"  RESULTS: {n_pass}/{len(results)} passed, "
          f"{n_fail} failed, {total_time:.1f}s total")
    print("=" * 72)

    if n_fail > 0:
        print("\n  FAILURES:")
        for cat, path, desc, passed, _ in results:
            if not passed:
                print(f"    - {path}: {desc}")
        return 1

    print("\n  All proofs verified successfully.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
