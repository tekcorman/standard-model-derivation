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

# NOTE (panel C6, 2026-06-12): suite greenness is LOAD-CONDITIONAL -- the
# heavy spectral probes are compute-bound and can exceed their budget on a
# loaded machine; a TIMEOUT is not a gate failure. Heavy entries carry a
# per-entry timeout (4th tuple element, seconds).
# Backbone proofs: (category, script_path, short_description[, timeout_sec])
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
    ("foundations", "proofs/foundations/zeta_factorization_srs_srsz_2026-06-10.py",
     "Ihara/Bass zeta + Z2 mirror-cover factorization (Phase 1.1)"),
    ("foundations", "proofs/foundations/zeta_channel_dictionary_probe_2026-06-10.py",
     "Counting+winding channels as zeta functionals; N_10=120, V_us=k*^3/N_10 (Phase 1.2)"),
    ("foundations", "proofs/foundations/phase1_3_s1_mirror_is_bodycentering_2026-06-11.py",
     "Mirror = body-centering translation; srs-z = folded srs = bipartite double (Phase 1.3 S1)"),
    ("foundations", "proofs/foundations/phase1_3_zeta_sectors_parity_mirrorgirth_2026-06-11.py",
     "Zeta sectors: parity theorem, mirror girth 3=k* on <111>, screw 4+8 (Phase 1.3)"),
    ("foundations", "proofs/foundations/phase1_3_s3_winding_towers_2026-06-11.py",
     "Winding towers: 4_1 screw (4n:4) + C3 mirror (3n:3); u^8m = even screw windings (Phase 1.3 S3)"),
    ("foundations", "proofs/foundations/phase1_3_neutrino_mirror_saddles_2026-06-11.py",
     "Saddle orbit map: H = Gamma+Delta, P/N self-conjugate; holonomy inventory (Phase 1.3, panel-corrected)"),
    ("foundations", "proofs/foundations/phase1_3_trim_dichotomy_majorana_fork_2026-06-11.py",
     "TRIM dichotomy: P unique non-TRIM saddle (-P = P+Delta); Majorana fork 162.39 vs 27.05 (Phase 1.3)"),
    ("foundations", "proofs/foundations/phase1_3_c3_characters_majorana_fork_2026-06-11.py",
     "C3 characters: H/Gamma Ramanujan = regular rep (H-reading alpha=0); P splits classes (Phase 1.3)"),
    ("foundations", "proofs/foundations/phase1_3_c3_invariant_majorana_structure_2026-06-11.py",
     "C3-invariant Majorana bilinear: phase pi + degenerate pair => breaking required (Phase 1.3)"),
    ("foundations", "proofs/foundations/phase1_3_od_nspillover_takagi_2026-06-11.py",
     "N-spillover OD refutation, Takagi-correct: 15+ placements, 0 passes (Phase 1.3, panel-ordered)"),
    ("foundations", "proofs/foundations/phase1_3_w49_koide_wiring_negative_2026-06-11.py",
     "NEGATIVE: W49/Koide zero-parameter nu-Dirac wiring fails R_nu (529/142724 vs 32.57) (Phase 1.3)"),
    ("foundations", "proofs/foundations/phase1_3_L8_sector_census_2026-06-11.py",
     "L=8 sector census: 12x8 + 6x4 + 6x8 = 168 complete; address plural/unforced (Phase 1.3)"),
    ("foundations", "proofs/foundations/phase1_3_delta_eps_budget_closure_2026-06-11.py",
     "delta/eps^2 budget closure: K2-PARTIAL; delta = (2/9)/{1,2,3}, eps^2_e = k*-1 (Phase 1.3)"),
    ("foundations", "proofs/foundations/phase2_1_grover_walk_konno_sato_2026-06-11.py",
     "Bloch-Grover unitary walker + Konno-Sato law; magic/tetrahedral anchors; mirror (Phase 2.1)"),
    ("foundations", "proofs/foundations/phase2_2_born_koide_weights_2026-06-11.py",
     "Born-Koide stage 1 (panel-corrected): weights derived; P2 PARTIAL, alignment lemma open (Phase 2.2)"),
    ("foundations", "proofs/foundations/phase2_2_saddle_uniqueness_koide_2026-06-11.py",
     "Q=2/3 P-unique among C3-fixed saddles; Gamma/H non-Koide (panel-promoted corroboration)"),
    ("foundations", "proofs/foundations/phase2_2_alignment_lemma_2026-06-11.py",
     "ALIGNMENT LEMMA: conjugate phases = Hermiticity of sqrt(M); Q=(1+eps^2/2)/3 (Phase 2.2 stage 2)"),
    ("foundations", "proofs/foundations/phase2_3_page_wootters_two_clocks_2026-06-11.py",
     "Two-clock conversion consistency note (PW framing, classical impl; Phase 2.3, grade-conflicted)"),
    ("foundations", "proofs/foundations/phase3_1_davies_gkls_compression_2026-06-11.py",
     "Lindblad DERIVED: Davies-scaled compression generators GKLS; unitality no-go (Phase 3.1)"),
    ("foundations", "proofs/foundations/phase3_2_s1_step_isometry_2026-06-11.py",
     "Step isometry: 3.1 channel = visible marginal of an explicit kept record (Phase 3.2 S1)"),
    ("foundations", "proofs/foundations/phase3_2_s2_dark_face_2026-06-11.py",
     "Dark face: edge labels canonically superselected; timing coherence FINDING (Phase 3.2 S2)"),
    ("foundations", "proofs/foundations/phase3_2_s3_s4_forced_leakage_2026-06-11.py",
     "Forced dark-sink leakage; LM1 p = 1-61e^-6 = Omega_DM zero-parameter (Phase 3.2 S3+S4)"),
    ("foundations", "proofs/foundations/phase3_3_ns_gate_2026-06-11.py",
     "n_s gate: rate density kappa-flat (null persists, L6 blocked); M1 m=1 lever (Phase 3.3)"),
    ("foundations", "proofs/foundations/phase5_1_little_groups_saddle_irreps_2026-06-11.py",
     "I4_132 little groups: P doublets FORCED (nontrivial cocycle), Gamma/H triplets FORCED, N free (Phase 5.1 S1-S3)"),
    ("foundations", "proofs/foundations/phase5_1_s4_spgrep_crosscheck_2026-06-11.py",
     "spgrep cross-check: orders 24/24/12/4 + menus confirmed, P all-even independent (Phase 5.1 S4; needs spgrep)"),
    ("foundations", "proofs/foundations/phase5_1_s5_ebr_decomposition_2026-06-11.py",
     "EBR layer: band rep = two midpoint EBRs (stages); adds NO forcing (gated negative) (Phase 5.1 S5)"),
    ("foundations", "proofs/foundations/phase5_1_s6_forced_vs_free_map_2026-06-11.py",
     "48-mode map: 24 LG-forced + 2 IB clusters; 8 family signatures distinct -> 0 free bits (Phase 5.1 S6)"),
    ("foundations", "proofs/foundations/phase5_2_repricing_enumeration_2026-06-11.py",
     "A5-mass tree verified: 8!->12->4->2->Z2=1.3-orientation-bit; row awaits panel (Phase 5.2)"),
    ("foundations", "proofs/foundations/phase5_2_ss22_grain_enumeration_2026-06-11.py",
     "A5-mass at the ledger-cited Sec-2.2 grain: 24->12->6 = 2.585 bits in-row; row 3.0 (Phase 5.2, panel-ordered)"),
    ("foundations", "proofs/foundations/phase4_1_spectral_triple_srsz_2026-06-11.py",
     "Even triple on srs-z: Phi gauge-covariant; UNDRESSED J forced crossing at P; KO-2 deck class (Phase 4.1)"),
    ("foundations", "proofs/foundations/phase4_2_heat_expansion_sectors_2026-06-11.py",
     "Heat expansion (panel-corrected): 3 sectors induced + sigma propagated; ladder m2/m4|m6|m8 (Phase 4.2)", 900),
    ("foundations", "proofs/foundations/phase4_3_beta_content_jeopardy_2026-06-11.py",
     "R-19 jeopardy (panel-scoped): b3 = -7 | anchor (2HDM); no gaugino seat in the triple (Phase 4.3)", 900),
    ("foundations", "proofs/foundations/phase4_e2_sigma_census_aliasfree_2026-06-12.py",
     "ERRATUM E2: alias-free sigma census BLOCK-DISCRIMINATING (octet sign-flip); GRID2 artifact shown", 900),
    ("foundations", "proofs/foundations/phase4_4_cS_mutual_information_2026-06-11.py",
     "c_S = 2 GIVEN named identification (ratification refused); cut-correlation = 2x record (Phase 4.4)"),
    ("foundations", "proofs/foundations/phase5_2_psign_omega_identity_2026-06-11.py",
     "ORDERED CHECK: P-sign bit = omega/omega2 convention (sign=class=conj content; mirror-paired); row 2.0 (Phase 5.2)"),
    ("foundations", "proofs/foundations/phase5_2_r1_perron_vev_uniform_mode_2026-06-11.py",
     "R1 Leg A: mean functional = Gamma-Perron coordinate (torus-exact, PF); row move awaits panel (Phase 5.2/R1)"),
    ("foundations", "proofs/foundations/phase5_3_b1_kitaev_srs_anchor_2026-06-11.py",
     "Kitaev-on-srs anchor: 6 colorings; Majorana FERMI SURFACE (codim-1); Z2 gauge at quadratic level (Phase 5.3 B1)"),
    ("foundations", "proofs/foundations/phase5_3_b2_gauge_sector_placement_2026-06-11.py",
     "Gauge sector = cycle space = the 18 trivial modes; 6 rings; Gamma anomaly + stable deficit 2 (Phase 5.3 B2)"),
    ("foundations", "proofs/foundations/phase5_3_b3_bridge_global_car_2026-06-11.py",
     "THE BRIDGE: global CAR from per-node Cl(6), link dressing only; D4 = end-root case (Phase 5.3 B3, panel-reworded)"),
    ("foundations", "proofs/foundations/phase5_3_b3b_lattice_extension_2026-06-11.py",
     "Lattice extension: middle-root automorphism; star CAR; cycle two-trees = gauge within flux sector (Phase 5.3 B3b)"),
    ("foundations", "proofs/foundations/dark_feshbach_a2_closure.py",
     "Dark correction c = 5/12 (theorem-grade, A2 winding series)"),
    ("foundations", "proofs/foundations/exponent_ladder.py",
     "Feshbach exponent ladder ((k-1)/k)^(g-n_fixed)"),
    ("foundations", "proofs/foundations/hashimoto_exponents.py",
     "Hashimoto exponent enumeration (K_4 + srs verification)"),

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
    ("flavor", "proofs/flavor/vus_l2_density.py",
     "V_us = 9/40 (theorem-grade, Level 2 counting density)"),
    ("flavor", "proofs/flavor/vcb_hashimoto_bfs.py",
     "V_cb = 256/6305 (theorem-grade, A2 geometric series)"),

    # Masses
    ("masses", "proofs/masses/koide_scale_proof.py",
     "Lepton masses from Koide formula"),
    ("masses", "proofs/masses/srs_mdl_meanfield_theorem.py",
     "MDL mean-field uniquely optimal"),
    ("masses", "proofs/masses/ytau_corollary.py",
     "y_τ = α₁_full/k*² (theorem, session 25)"),

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

    # Lorentz / LIV
    ("lorentz", "proofs/lorentz/hashimoto_dispersion_symbolic.py",
     "η_lattice = 1/12 dim-6 LIV (CAS-verified to 24+ digits)"),
]

# Quick subset: fastest-running proofs for rapid checks
QUICK = [
    "proofs/foundations/toggle_arity.py",
    "proofs/gauge/cl8_verification.py",
    "proofs/cosmology/dm_hierarchy_derivation.py",
    "proofs/foundations/srs_foundation_closure.py",
    "proofs/masses/koide_scale_proof.py",
]


def run_proof(path, description, timeout_sec=360):
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
    proofs = [(e[0], e[1], e[2], e[3] if len(e) > 3 else 360) for e in BACKBONE
              if not quick or e[1] in QUICK]

    print("=" * 72)
    print("  Standard Model Derivation -- Verification Suite")
    print("=" * 72)
    print(f"  Running {'quick' if quick else 'full'} suite: "
          f"{len(proofs)} proofs\n")

    results = []
    total_time = 0

    for category, path, description, t_budget in proofs:
        sys.stdout.write(f"  [{category:12s}] {description:45s} ... ")
        sys.stdout.flush()

        passed, elapsed, output = run_proof(path, description, t_budget)
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
