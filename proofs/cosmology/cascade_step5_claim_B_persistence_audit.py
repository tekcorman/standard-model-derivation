#!/usr/bin/env python3
"""
Cascade Step 5 amplitude — Claim B (persistence audit)

GOAL: under the framework's renewal Markov dynamics on (Beta-posterior ×
direction) space, does the IC anisotropy α_IC = ε_toggle persist from
N = 1 (cosmological IC) to N = N_hub ≈ 10⁶¹ (observer epoch)?

This is an HONEST AUDIT. The framework's prediction H_obs = (16/15) H_sub
requires the leading anisotropic moment of the substrate's per-direction
event rate to be ε_toggle × ẑẑᵀ at observer epoch. If standard renewal
Markov mixing washes the IC anisotropy out exponentially, the framework's
claim fails under this rigorous reading.

THREE MODELS tested
-------------------

Model 1 (Direction-uniform sampling, intrinsic Beta dynamics):
  At each Planck time, one direction is sampled uniformly from the 24
  srs directed bonds. Its Bayesian state evolves via Beta conjugate update.
  IC sets ẑ-aligned to Beta(2,1), transverse to Beta(1,1). Run forward,
  measure anisotropy.
  Expected (standard Markov): exponential decay of IC anisotropy.

Model 2 (Direction-anisotropic sampling, fixed ẑ-bias):
  Sampling distribution p(e) ∝ 1 + γ × (e·ẑ)² with γ = ε_toggle. The
  anisotropy lives in the SAMPLING distribution, not the Bayesian state.
  Per-direction Bayesian state reaches isotropic stationary (π_F = 2/5,
  π_P = 3/5). Per-direction EVENT RATE = π_F P_fresh + π_P P_disconfirm
  weighted by p(e). The (1 + γ(e·ẑ)²) anisotropy is preserved in the
  event rate via the fixed sampling distribution.
  Expected: persistence, but the persistence is ASSUMED via fixed γ.

Model 3 (Non-equilibrium steady state, axis-aligned flux):
  Each direction has its own perpetual fresh-creation flux. ẑ-aligned
  flux is fresh-dominated; transverse flux is disconfirm-dominated.
  Stationary direction distribution NESS retains anisotropy.
  Expected: persistence at ε_toggle if the flux ratio matches.

The probe runs Model 1 directly (most rigorous; no fixed γ ansatz) and
reports the result. Models 2 and 3 are documented but not numerically
tested here — they would require additional structural ansatzes.

KEY QUESTION
------------
If Model 1 gives exponential decay, the framework's "α_IC = ε_toggle
persists structurally" claim is INCORRECT under direction-uniform Markov
dynamics. The framework would need to invoke either Model 2 (fixed γ
ansatz) or Model 3 (NESS) to save the prediction. Both Model 2 and 3
require an independent structural argument for the anisotropy source.

This audit is intentionally pessimistic: we test the simplest model that
would close the framework's claim WITHOUT additional structural ansatzes.
If that fails, we report honestly.
"""

import os
import sys
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def get_srs_directions():
    """Return the 24 srs directed bond unit vectors."""
    from proofs.flavor.srs_bloch_hamiltonian import build_unit_cell, find_connectivity
    verts = build_unit_cell()
    bonds = find_connectivity(verts)
    edges = np.array([dr / np.linalg.norm(dr) for _, _, _, dr in bonds])
    return edges


def fit_anisotropy(state, edges, z_hat, P_fresh, P_disconfirm):
    """Fit leading anisotropic amplitude α from per-direction rate(e)."""
    rates = np.where(state == 0, P_fresh, P_disconfirm)
    rate_avg = rates.mean()
    cos_z = edges @ z_hat
    proj_sq = cos_z ** 2
    proj_sq_centered = proj_sq - proj_sq.mean()
    # rate(e) - rate_avg ≈ α × rate_avg × (proj_sq - 1/3)
    x = rate_avg * proj_sq_centered
    y = rates - rate_avg
    if float(np.dot(x, x)) > 1e-15:
        return float(np.dot(x, y) / np.dot(x, x))
    return 0.0


def model_1_direction_uniform_simulation(n_events, edges, z_hat, rng,
                                         n_replicas=400):
    """
    Model 1: direction-uniform sampling, intrinsic Beta dynamics.

    Run n_replicas independent runs to N events each; return mean and
    std of the leading anisotropic amplitude at log-spaced sample points.

    State per direction: F (= Beta(1,1)) or P (= Beta(2,1)). IC sets
    ẑ-aligned to P, all others to F. At each step, sample direction
    uniformly from 24 srs bonds; evolve state via Beta conjugate update.
    """
    n_dirs = len(edges)
    cos_z = edges @ z_hat
    z_aligned_mask = np.abs(cos_z) > 0.9

    P_fresh = 0.5
    P_disconfirm = 1.0 / 3.0

    sample_points = np.unique(
        np.concatenate(([0], np.logspace(0, np.log10(n_events), 40).astype(int)))
    )
    sample_points = sample_points[sample_points <= n_events]

    # collect anisotropy samples across replicas at each sample point
    alphas_at_N = {int(n): [] for n in sample_points}

    for r in range(n_replicas):
        # Reset to IC
        state = np.zeros(n_dirs, dtype=int)
        state[z_aligned_mask] = 1
        # Record at N = 0
        alphas_at_N[0].append(
            fit_anisotropy(state, edges, z_hat, P_fresh, P_disconfirm)
        )

        n_done = 0
        for N_target in sample_points[1:]:
            n_to_run = N_target - n_done
            for _ in range(n_to_run):
                d = int(rng.integers(0, n_dirs))
                if state[d] == 0:  # F
                    if rng.random() < P_fresh:
                        state[d] = 1
                else:  # P
                    if rng.random() < P_disconfirm:
                        state[d] = 0
            n_done = N_target
            alphas_at_N[int(N_target)].append(
                fit_anisotropy(state, edges, z_hat, P_fresh, P_disconfirm)
            )

    # Aggregate
    summary = []
    for n in sample_points:
        arr = np.array(alphas_at_N[int(n)])
        summary.append((int(n), float(arr.mean()), float(arr.std())))
    return summary


def main():
    print("=" * 76)
    print(" Cascade Step 5 — Claim B persistence audit")
    print(" Honest test: does direction-uniform Markov mixing wash out IC?")
    print("=" * 76)
    print()

    edges = get_srs_directions()
    z_hat = np.array([0.0, 0.0, 1.0])
    n_dirs = len(edges)
    print(f"  srs directed bonds: {n_dirs}")
    print(f"  preferred axis ẑ: {z_hat}")
    print()

    # Theoretical IC anisotropy
    epsilon_toggle = (1/2 - 1/3) / (1/2 + 1/3)
    print(f"  Target ε_toggle = {epsilon_toggle:.6f} (= 1/5 exact)")
    print()

    print("  Running Model 1 (direction-uniform sampling, intrinsic Beta)...")
    print("  IC: ẑ-aligned in P state (Beta(2,1)); transverse in F state (Beta(1,1))")
    print()

    rng = np.random.default_rng(42)
    n_events = 5_000  # well past mixing time τ ≈ n_dirs × few = ~100
    n_replicas = 400
    samples = model_1_direction_uniform_simulation(
        n_events, edges, z_hat, rng, n_replicas=n_replicas
    )

    print(f"  Replicas: {n_replicas}; max events per replica: {n_events}")
    print()
    print(f"  {'N events':>10} {'⟨α⟩':>12} {'σ(α)':>10} "
          f"{'⟨α⟩/ε_toggle':>14}")
    print(f"  {'-' * 50}")
    for n, mean_a, std_a in samples:
        ratio = mean_a / epsilon_toggle
        print(f"  {n:>10} {mean_a:>12.6f} {std_a:>10.6f} {ratio:>14.4f}")

    # Late-time mean
    late = [(n, m, s) for n, m, s in samples if n >= n_events // 2]
    avg_late = float(np.mean([m for _, m, _ in late])) if late else 0.0
    std_late = float(np.mean([s for _, _, s in late])) if late else 0.0
    print()
    print(f"  Late-time average anisotropy (N ≥ {n_events // 2}, "
          f"averaged across replicas + sample points):")
    print(f"    ⟨α(t→late)⟩      = {avg_late:.6f}")
    print(f"    typical σ(α)     = {std_late:.6f}")
    print(f"    target ε_toggle  = {epsilon_toggle:.6f}")
    print(f"    ratio ⟨α⟩/ε      = {avg_late / epsilon_toggle:.4f}")
    print()

    # Assessment
    print("=" * 76)
    print(" Verdict")
    print("=" * 76)
    print()

    if abs(avg_late) < 0.05 * epsilon_toggle:
        print(" Result: IC anisotropy DECAYS to zero (or near-zero) under")
        print(" direction-uniform renewal Markov dynamics.")
        print()
        print(" Implication: the framework's α_IC = ε_toggle persistence claim")
        print(" CANNOT be derived from Model 1 (the simplest, most rigorous")
        print(" reading). The framework needs additional structural input —")
        print(" either Model 2 (fixed γ ansatz from external preferred axis)")
        print(" or Model 3 (non-equilibrium steady state with axis-aligned")
        print(" flux) — to save the prediction.")
        print()
        print(" Both Model 2 and Model 3 require structural ansatzes beyond")
        print(" the per-vertex Bayesian dynamics. The cleanest form would be:")
        print(" the cosmological preferred axis ẑ is a STRUCTURAL feature")
        print(" (not a dynamical state) with FROZEN amplitude γ = ε_toggle,")
        print(" justified by a separate structural argument (e.g., Stage 3")
        print(" causal-sector freezing of cosmological-scale anisotropy).")
        print()
        print(" Status of Claim B under Model 1: NEGATIVE.")
        print(" Path forward: Model 2 / Model 3 require independent structural")
        print(" derivation, NOT a Markov chain calculation.")

    elif abs(avg_late - epsilon_toggle) < 0.1 * epsilon_toggle:
        print(" Result: IC anisotropy PERSISTS at amplitude ≈ ε_toggle under")
        print(" direction-uniform renewal Markov dynamics.")
        print()
        print(" Implication: the framework's α_IC = ε_toggle persistence claim")
        print(" is supported by Model 1 numerics. Claim B closes at theorem")
        print(" grade once the persistence mechanism is identified analytically.")

    else:
        print(f" Result: IC anisotropy persists at AMPLITUDE α = {avg_late:.4f},")
        print(f" but this value does NOT match ε_toggle = {epsilon_toggle:.4f}.")
        print()
        print(" Implication: the framework's structural identification α = ε_toggle")
        print(" is incorrect under Model 1. The actual amplitude has a different")
        print(" structural source. Worst-case outcome — needs investigation.")

    print()
    print(" This audit deliberately tests the simplest, no-extra-ansatz model.")
    print(" A negative result here doesn't refute the framework — it tells us")
    print(" the framework's prediction requires structural input beyond")
    print(" pure renewal Markov mixing.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
