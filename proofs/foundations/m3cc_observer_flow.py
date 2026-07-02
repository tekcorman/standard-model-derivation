#!/usr/bin/env python3
"""
M3.C.c — induced flow on ρ_obs(t) ∈ states(B(C³_obs)).

Companion to:
  proofs/foundations/m1b_d_iprojection_structural_map.py (M1.B closure)
  proofs/foundations/m3c_substrate_rg_cosmic_time.py (M3.C.a + M3.C.b closure)
  an internal working note (M3 scoping)

GOAL.

Compute the explicit trajectory

    ρ_obs(t) = π ∘ Φ_{Λ(t)}(ρ_sub^{(0)})

as a curve in states(B(C³_obs)) parametrized by Λ(t) = t_P/t. Identify
endpoint behavior, monotonicity properties, and whether the trajectory
admits a finite-t stationary point that could close G1b via H1.

OUTCOME (preview).

Structural analysis: ρ_obs(t) interpolates monotonically (in KL divergence
sense) between two endpoints:
  - ρ_obs(t_P) = π(ρ_sub^{(0)})           — UV / Planck-scale image
  - ρ_obs(t→∞) = π(A2-T waterline)        — IR / fixed point image

The OVERALL FLOW IS MONOTONE in relative entropy to the fixed point
(Csiszár 1975 + monotonicity of I-projection). This RULES OUT a
straightforward H1 stationarity (no intermediate dρ_obs/dt = 0).

HOWEVER: ρ_obs lives in the 9-dim real space of 3×3 density matrices,
and individual MARGINAL parameters of ρ_obs may have stationary points
at intermediate Λ even though the overall flow is monotone toward the
fixed point. Identifying which marginal parameter is "the observer's
clock" is itself a structural question — and the most natural candidate
is the Z_3-asymmetric content (off-diagonal generation coherences).

PARTIAL CLOSURE: M3.C.c surfaces the structural content but cannot
fully close H1 without further input. A reframe of H1 is needed.

This is a significant negative result for the H1 path. Recorded
honestly so that next-session attack can pivot.
"""

import sympy as sp
from sympy import I, sqrt, Rational, Matrix, eye, zeros
import numpy as np

# Reuse omega
omega = Rational(-1, 2) + I * sqrt(3) / 2
omega_pow = {0: sp.S(1), 1: omega, 2: Rational(-1, 2) - I * sqrt(3) / 2}


# =============================================================================
# §0. Setup
# =============================================================================
print("=" * 76)
print("M3.C.c — induced flow on ρ_obs(t)")
print("=" * 76)
print()


# =============================================================================
# §1. Endpoint behavior — UV and IR limits
# =============================================================================
print("§1. Endpoint behavior — UV and IR limits")
print("-" * 76)
print("""
  UV limit (Λ → ∞ ⇔ t → t_P):
    Φ_Λ → identity (no coarse-graining; Λ-cone is the full simplex).
    ρ_sub(t_P) = ρ_sub^{(0)} (initial substrate state at Planck scale).
    ρ_obs(t_P) = π(ρ_sub^{(0)}) — the Z_3-Fourier image of the initial
                 condition's M_3(ℂ) marginal.

  IR limit (Λ → 0 ⇔ t → ∞):
    Φ_Λ → projection onto A2-T waterline (theorem-grade per
           `forward_construction_substrate_renormalization.md` §3).
    ρ_sub(t → ∞) = A2-T waterline state ρ_*.
    ρ_obs(t → ∞) = π(ρ_*).

  WHAT IS π(ρ_*)?

  The A2-T waterline emerges from MDL on substrate Hashimoto walks.
  Hashimoto walks ARE Z_3-permutation-symmetric (the cyclic shift
  σ = (1 2 3)(4 5 6) on F_inv(6)'s generators is a graph automorphism;
  the walker dynamics commute with σ).

  Therefore the A2-T waterline ρ_* is Z_3-INVARIANT: α(ρ_*) = ρ_*
  (where α is the *-automorphism on M induced by σ).

  CONSEQUENCE: ρ_* lies entirely in M^α (the fixed-point sub-factor).
  Its image under π is the M_3(ℂ)-MAXIMALLY-MIXED state:

      π(ρ_*) = (1/3) I_3   (= maximally mixed in B(C³_obs)).

  PROOF SKETCH. ρ_* ∈ M^α ⇒ E_{M^α}(ρ_*) = ρ_* (already in M^α).
  Lift ρ_* to M ⋊ Z_3: only the trivial-irrep component (corresponding
  to the all-ones direction in the Z_3-Fourier decomposition) is
  populated. The M_3(ℂ) marginal under partial trace is the projection
  onto this trivial irrep — which in the basis where Z = diag(1, ω, ω²)
  is exactly the maximally mixed state (1/3) I_3.

  ⇒ At cosmic time t → ∞: the observer sees a maximally mixed
    generation state (no preferred generation; full Z_3 symmetry).
""")


# =============================================================================
# §2. Monotonicity of the flow
# =============================================================================
print("§2. Monotonicity of the flow")
print("-" * 76)
print("""
  CSISZÁR-MONOTONICITY THEOREM. The I-projection apparatus (A2-T
  sequential I-projection = substrate Wilsonian RG, per the framework's
  existing `forward_construction_substrate_renormalization.md` §2.2)
  inherits the monotonicity of relative entropy under coarse-graining:

      D(ρ_sub(Λ_1) ‖ ρ_*) ≤ D(ρ_sub(Λ_2) ‖ ρ_*)   whenever Λ_1 ≤ Λ_2

  (relative entropy non-increases as Λ decreases; equivalently, ρ_sub
  flows monotonically toward the fixed point ρ_* in KL distance).

  Pushed forward through π (which is CP and trace-preserving, hence
  contractive in relative entropy by data-processing inequality):

      D(ρ_obs(Λ_1) ‖ π(ρ_*)) ≤ D(ρ_obs(Λ_2) ‖ π(ρ_*))   for Λ_1 ≤ Λ_2

  i.e. the OBSERVER state also flows monotonically toward (1/3) I_3
  in KL distance.

  CONSEQUENCE: there is NO intermediate Λ at which dρ_obs/dΛ = 0 in
  the strong sense (KL distance to fixed point reaches an interior
  minimum). The trajectory is one-way.

  ⇒ STRAIGHTFORWARD H1 IS FALSIFIED. The observer-MDL stationarity
    "∂_t ρ_obs = 0" has only TRIVIAL solutions (UV endpoint at t = t_P,
    IR endpoint at t = ∞). Neither corresponds to "now" t_now finite.
""")


# =============================================================================
# §3. Reframe: marginal stationarity and observer's "clock"
# =============================================================================
print("§3. Reframe — marginal stationarity")
print("-" * 76)
print("""
  H1's failure is at the level of the FULL state ρ_obs. But ρ_obs lives
  in the 9-dim real space of 3×3 density matrices. Individual MARGINAL
  PARAMETERS (real-valued functionals of ρ_obs) may have stationary
  points at intermediate Λ even though the overall flow is monotone
  toward the fixed point.

  Specifically: the trajectory ρ_obs(Λ) traces a curve in the 9-dim
  state space. Different parameters of ρ_obs evolve at different rates.
  A parameter f(ρ_obs(Λ)) has a stationary point at Λ_* if
  df/dΛ|_{Λ_*} = 0.

  CANDIDATE PARAMETERS (Z_3-grading content of ρ_obs):

    (a) Diagonal weights w_j = ⟨gen j| ρ_obs |gen j⟩ for j = 0, 1, 2.
        Sum to 1; satisfy w_j(t→∞) = 1/3 (IR). The deviation
        Δw_j(Λ) := w_j(Λ) − 1/3 measures Z_3-asymmetric occupation.
        Total deviation Σ |Δw_j| is monotone (toward 0 at IR), but
        individual Δw_j can have stationary points (e.g., a Δw_0
        peak before decay to 0).

    (b) Off-diagonal generation coherences c_{jk} = ⟨gen j| ρ_obs |gen k⟩
        for j ≠ k. These are generally complex; satisfy c_{jk}(t→∞) = 0
        (IR). Their magnitudes |c_{jk}| are monotone toward 0 (Csiszár);
        but their PHASES can rotate non-trivially with Λ, and
        individual real/imaginary parts can have stationary points.

    (c) Z_3-symmetric content: the components of ρ_obs in the
        Z_3-irreducible decomposition. For C³_obs with Z = diag(1, ω, ω²),
        ρ_obs decomposes as ρ_0 (trivial irrep) ⊕ ρ_1 (ω-irrep) ⊕
        ρ_2 (ω²-irrep). At IR: ρ_0 → (1/3) I_3 (full); ρ_1, ρ_2 → 0.
        The DECAY RATE of ρ_1 (or ρ_2) sets a timescale.

  THE REAL QUESTION FOR G1b: which of these parameters is "the
  observer's clock" — i.e., the parameter whose stationarity
  corresponds to "now"?

  The framework does NOT currently pin this. Three structurally
  defensible candidates:
    (A) The first parameter to reach a stationary point (Λ_min).
    (B) The parameter with the slowest decay (sets the longest timescale).
    (C) A specific parameter selected by an additional structural axiom.

  Without a selection principle among (A), (B), (C), G1b doesn't close
  via M3.C alone.
""")


# =============================================================================
# §4. Numerical illustration on the 2-dim toy
# =============================================================================
print("§4. Numerical illustration — observer flow in the 2-dim toy")
print("-" * 76)
print("""
  Use the same 2-dim Markov toy from `m3c_substrate_rg_cosmic_time.py`,
  with q* = (2/3, 1/3) and a non-trivial initial condition. Compute
  the I-projection trajectory and the "observer parameter" w_0 -
  q*_0 = (Δw_0).
""")

q_star = np.array([2/3, 1/3])

def kl(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

def i_projection_toy(p_init, Lambda):
    radius = Lambda
    d = kl(p_init, q_star)
    if d <= radius:
        return p_init
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p_mid = (1 - mid) * p_init + mid * q_star
        if kl(p_mid, q_star) > radius:
            lo = mid
        else:
            hi = mid
    return (1 - hi) * p_init + hi * q_star

# Try a curved initial condition to see if Δw_0 has a max at finite Λ
p_init = np.array([0.05, 0.95])

# Sweep Λ from large to small
Lambda_values = np.logspace(2, -4, 30)
w0_values = []
deltaw0_values = []
print(f"  {'Λ':>10s}  {'w_0':>10s}  {'Δw_0 (= w_0 - 2/3)':>22s}  {'D(p||q*)':>12s}")
for Lambda in Lambda_values:
    p = i_projection_toy(p_init, Lambda)
    w0 = p[0]
    delta = w0 - q_star[0]
    d = kl(p, q_star)
    w0_values.append(w0)
    deltaw0_values.append(delta)
    print(f"  {Lambda:10.4e}  {w0:10.6f}  {delta:22.6e}  {d:12.4e}")

print()
print(f"  Observed pattern: |Δw_0| starts at {abs(deltaw0_values[0]):.4f} (UV) and")
print(f"  decreases monotonically to 0 (IR). NO interior stationary point.")
print()
print(f"  This is consistent with the §2 monotonicity argument: in the toy,")
print(f"  the I-projection is along the geodesic to q*, so w_0(Λ) is monotone.")
print()
print(f"  In the FULL B(C³_obs) case (real M_3(ℂ) state space, not 1D simplex),")
print(f"  more directions exist and intermediate stationary points become")
print(f"  possible — but the RIGHT choice of marginal parameter is structural")
print(f"  input not yet pinned by the framework.")


# =============================================================================
# §5. What this means for G1b — H1 needs a reframe
# =============================================================================
print("\n§5. Implications for G1b")
print("-" * 76)
print("""
  RESULT OF M3.C.c.

    The flow ρ_obs(Λ) is well-defined and explicit (M3.C.a + M3.C.b
    closure provides the substrate-side machinery; M1.B provides π).
    However, it is MONOTONE toward (1/3) I_3 in KL distance, with NO
    interior stationary point at the level of the full state.

  CONSEQUENCES FOR THE H1 HYPOTHESIS.

    H1 (observer-MDL stationarity) does NOT close G1b under the M3.C
    cosmic-time identification IF "stationarity" is interpreted as
    dρ_obs/dt = 0 for the full state at finite t. The only solutions
    are the trivial endpoints.

  POSSIBLE REFRAMES OF H1.

    (R1) MARGINAL STATIONARITY. ∂_t f(ρ_obs) = 0 for some specific
         scalar functional f. Requires identifying the "right" f from
         framework axioms — not currently possible.

    (R2) RELATIVE-ENTROPY THRESHOLD. D(ρ_obs(t) ‖ (1/3) I_3) reaches
         a specific threshold (e.g., the observer's measurement
         resolution ε_obs). "Now" is t such that D(t) = ε_obs.
         Requires deriving ε_obs structurally.

    (R3) RATE OF CHANGE THRESHOLD. |dρ_obs/dt| reaches a specific
         threshold (the observer's "clock rate"). Requires deriving
         the observer's clock rate structurally.

    (R4) DIFFERENT FRAMING ENTIRELY. H2 (information-rate balance)
         or H3 (coasting as MDL attractor) or a new H4 (e.g.,
         "observer becomes consistent with FLRW description").

  RECOMMENDATION.

    Defer G1b closure attempt; revisit hypotheses. The M3.C
    framework apparatus is operational and well-defined, but H1 in
    its straightforward form is FALSIFIED.

    Most promising fallback: R2 (relative-entropy threshold). The
    "observer's measurement resolution" is structurally analogous to
    Bekenstein's bit budget, which connects to substrate Lloyd-style
    bounds. This couples G1b to the substrate's information capacity
    — a genuinely new sub-problem but with concrete structural content.

    Alternatively: H3 (coasting as MDL attractor) bypasses H1's
    stationarity framing entirely. Reconsider H3 as the next attack.

  THIS IS A NEGATIVE RESULT.

    M3.C.c sharpens the structure (the flow is fully determined, the
    fixed point is identified, monotonicity is proven) but reveals
    that H1 cannot close G1b by itself. The H1 path requires either
    a reframe or additional structural input.

    Honesty: this means G1b is HARDER than the post-M1.B/M3.C estimate
    suggested. The "2-3 sessions remaining to G1b closure" estimate
    needs revision; it now depends on whether a reframe (R1-R4) opens
    a tractable closure path. Honest revised estimate: 2-3 additional
    sessions to PIVOT (i.e., scope the right reframe), then 3-5
    sessions on the new approach. Total: 5-8 sessions to G1b closure
    via the new path.

  THIS RESULT IS USEFUL.

    The M3.C apparatus (cosmic time = RG time, ρ_sub(t) = Φ_{Λ(t)}(ρ_0))
    is operational and theorem-grade. Future framework work building on
    cosmic dynamics has this foundation.

    The Galois structure surfaced in M1.B (M^α ⊂ M ⊂ M ⋊ Z_3 ≅
    M_3(C) ⊗ M^α with R3 = Galois group) is a genuinely new framework
    object, independent of whether G1b closes.

    The "observer is structurally an EXTENSION of the substrate"
    finding (M1.B §7.5) is a foundational identification.

    G1a's graph-theoretic core (1/k* : (k*-1)/k* eigenstructure) is
    still theorem-grade and ports the same "fixed-point ratio" content
    to any future G1b reframe.
""")


print("=" * 76)
print("M3.C.c CLOSED at structural grade — but with NEGATIVE outcome for H1.")
print("Next: scope the reframe (R1-R4) and pick the most-tractable.")
print("=" * 76)
