# β cosmic-birefringence coefficient — partial closure (NOT theorem-grade)

**Date:** 2026-04-25 (PM); **updated 2026-04-28 evening late** with four new structural constraints from bounded-route falsifications.
**Status:** **NOT CLOSED.** Schur + T-symmetry pins V_chir's STRUCTURE to c·diag(+1,−1) on L⊕R; Lemmas 1+2 fix its DIMENSIONAL CONTENT to sin(arg h); the explicit MAGNITUDE c = 1 (i.e. no 1/2, 1/k*, n_g/something prefactor) is **NOT derived by an operator-level Feshbach chain** analogous to F0–F3 for the Higgs vertex.

**2026-04-28 update.** Four bounded-session closure attempts are now exhausted (L3-tree, P4 Cl(6,0) γ_7, L3-trace-survey, Approach Q ∂_k B 1-loop). Each failed with a sharp structural cause that constrains what the F0_γ–F3_γ+Spectral_γ recipe must satisfy. See §"Constraints from bounded-route falsifications" below for the four new structural inputs.

This document records what IS derived, exactly where the gap remains, and why a structural-counting analogy alone does not suffice for Nature-grade claim.

---

## What this document closes

### Step A: photon C₃-irrep structure (CLOSED, theorem-grade)
- Constructed C₃_v (4×4) and C₃_e (6×6, Bloch-phase orientation flips) at k_P; chain map `d·C₃_v = C₃_e·d` exact (max error 0).
- `[C₃_e, Δ_1] = 0` to **2.81e-15** (machine precision).
- Photon transverse eigenspace at ω² = 36 splits as **ω ⊕ ω²** under C₃ (trace = −1, det = +1; eigenvalues exactly e^{±2πi/3}).
- L (helicity +1) ↔ ω-irrep, R (helicity −1) ↔ ω²-irrep, by standard photon-helicity correspondence along [111].

Script: `proofs/cosmology/srs_photon_c3_chainmap.py`.

### Step B (partial): Schur+T-symmetry constrains V_chir SHAPE
Any C₃-invariant T-odd operator on L⊕R must have form `c·(|L⟩⟨L| − |R⟩⟨R|)` for some real c. (L and R are inequivalent C₃-irreps ω vs ω², so off-diagonals vanish by Schur; T exchanges helicities, forcing antisymmetric diagonal.)

### Step C (partial): Lemma 2 fixes V_chir DIMENSIONAL CONTENT
The substrate quantity coupling to a unit-vector polarization observable must be unit-bounded and parity-odd. By Lemma 1 (bit-cost ranking under specified description language), the cost-minimum such functional is sin(arg h). So c involves sin(arg h) as the "natural scale" — but the EXACT magnitude (whether c = sin(arg h), c = (1/2)·sin(arg h), c = k*·sin(arg h), etc.) is not fixed by Lemma 2 alone.

---

## What is NOT closed

The c = 1 step (i.e. no further structural prefactor) requires:

**An explicit operator V_chir on the photon Hodge bundle whose restriction to L⊕R is ±sin(arg h)·diag(+1, −1) — derived from first-principles A2-edge-process counting on the photon's coupling structure, with each prefactor (analog of n_g, k*², N_ATOMS) accounted for.**

Naive candidate operators tested (in `proofs/cosmology/srs_photon_chirality_coefficient.py`):
- π†·Im(B)·π (symmetric lift): yields V_chir = 0 because Im(B) lives in the parity-odd sector under bond orientation reversal.
- π†·B·π then `(B − B†)/(2i)`: yields V_chir = 0 because B becomes effectively Hermitian on the symmetric subspace.
- (B − Bᵀ)/(2i), (B² − (Bᵀ)²)/(2i), etc.: all give c_L = c_R on the photon space (act as scalars on the eigenspace).

None of these is the "right" V_chir. The right V_chir is a CROSS-SECTOR coupling between the photon Hodge bundle and the dark walker, and constructing it explicitly is an open research item.

Existing β.E perturbation tests (`proofs/lorentz/birefringence_c3_irrep_O2.py`):
- Under helicity-correct L3 identification, V_proj gives **identically zero** at first order.
- Under L1, V_proj gives Im(h) = √5/2 (the MDL-more-expensive term, not sin(arg h)).
- Under L2, similar.

So no first-order perturbation reproduces sin(arg h)·α_EM cleanly, and the c = 1 magnitude remains an MDL-leading-term ASSERTION rather than an operator-level derivation.

---

## Comparison with Higgs 5/12 — what would close c = 1 by analog

`proofs/foundations/dark_feshbach_a2_closure.py` derives c = n_g/(k*²·N_ATOMS) = 5/12 via:

| Step | Statement | Gate |
|---|---|---|
| F0 | A2 edge process: vertex coupling = ALL k* outgoing × k* incoming directed-edge pairs | Derived from A2's definition |
| F1 | Σ_v has k*² = 9 algebraic terms | `H_QP·H_PQ = adjacency` identity |
| F2 | Backtrack pairs (i,i) contribute 0 girth cycles | Simple-cycle theorem |
| F3 | n_g = 15 unoriented (not 30 oriented) under A2-T | C/C̄ MDL-equivalent |
| Spectral | 1/N_ATOMS factor | `H(k_P)² = k*·I_4` theorem |
| Result | c = 15 / (9·4) = 5/12 | Arithmetic |

Each factor is a COUNTED QUANTITY from explicit lattice structure.

The β analog requires:
- **F0_γ**: an A2-style edge-process statement for the photon Hodge mode at k_P. (What is the analog of "vertex couples to all k* incident edges" for the photon?)
- **F1_γ**: an algebraic identity giving the "edge-pair multiplicity" for the photon mode. (Is it 1, k*, k*², or something else?)
- **F2_γ**: a zero-contribution theorem for some part of the sum. (For Higgs it was backtracks; what's the analog for photons?)
- **F3_γ**: a counting reduction (analog of unoriented-vs-oriented). (For Higgs it was C/C̄ identification; what's the analog for photons?)
- **Spectral_γ**: the Hodge-spectral identity giving the per-mode normalization. (Analog of `H(k_P)² = k*·I_4`; presumably involving Δ_1 transverse spectrum.)

None of these has been worked out for the photon case. Without them, c = 1 is asserted by analogy, not derived.

---

## Constraints from bounded-route falsifications (added 2026-04-28 evening)

Four bounded single-session approaches have been attempted and falsified. Each result is structurally informative — the F0_γ–F3_γ+Spectral_γ recipe must respect all four.

| # | Approach | Script | Result | Constraint on the recipe |
|---|---|---|---|---|
| 1 | L3-tree: π†·Im(B)·π onto photon | `proofs/cosmology/srs_photon_chirality_coefficient.py` | c = 0 | F0_γ vertex CANNOT be Im(B) projected via π — Im(B) lives in parity-ODD bond-reversal sector, photon Hodge bundle is parity-EVEN |
| 2 | P4: Cl(6,0) γ_7 transferred via B6, π†·Γ_7^12·π | `proofs/foundations/arg_h_path_b_p4_cl60_gamma5_attempt.py` | c = 0 gauge-inv | **Photon Hodge bundle ⊥ V_Ram at k_P** (overlap ~10⁻¹⁵). 12-dim = V_Ram (8) ⊕ V_kernel (4); photon ⊂ V_kernel entirely. F0_γ vertex must operate in V_kernel or as an explicit V_Ram↔V_kernel cross-sector coupling |
| 3 | L3-trace-survey: tr_ω[γ_7·F(B)] on V_Ram C_3 sectors | `proofs/foundations/arg_h_path_b_l3_{trace_survey,gauge_check}.py` | apparent c=2 on ω, **gauge-DEP** | **γ_7 transferred via B6 has irreducible U(2) gauge ambiguity** on each C_3 non-trivial sector. F0_γ must NOT use γ_7 as the chirality projector — γ_7 is preserved only by U(1)×U(1) ⊂ U(2), the U(2)/(U(1)×U(1)) ≅ S² coset moves γ_7 eigenstates |
| 4 | Q: ∂_k B 1-loop self-energy with kinematic vertex | `proofs/foundations/arg_h_path_b_q_qed_1loop_attempt.py` | parity-EVEN, no sin(arg h) | F0_γ vertex CANNOT be ∂_k B at single momentum — gives Re(h) mass shift instead of Im(h) phase rotation. (M_LL−M_RR)/2 = -|c|²·(1/h+1/h̄) = -|c|²·√3/2, parity-EVEN |

### Sharpened recipe constraints

The F0_γ–F3_γ+Spectral_γ chain that closes c = 1 must:

**C1 (from L3-tree):** Avoid Im(B) projection through π. If a parity-odd structure is built, it must NOT lie purely in the bond-reversal-odd sector.

**C2 (from P4):** Operate as an explicit cross-sector coupling V_chir : V_kernel → V_Ram → V_kernel (with V_kernel containing the photon Hodge bundle). Cannot be a pure V_Ram-internal operator.

**C3 (from L3-trace-survey):** Use a chirality projector that is gauge-canonical — either intrinsic to B(k)'s spectrum (e.g., the polar decomposition's unitary part U(B) = B/|B| on V_Ram), or fixed by external structure that commutes with U_C3 on each sector.

**C4 (from Q):** The vertex F0_γ cannot reduce to ∂_k B at a single momentum (the 1-loop self-energy at zero external momentum is parity-EVEN regardless of propagator form). Either the vertex has additional structure beyond ∂_k B, or the diagram is integrated over external momentum (a full BZ integral, where the Berry-curvature-like contribution from the walker eigenmode bundle adds parity-odd content).

### Two narrowed routes for F0_γ–F3_γ+Spectral_γ

After the four falsifications, the recipe pivots to:

**Route 1 (sharpened): Polar-decomposition-explicit vertex.** Build V_chir using B(k)'s polar decomposition U(B(k)) = B(k)/|B(k)| explicitly. The unitary U(B) commutes with C_3 (since [B, C_3] = 0), so U(B) is C_3-equivariant. On each B-eigenspace, U(B) acts as e^{i·arg(λ_n)}. The unit-phasor parity-odd projection is (U − U†)/(2i) = sin(arg(B)) on the spectrum. This bypasses constraint C3 (no γ_7 transfer needed). The remaining structural step is to identify how the photon Hodge bundle's L/R components couple to U(B) without going through V_Ram — i.e., satisfying C2 by some V_kernel ↔ V_kernel mechanism that picks up sin(arg(B)) from a virtual V_Ram excursion.

**Route 2 (sharpened): BZ-integrated 1-loop with Berry-curvature.** Integrate the 1-loop self-energy over the full BZ instead of evaluating at k = k_P only. The Berry curvature of the walker eigenmode bundle over the BZ provides a topological contribution that is parity-ODD (analogous to QED's chiral anomaly via the Adler-Bell-Jackiw triangle). This satisfies C4 by adding non-trivial momentum integration. Implementation requires Berry-phase calculation on V_Ram(k) over a non-trivial cycle of the BZ; substantial but bounded.

Both routes remain ~2–3 sessions of new structural work.

### Q' Berry-phase attempt 2026-04-28 night — partial progress, SU(2) topology found (non-Abelian)

Conceptual setup: the no-go theorem rules out k_P-local closure of c = 1. But k_P is a band-crossing point, and the walker's doubly-degenerate eigenvalue h there might be a topological monopole. A small loop in the BZ encircling k_P would pick up a Berry phase that's quantized by the monopole charge — exactly the topological winding the recipe needs.

**Setup attempted.** Small C_3-invariant triangle of radius eps around k_P in the (1,1,1)-perpendicular plane, discretized into 3M points. Wilson loop computed by parallel-transport of the +h band's 2-dim eigenspace via overlap matrices.

**Findings (`proofs/foundations/arg_h_path_b_q_prime_{berry_phase_attempt,diagnostics,su2_convergence}.py`):**

1. **Degeneracy splits linearly off k_P** (Δλ ≈ 3.24·eps). True band crossing — k_P IS a topological singularity, not just an accidental degeneracy.

2. **U(1) Berry phase γ_B → 0** as eps → 0 (γ_B ~ -4.27·eps²). NOT a charge-1 Berry monopole; the Abelian holonomy goes to zero in the small-loop limit.

3. **SU(2) (non-Abelian) Wilson holonomy IS topological**: converges to θ_∞ ≈ 269.030° as eps → 0 (clean at 6 decimal places). Not zero — there IS non-Abelian topological winding.

4. **The angle 269.03° does not match clean candidates**: not 2π − 2·arg(h) = 255.52°, not 2π·sin(arg h) = 285°, not 2·arg(h) = 104.48°. cos(θ/2) ≈ -0.7011 ≈ -(close to but not exactly) 1/√2.

**Honest assessment.** Q' confirms the framework's substrate has non-trivial topology at k_P, but the specific topological invariant (SU(2) holonomy angle 269.03°) doesn't cleanly equal any structural target. **Partial progress, not closure.**

Two interpretations remain open:

(i) The angle 269.03° is structurally meaningful but I haven't identified the right algebraic expression for it. Future analytic work might recognize it (e.g., it could relate to the I4₁32 space-group structure, not just walker eigenvalues h, h̄).

(ii) The SU(2) holonomy carries the topology, but mapping it to PHOTON POLARIZATION ROTATION requires an additional structural step (photon is spin-1; the walker's rank-2 band carries SU(2); the spin-1 ↔ SU(2) conversion is not trivial). The Higgs analog never had this issue because Higgs is scalar.

Both interpretations point to substantial follow-on work (~2-4 more sessions) beyond what was attempted here.

**Practical assessment.** Q' produced useful structural findings (non-trivial SU(2) topology at k_P) but did NOT close c = 1. The bounded path to status improvement remains **ADOPTED-ARG-H-PROJECTION**: 4 P-rows graduate BLOCKED → CONDITIONAL.

### Q' continuation 2026-04-28 night late — corrected band + analytic perturbation

After noticing that the +h band at k_P decomposes into C_3 characters (trivial + ω², NOT the (ω, ω²) I'd assumed), three follow-up scripts:

- **`arg_h_path_b_q_prime_correct_band.py`** computes Wilson holonomy on each of the 4 B-eigenvalue bands (±h, ±h̄). Result: all four bands give SU(2) angles in 263°–272° range (close to 270° = 3π/2, with band-specific variations of a few degrees). Tracking the "ω-sector" (rank-2 band spanning 1 ω-state from -h + 1 ω-state from -h̄) failed (|det W| → 0) due to band crossings between -h and -h̄ along the loop.

- **`arg_h_path_b_q_prime_c3_split.py`** decomposed the +h band into C_3 components and tracked each separately. Result: γ_ω + γ_ω² → 0 (trivial Abelian sum), γ_ω − γ_ω² → 0 (no chirality difference). The non-Abelian SU(2) topology doesn't split into 1-dim U(1) phases.

- **`arg_h_path_b_q_prime_perturbation.py`** computed first-order Bloch perturbation H_eff(δk) = P_+h · ∂_k B · δk · P_+h on the rank-2 +h band. Findings:
  - H_eff is traceless on each axis (confirms band crossing, no center shift).
  - |H_eff| = 2.774 isotropic across axes (C_3-symmetric).
  - σ-vector decomposition: |σ_Hermitian|² ≈ 0.98, |σ_anti-Hermitian|² ≈ 2.87 (≈ k* = 3).
  - **Berry monopole charge ≈ 1.2570** (winding number on a small 2-sphere). Non-integer (close to 5/4 = 1.25, but off by 0.6%). The +h band crossing is NOT a standard charge-1 Weyl monopole.

**Bottom line (NIGHT SESSION).** The walker bundle has a non-trivial topological feature at k_P with several specific structural numbers (SU(2) angle ≈ 269.03°, monopole charge ≈ 5/4, |σ_AH|² ≈ k* = 3). None matches obvious targets like 2π·sin(arg h) or 2π−2·arg(h) directly. Pinning these to a framework-derived structural identity requires multi-session theoretical work beyond bounded-session reach.

### Q' analytic follow-up 2026-04-29 — REFUTES the Berry-monopole reading

The night-session "5/4 Berry monopole charge" and "269.030° SU(2) holonomy" claims are both numerical artifacts. Subsequent analytic work:

1. **σ-norms in clean closed form**: per-axis |σ_H|² = 7π²/(27φ²) and |σ_AH|² = π²φ²/9 where φ = golden ratio. These are theorem-grade (basis-invariant traces over the rank-2 +h band).

2. **σ_H winding = ±1 (integer)**, not 5/4. The 1.2570 estimate from `arg_h_path_b_q_prime_perturbation.py` came from a discretized triple-product formula without proper Van Oosterom-Strang normalization. Refined numerics with the exact spherical-solid-angle formula give exactly −1 across all mesh refinements. Symbolic det(Λ_H) = 2π³√3(7−3√5)/243 (positive, ~0.129) confirms degree +1 sign(det Λ).

3. **Full non-Abelian first Chern = 0** on a small sphere around k_P (Wilson plaquettes on actual non-Hermitian B(k) eigenvectors, refined numerics). The +h band crossing has **NO Berry monopole**.

4. **SU(2) Wilson holonomy DOES NOT CONVERGE** as eps → 0. eps-scan at mpmath 30 dps gives wildly varying values: 207° at eps=10⁻⁵, 287° at eps=10⁻¹⁰. The "269.03° plateau" was a coincidental match at one specific (eps, M, precision) combination, not a topological invariant.

**Implications for c = 1 closure**: the Berry-curvature route is now closed off as a viable closure path. The "5/4 ↔ 5/12 Higgs ratio" speculation is REFUTED. The walker's +h band crossing has trivial Berry topology with no integer/clean-rational topological invariant that could enforce c = 1.

This is the **SEVENTH** bounded closure attempt for arg(h) Path B''. All seven give negative results. Honest assessment: c = 1 in β does NOT have a single-session-bounded structural derivation within the framework's current apparatus. ADOPTED-ARG-H-PROJECTION is the rational closure path.

### F0_γ attempt 2026-04-28 evening late — sharper structural NO-GO theorem

**Script:** `proofs/foundations/arg_h_path_b_f0_gamma_attempt.py`. Tested the gauge-canonical chirality projector γ_7^B := sign(Im(B|V_Ram)), which is intrinsic to B(P)'s spectrum (no A intertwiner needed; basis-covariant under U(8) on V_Ram, verified Step 2).

**Key result.** Computed tr_ω[γ_7^B · Im(B)/|h|] and tr_ω²[γ_7^B · Im(B)/|h|]:

```
  tr_ω [γ_7^B · Im(B)/|h|]  =  +1.8974    (= 2.4 · sin(arg h))
  tr_ω²[γ_7^B · Im(B)/|h|]  =  +1.8974    (= 2.4 · sin(arg h))
  ----------------------------------------
  Difference (ω − ω²)        =   0.00000   (machine precision)
```

The ω-ω² **chirality asymmetry vanishes** under any intrinsic-to-B chirality projector. Verified gauge-invariantly across 5 random C_3-equivariant U(8) trials on V_Ram.

**No-go theorem (informal).** *For any C_3-equivariant Hermitian operator F on V_Ram and any chirality projector γ_int intrinsic to B (commuting with C_3), the chirality asymmetry tr_ω[γ_int · F] − tr_ω²[γ_int · F] = 0.*

**Proof sketch.** γ_int and F both commute with C_3 and are functions of B's spectrum. The ω and ω² C_3 sectors of V_Ram contain B-eigenvalue conjugate pairs (-h, -h̄) and (+h, +h̄) respectively, related by the SAME intrinsic-to-B structure. So any intrinsic-to-B observable treats ω and ω² symmetrically.

**Implication.** The L3-trace-survey's apparent c = 2 result on ω-sector alone (`arg_h_path_b_l3_trace_survey.py`) was non-zero ONLY BECAUSE γ_7-via-B6 BREAKS the conjugate symmetry through the gauge choice — which is why it was gauge-dependent in the gauge-check follow-up.

**c = 1 cannot come from k_P-local spectral traces.** The required ω↔ω² asymmetry source must be EXTERNAL to k_P:

- **Cosmic time direction** (IR-level, photon-walker accumulates phase arg(h) over many cosmic time steps because time has a direction).
- **BZ momentum integral with non-trivial topology** (Berry phase of the walker eigenmode bundle over a BZ cycle).

These are exactly the Q' and A' routes named above. The k_P-local F0_γ–F3_γ recipe **cannot in principle** close c = 1 — by the no-go theorem, the lattice-level chirality observable at k_P is structurally zero.

**Sharper recipe** (post-no-go-theorem):

  F0_γ' (revised): The photon-walker dark vertex is NOT a vertex/edge-local A2 process at k_P. It is a **non-local accumulation** of per-step walker phase over either cosmic time (IR limit) or BZ topology (Berry phase). The Higgs F0 vertex-local edge process structure does not directly transfer.

  This means β closure requires a **different kind of derivation** than Higgs 5/12 — not a Feshbach loop sum at k_P, but an asymptotic/topological integration. Estimate revises upward to ~3–5 sessions for either Q' or A'.

**Practical recommendation.** Given the no-go theorem, the bounded path to status improvement is **ADOPTED-ARG-H-PROJECTION**: adopt that the photon-channel parity-odd projection of h is sin(arg h) = Im(h)/|h|, justified by Lemma 2 of `theorem_dark_correction_mdl.md` (photon polarization couples to the unit walker phasor h/|h| by dimensional matching — uniquely fixes the parity-odd part as sin(arg h)). Lemma 1 (canonical-encoding identification within the bit-cost description language) is auxiliary; the structural load is on Lemma 2 (REFRAMED 2026-05-05; the earlier "MDL bit-cost minimum" framing was a strict-minimum smuggle in violation of A2-T waterline). The structural derivation of c = 1 (Lemma 3) is deferred. 4 P-rows graduate BLOCKED → CONDITIONAL-on-adoption.

## Honest status

**β = sin(arg h)·α_EM = 0.331° remains at A−** — same status as before this work, NOT theorem-grade.

What this work added:
- Closed Step A (photon C₃-irrep structure) — useful infrastructure for any future closure.
- Tightened the open gap: previously stated as "α_EM coefficient exactly 1 (1-loop QED calculation)"; now sharper: "the magnitude of V_chir on L⊕R, derived from an A2-edge-process Feshbach chain on the photon Hodge bundle, must yield exactly ±sin(arg h)".

What this work did NOT do:
- Did not derive c = 1 to Nature-grade rigor.
- Did not construct the explicit operator V_chir whose first-order perturbation gives the canonical formula.
- Did not show that the existing β.E perturbation results (which give 0 under L3, Im(h) under L1) are reconcilable with c = 1 via the MDL framework.

---

## Path forward to Nature-grade closure

Two routes are visible:

**Route 1: explicit photon Feshbach chain.** Write down the photon-walker cross-sector coupling Lagrangian (or operator) on the srs lattice. Perform the analog of F0–F3 to count chirality channels and coupling slots. Derive c = 1 from the counting.

**Route 2: cross-check via β.E reframe.** The existing β.E perturbation results (L3 → 0, L1 → Im(h)) can be reframed as MDL bit-counting probes (Lemma 1 + Lemma 2). Show that the L3 zero result is consistent with c = 1 under the MDL synthesis (e.g., by demonstrating that V_proj is the wrong operator and the right operator gives a non-zero perturbative result). This requires constructing the right V_chir explicitly and re-running the perturbation.

Both routes require lattice-specific structural work that has not yet been done. Estimate: 1–3 sessions per route.

---

## References

- `theorem_dark_correction_mdl.md` — Lemmas 1, 2, 3 (Lemma 3 remains conditional).
- `theorem_A5b_level_prescription.md` — A5(b) Path 1 (does not specifically address β's coupling slot counting).
- `proofs/foundations/dark_feshbach_a2_closure.py` — Higgs-v 5/12 prototype (parallel structure, NOT a derivation of β's c=1).
- `proofs/cosmology/srs_photon_c3_chainmap.py` — chain-map verification (Step A, theorem-grade).
- `proofs/cosmology/srs_photon_chirality_coefficient.py` — failed attempts to construct V_chir from naive candidates.
- `proofs/lorentz/birefringence_c3_irrep_O2.py` — existing β.E perturbation tests.
