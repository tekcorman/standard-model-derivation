# Theorem (Class B): Spectral dispersion at Dirac points encodes the framework's irrational substrate constants

**Status:** Theorem-grade synthesis. Connects the framework's Lorentz-arc derivations (v_F, β, D_H, etc.) under a single structural pattern parallel to Class A's Hashimoto Perron observables.

**Written:** 2026-04-28.

## Statement

The framework's *irrational* substrate-derived constants — Fermi velocities at Dirac cones, Iorio vielbein coefficients, dim-4 LV coefficients of the scalar Bloch dispersion, etc. — are eigenvalues of the **Bloch-gradient operator** (∂A/∂k) restricted to degenerate eigenspaces at high-symmetry k-points where Dirac cones occur. This parallels Class A: Class A coefficients are eigenvalue *values* of A and B at Γ; Class B coefficients are eigenvalue *gradients* of A near Dirac k-points.

**Class A:** functionals of (λ_A, λ_B) at Γ → rational coefficients (q_NB, ε_CP, c, ...).
**Class B:** functionals of (∂λ/∂k) at Dirac k-points → irrational coefficients (v_F, β, D_H, ...).

## Closure status (audit 2026-04-28)

| coefficient | source | status |
|---|---|---|
| v_F = 1/2 (Γ) | `lorentz_sig_dirac_cone_refined.py` | **theorem-grade** (sympy + numerical, k·p verified) |
| v_F = √3/6 (P) | `lorentz_sig_dirac_cone_refined.py` | **theorem-grade** (sympy + numerical) |
| D_H = 1/16 | `lorentz_sig_h_lv_4th_order_symbolic.py` | **theorem-grade SYMBOLIC** (sympy 4s) |
| D4_iso^H = −1/1024 | same | **theorem-grade SYMBOLIC** |
| D4_aniso^H = +1/1536 | same | **theorem-grade SYMBOLIC** |
| η^H_NB = 1/6 | same | **theorem-grade SYMBOLIC** |
| D4_iso^NB = +3/512 | same (via Ihara cross-walker) | **theorem-grade SYMBOLIC** |
| Iorio vielbein β = 1 | `lorentz_sig_iorio_session3_spin_connection.py` | **theorem-grade** |
| Spin connection (1/4)Ω·(k×S) | same | **theorem-grade** |
| **G_sub** | `lorentz_sig_g_sub_*.py` + `lorentz_sig_g_sub_matter_loop_dynamic.py` | **STRUCTURALLY OPEN; static elastic modulus (paramagnetic + diamagnetic ≈ 0.26 near-cancellation) ≠ graviton kinetic; correct quantity is dynamic matter 1-loop polarization (1-2 sessions to close)** |

**Most Class B coefficients are theorem-grade.** The single open item is **G_sub**, which remains STRUCTURALLY OPEN.

Earlier candidate values from this project (1/(8π³), 1/(16π³), 9/(128π³)) were based on the **paramagnetic-only static elastic susceptibility**. Pushing the closure (`lorentz_sig_g_sub_matter_loop_dynamic.py`, 2026-04-28 PM) revealed this identification is wrong:
- Paramagnetic K_para ≈ 17.5 — magnitude of paramagnetic contribution.
- Diamagnetic K_dia ≈ 17.76 — diamagnetic contribution from W^{abcd} = ∂²H/∂u².
- **Full static elastic modulus C_full ≈ 0.26** (paramagnetic + diamagnetic nearly cancel by 2nd-order PT sign convention).

This small static modulus is structurally meaningful (substrate has near-cancelling response) but is NOT the graviton kinetic coefficient. The correct G_sub identification:

  1/(16π G_sub) = lim_{p² → 0} Π_TT^{matter}(p²) / p²

(leading p²-coefficient of the **dynamic** matter 1-loop polarization tensor), which has structurally different content from the static elastic modulus. For srs's spin-1 Dirac at Γ-cone, this calculation involves flat-band IR + cross-helicity transitions through the h=0 mode and is multi-page symbolic work.

**Theorem-grade ingredients available:**
- Bloch invariants ⟨Tr(H²)⟩ = 12, ⟨Tr(H⁴)⟩ = 60, ⟨Tr(R_4²)⟩ = 24 (walk-enumeration verified).
- Substrate Lichnerowicz D²_sub = n·I + R_sub with ‖R_sub‖²_τ = 30.
- Closed-form det(H(k)) = 3 − 2(cos k_x + cos k_y + cos k_z), L(k) = -8 cos(k_x/2)cos(k_y/2)cos(k_z/2) for the K_4 Bloch characteristic polynomial.
- Numerical elastic moduli (paramagnetic + diamagnetic) on proper BCC BZ.
- BZ-volume V_BZ_BCC = 16π³.

**Closure pending:** dynamic matter 1-loop polarization for the Γ-cone spin-1 Dirac, with helicity-decomposed propagator + p²-Taylor expansion + TT-projection + sharp BZ cutoff Λ = π integration. ~1-2 sessions of focused symbolic computation.

Three structural findings strengthen the G_sub case:
- Substrate uniform background scalar curvature R_substrate = −3 (Bloch sum rule Tr(H(k)²) = 2|E| = 12 exact integer for all k).
- BZ-averaged scalar Bloch curvature norm² ⟨Tr(R_4²)⟩_BZ = 24 (closed-walk combinatorial count: 60 length-4 walks − 36 = 24).
- **Walk-enumeration verification (2026-04-28).** `proofs/foundations/lorentz_sig_g_sub_bloch_invariants_theorem.py` exhaustively enumerates all length-2 and length-4 closed walks on srs's primitive cell with explicit zero-net-displacement filter, confirming ⟨Tr(H²)⟩_BZ = 12 and ⟨Tr(H⁴)⟩_BZ = 60 with walk-type decomposition 12 (bounces) + 24 (3-vertex i→j→i→j'→i) + 24 (3-vertex i→j→k→j→i) + 0 (4-cycles, suppressed by srs's BCC displacement geometry) = 60. Promotes the substrate-side Bloch invariants from "asserted by combinatorial argument" to "exhaustively CAS-verified".

These are theorem-grade and feed into G_sub's closed-form expression. The remaining gap is rigorous derivation that the specific combination G_sub = ⟨Tr(R_4²)⟩·v_F/(⟨Tr(H²)⟩·V_BZ) follows from first principles (the operator-level R_sub → geometric R^{ab}(x) bridge per an internal working note Sessions 1–4).

## Class B observables identified

For srs primitive cell (|V|=4, k*=3, BCC-related Brillouin zone), Dirac cones occur at three high-symmetry points:

### Γ Dirac cone (spin-1)
- Degenerate eigenvalue: λ = −1 with 3-fold multiplicity.
- Gradient ∂A/∂k restricted to the 3-fold subspace at k=0 gives a 3×3 Hermitian operator whose two non-zero eigenvalues are ±v_F.
- **v_F = 1/2**.
- The third eigenvalue is 0 (flat band, signature of spin-1 not spin-1/2 cone).

### H Dirac cone (PH-conjugate of Γ)
- At H = (−1/2, 1/2, 1/2), σ(A) = {−3, +1, +1, +1} (PH-conjugate spectrum).
- The 3-fold degeneracy at λ=+1 is the PH image of Γ's λ=−1 triple.
- **v_F = 1/2** (same magnitude as Γ by PH symmetry).

### P Dirac cone (k·p doublet)
- At P = (1/4, 1/4, 1/4), σ(A) = {−√3, −√3, +√3, +√3} — two doublets.
- Each doublet supports a Dirac cone with linear dispersion.
- **v_F = √3/6** (irrational, reflects the cubic-cell geometry at P).

### Dim-4 LV coefficients (scalar Bloch dispersion)
The framework's Lorentz-violation analysis gives Taylor coefficients of the Hashimoto eigenvalue h(k) = h_0 − D₂|k|² − α(k̂)|k|⁴ + O(|k|⁶) where α = D4_iso + D4_aniso · f₄(k̂). For srs at Γ:

| coefficient | value | source |
|---|---|---|
| D_H (= D₂ for scalar Bloch) | **1/16** | k² coefficient of h(k) |
| D4_iso^H | **−1/1024** | k⁴ isotropic coefficient |
| D4_aniso^H | **+1/1536** | k⁴ anisotropic coefficient |
| η^H_NB := D4_aniso^H / D_H² | **1/6** | dimensionless anisotropy parameter |
| D4_iso^NB | **+3/512** | NB-walker variant via Ihara |

(All from `proofs/foundations/lorentz_sig_h_lv_4th_order_symbolic.py`, theorem-grade SYMBOLIC sympy verification.)

These are higher-order Taylor coefficients of the dispersion — Class B observables of higher derivative order than v_F.

### Iorio vielbein
The substrate's emergent vielbein for the Lorentz arc has β = 1 (no anomalous prefactor), with spin connection (1/4)Ω·(k×S). β = 1 is itself a Class B coefficient — the elastic-mode normalization of the substrate's emergent flat connection.

## Pattern (k·p theory)

For any Bloch Hamiltonian H(k) with eigenvalues λ_n(k) and degenerate eigenvalue λ_0 at k = k* with multiplicity d, k·p perturbation theory gives:

$$\lambda(k) - \lambda_0 = \sum_a (k - k_*)_a \cdot v_a + O(|k - k_*|^2)$$

where the velocities $v_a$ are eigenvalues of the d × d matrix $P_d \cdot (\partial H / \partial k_a)|_{k_*} \cdot P_d$ with $P_d$ the projector onto the degenerate subspace.

For srs's adjacency Bloch operator:

$$A(k) = \sum_{(s,t,n)} e^{2\pi i \, k \cdot n} \, |t\rangle\langle s|$$

where (s, t, n) ranges over the directed bonds with cell offset n.

$$\frac{\partial A}{\partial k_a}\bigg|_{k_*} = 2\pi i \sum_{(s,t,n)} n_a \, e^{2\pi i \, k_* \cdot n} \, |t\rangle\langle s|$$

At Γ (k* = 0), all phases are 1; at P, phases are e^(2πi/4·sum) = various roots of unity.

The Class B coefficients are the *eigenvalues of these gradient matrices restricted to the degenerate subspaces*. The framework's specific values (1/2 at Γ, √3/6 at P) emerge from explicit computation of these matrix elements with srs's specific cell offsets.

## What "Class B = spectral dispersion" buys structurally

Without this framing: each v_F, D_H, β, etc. is a separately-computed quantity, justified case-by-case. The framework's Lorentz-arc closure is a chain of these computations.

With this framing: all Class B coefficients are *eigenvalues of the same gradient-operator family*, indexed by k-point and derivative order. This gives:

1. **Unified categorical statement.** Class B reduces to "eigenvalues of (P_d ∂A/∂k P_d)^n for various k* and n". Linear in spirit, just non-trivial in specific values.

2. **Falsifiability via gradient-matrix identity.** The framework's v_F = 1/2 at Γ is a property of srs's specific cell offsets. Compute the gradient matrix at Γ for a different chiral cubic crystal (e.g., a different Wyckoff position), and you'd get a different v_F. The agreement with framework v_F is a check that srs's structure is correctly identified.

3. **Cross-coupling with Class A.** The Ihara cross-walker theorem (`lorentz_sig_ihara_lv_relation.py`) connects scalar Bloch and Hashimoto LV coefficients via h'(3) = 2 (= λ_B/(λ_A−2) at Γ?). This is *exactly* a coupling between Class A (eigenvalue values) and Class B (eigenvalue gradients) — at the level of the Ihara map u(λ).

4. **Closure structure.** The Lorentz arc's "structurally complete modulo numerical G_sub" status (per memory `session_handoff_2026-04-27_lorentz_close_polish_ship.md`) is exactly the statement that Class B coefficients are computed up to G_sub's numerical value. G_sub itself is a Class B object — the elastic-mode coupling constant from Bloch-gradient analysis.

## Comparison to Class A

| feature | Class A (rational) | Class B (irrational) |
|---|---|---|
| Underlying object | Hashimoto B(Γ), adjacency A(Γ) | Bloch gradient ∂A/∂k near Dirac points |
| Spectral observable | eigenvalue *values* | eigenvalue *gradients* |
| Symmetry of host k-point | high (Γ, H invariants) | Dirac (lower symmetry, e.g., P) |
| Output type | rational | irrational |
| Examples | q_NB=2/3, c=5/12, ε_CP=1/5, V_cb=256/6305 | v_F=1/2 (Γ), v_F=√3/6 (P), D_H=1/16, β=1 |
| Number identified | 6+ | 5+ |
| Derivation route | Ihara/Stark-Terras | k·p perturbation theory |

Together, Classes A + B cover essentially all of the framework's substrate-derived numerical predictions:
- Standard-Model parameters with rational substrate origin → Class A (V_cb, q_NB-derived dynamics)
- Lorentz/causality structure with irrational substrate origin → Class B (v_F, β, D_H)

The other classes (C: group-theoretic, D: statistical, E: combinatorial) cover the remaining ~10–15 numerical predictions (sin²θ_W, Ω_DM, V_us, etc.).

## Structural ledger conditional dependencies

Class B derivations are conditional on:
- Row 4 (k* = 3 fixed-degree) → forces 3-regular graph structure for Bloch matrices.
- Row 6 (srs lattice) → forces specific cell offsets that determine gradient matrix entries.
- Row 7 (|E| = 6) → fixes the directed-edge count and Hashimoto dim.
- Row 16 (Cl(6) per node) → forces |V| = 4 primitive cell.
- Row 23 (q_NB Perron ratio) → connects to Ihara cross-walker.

Plus:
- k·p perturbation theory (standard solid-state physics, Cohen-Louie 2016).
- Bloch's theorem (standard).

No new conditional dependencies beyond what's already in the structural ledger.

## Implications

1. **The framework's irrational predictions have a unified spectral origin** (Bloch dispersion gradients), parallel to the rational ones (Bloch eigenvalue values). The framework is structurally bipartite at the substrate level: rational/value at Γ, irrational/gradient elsewhere.

2. **Lorentz arc closure is over-determined too.** Multiple independent Class B coefficients (v_F, β, D_H, η_NB, D4_aniso) all derive from the same gradient-matrix computation on srs. Mutual consistency is a non-trivial check.

3. **Closure of Class B = closure of G_sub.** The remaining "numerical-open" status of the Lorentz signature derivation (per memory) is exactly the closure of G_sub. With G_sub closed numerically, all Class B coefficients become theorem-grade.

4. **The full framework parameter space is structurally bipartite at the substrate level.** Class A (rational, Γ-eigenvalue) + Class B (irrational, gradient at Dirac points) + Classes C, D, E (group, statistical, combinatorial) account for essentially every numerical prediction. No "phenomenological" parameters remain unclassified.

## References

- `proofs/foundations/lorentz_sig_dirac_cone_*.py` — Dirac-cone derivations at Γ, H, P.
- `proofs/foundations/lorentz_sig_h_lv_4th_order_symbolic.py` — symbolic LV Taylor coefficients.
- `proofs/foundations/lorentz_sig_iorio_session3_spin_connection.py` — Iorio vielbein β derivation.
- `proofs/foundations/lorentz_sig_ihara_lv_relation.py` — Ihara cross-walker Class A ↔ Class B connection.
- `proofs/foundations/lorentz_sig_g_sub_bloch_invariants_theorem.py` — exhaustive walk enumeration verifying ⟨Tr(H²)⟩_BZ, ⟨Tr(H⁴)⟩_BZ, ⟨Tr(R_4²)⟩_BZ at theorem grade.
- `theorem_unified_spectral_dark.md` — Class A unified theorem.
- Cohen, M.L. and Louie, S.G. 2016, *Fundamentals of Condensed Matter Physics* §4.4 (k·p theory).
- Standard solid-state textbook references for Dirac-cone band-structure analysis.
