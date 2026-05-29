# Operator Compressibility Table
**Date:** 2026-04-26 (snapshot pre-T1.1, pre-LORENTZ_SIG tag-split).
**Status:** First-pass per-operator (Φ, L, Net) table over the full 195-op operator-sweep catalog. **Pre-T1.1 numbers** — refer to `simulator.md` and `audit_pilot.md` for current canonical Φ/L/Net values. Preserved as snapshot/diff reference.
**Source:** identical-state counting on F_inv(E) (A2-T); operator catalog `../operator_sweep/operator_sweep_from_A1.md` and audits `docs/operator_sweep_audit_layer_*.md`.

## Definitions

- **Φ(op)** = log₂(N_configs_before / N_classes_after) — bits of identical-state collapse the op induces on F_inv(E) configurations at reference scale (n = 10, |E| = 6).
- **L(op)** = bits to specify the op given the layer below as primitive alphabet.
- **Net = Φ − L** — A2 waterline retention criterion. Positive = compressing op (retains by A2). Zero/negative = structural permission op (enables but doesn't compress).
- **fraction compressed** = 1 − 2^(−Φ).

## Φ-template constants (this reference scale)

| template | Φ (bits) | meaning |
|---|---|---|
| `STRUCT` | 0.000 | structural — permits compression; no equivalence on configs |
| `INVOL` | 2.367 | involutive cancellation T²=id; reduces |E|^n→|E|·(|E|−1)^(n−1) |
| `CYCL` | 3.584 | cyclic-rotation collapse |
| `QUOT_ABEL` | 17.482 | abelianization F_inv → (Z/2)^|E| |
| `QUOT_K4` | 2.000 | K_4 quotient (srs ↔ free-product passage) |
| `QUOT_C3` | 1.585 | C_3 isotypic / cyclic-3 quotient |
| `QUOT_S4` | 4.585 | S_4 cubic point-group quotient |
| `PROJ_RANK2` | 2.000 | rank-2 projection on 8-dim Cl(6;ℂ) spinor |
| `PROJ_RANK1` | 3.000 | rank-1 projection on 8-dim Cl(6;ℂ) spinor |
| `BLOCH_SRS` | 3.000 | Bloch decomposition over srs primitive cell (~8 atoms) |
| `PT_QUBIT` | 1.000 | partial trace over one-qubit subfactor |
| `PT_DIRAC` | 3.000 | partial trace over 8-dim Dirac auxiliary |
| `ENTROPY_C3` | 1.585 | C_3-class Shannon entropy (uniform, log₂ 3) |
| `ENTROPY_BERN` | 1.000 | Bernoulli ±1 information measure (1 bit) |
| `COARSE_STRAUCH` | 3.322 | discrete→continuum coarse-grain (Strauch 2006) |
| `COARSE_BZ` | 3.000 | BZ → isotypic / mean-field coarse-grain |
| `HOMOL_E2` | 6.000 | cohomology classes (Z/2)^|E| ≅ |E| bits per degree |
| `THERMAL_SRS` | 6.000 | thermal Z(β) on local Hilbert dim 2^|E| |
| `K_THEORY` | 6.000 | K_0 ⊃ Z + (Z/2)^|E| |
| `MODULAR` | 23.482 | modular-form Hecke-eigenvalue restriction |
| `ATIYAH_SINGER` | 3.000 | Dirac-index integer up to ker dim |
| `TQFT` | 2.000 | symmetric monoidal category classification |
| `CLASSIFYING` | 0.000 | asphericity → π_1 determines all data |
| `RG` | 3.322 | scale-equivalence collapse over RG flow |

## Aggregate

- **Total ops:** 195
- **Positive net (above A2 waterline):** 17 / 195 = 8.7%
- **Φ = 0 (structural-only):** 133 / 195 = 68.2%
- **Invoked (any verdict):** 149 / 195 = 76.4%
- **Unused (any verdict):** 46 / 195 = 23.6%
- **Mean Φ:** 1.318 bits; **max Φ:** 23.482 bits (['A.16', 'A.17', 'A.18'])
- **Mean Net:** -1.723 bits

## Top 25 ops by Net (Φ − L)

| op | layer | name | verdict | template | Φ | L | Net | frac |
|---|---|---|---|---|---|---|---|---|
| A.16 | 7 | modular forms (spectral) | unused-def | `MODULAR` | 23.48 | 6 | **+17.48** | 100.0% |
| A.18 | 7 | Selberg zeta function | unused-def | `MODULAR` | 23.48 | 6 | **+17.48** | 100.0% |
| A.17 | 7 | automorphic L-functions | unused-def | `MODULAR` | 23.48 | 7 | **+16.48** | 100.0% |
| 1.10 | 1 | quotient F_inv(E)/N (abelianization) | inv-negatively | `QUOT_ABEL` | 17.48 | 3 | **+14.48** | 100.0% |
| 4.46 | 4 | free energy F(β) | inv-direct | `THERMAL_SRS` | 6.00 | 2 | **+4.00** | 98.4% |
| 4.47 | 4 | Boltzmann distribution | inv-direct | `THERMAL_SRS` | 6.00 | 2 | **+4.00** | 98.4% |
| 4.45 | 4 | partition function Z(β) | inv-direct | `THERMAL_SRS` | 6.00 | 3 | **+3.00** | 98.4% |
| 5.34 | 5 | quantum partition Z(β)=Tr e^{-βH} | unused-def | `THERMAL_SRS` | 6.00 | 3 | **+3.00** | 98.4% |
| 5.35 | 5 | thermal density ρ(β) | unused-def | `THERMAL_SRS` | 6.00 | 3 | **+3.00** | 98.4% |
| 6.8 | 6 | de Rham cohomology H^k_dR | unused-def | `HOMOL_E2` | 6.00 | 3 | **+3.00** | 98.4% |
| 4.34 | 4 | Peter-Weyl decomposition | inv-indirect | `QUOT_S4` | 4.58 | 3 | **+1.58** | 95.8% |
| 5.13 | 5 | pure vs mixed state | inv-direct | `PT_DIRAC` | 3.00 | 2 | **+1.00** | 87.5% |
| A.1 | 7 | group cohomology H^n(F_inv;ℤ) | unused-def | `HOMOL_E2` | 6.00 | 5 | **+1.00** | 98.4% |
| A.7 | 7 | KMS states on C*_red | unused-def | `THERMAL_SRS` | 6.00 | 5 | **+1.00** | 98.4% |
| 1.8 | 1 | conjugation c_h | unused-def | `CYCL` | 3.58 | 3 | **+0.58** | 91.7% |
| 0.4 | 0 | involutive cancellation T_e²=id | inv-direct | `INVOL` | 2.37 | 2 | **+0.37** | 80.6% |
| 4.51 | 4 | BZJ scaling v∝N^{-1/4} | inv-direct | `RG` | 3.32 | 3 | **+0.32** | 90.0% |
| 4.18 | 4 | per-Brillouin-point fibers T(k) | inv-direct | `BLOCH_SRS` | 3.00 | 3 | **+0.00** | 87.5% |
| 4.22 | 4 | quotient under equivalence | inv-direct | `COARSE_BZ` | 3.00 | 3 | **+0.00** | 87.5% |
| 4.23 | 4 | coarse-graining (lossy projection) | inv-direct | `COARSE_BZ` | 3.00 | 3 | **+0.00** | 87.5% |
| 4.50 | 4 | mean-field approximation | inv-direct | `COARSE_BZ` | 3.00 | 3 | **+0.00** | 87.5% |
| 4.53 | 4 | Curie-Weiss mean-field model | inv-direct | `COARSE_BZ` | 3.00 | 3 | **+0.00** | 87.5% |
| 5.5 | 5 | spectral decomp (real eig, complex evec) | inv-direct | `BLOCH_SRS` | 3.00 | 3 | **+0.00** | 87.5% |
| 5.12 | 5 | density matrix ρ | inv-direct | `PT_DIRAC` | 3.00 | 3 | **+0.00** | 87.5% |
| 5.14 | 5 | partial trace ρ_A=Tr_B(ρ_AB) | inv-direct | `PT_DIRAC` | 3.00 | 3 | **+0.00** | 87.5% |

## Below A2 waterline (Net ≤ 0): 178 ops

These ops are *structural permissions* — they enable downstream compression but don't compress on their own. Φ=0 across the board; Net = −L from spec cost. Listed by (layer, opid) for orientation only:

| op | layer | name | verdict | L |
|---|---|---|---|---|
| 0.1 | 0 | identity element id | inv-direct | 1 |
| 0.2 | 0 | generator T_e | inv-direct | 1 |
| 0.3 | 0 | sequential composition | inv-direct | 1 |
| 1.1 | 1 | group element g ∈ F_inv(E) | inv-direct | 2 |
| 1.11 | 1 | Cayley graph | inv-direct | 2 |
| 1.12 | 1 | word length ℓ(g) | inv-direct | 2 |
| 1.13 | 1 | Cayley-graph distance d(g,h) | inv-direct | 2 |
| 1.2 | 1 | group multiplication | inv-direct | 2 |
| 1.3 | 1 | group inverse g⁻¹ | inv-direct | 2 |
| 1.4 | 1 | group identity ε | inv-direct | 2 |
| 1.5 | 1 | powers g^n | inv-direct | 2 |
| 1.6 | 1 | left action L_h | inv-direct | 2 |
| 1.7 | 1 | right action R_h | unused-def | 2 |
| 1.9 | 1 | subgroups, cosets | inv-direct | 2 |
| 2.1 | 2 | functions f: F_inv(E) → 𝔽 | inv-direct | 2 |
| 2.10 | 2 | unitary/SA/skew classifications | inv-direct | 3 |
| 2.11 | 2 | spectral content of bounded SA | inv-direct | 3 |
| 2.13 | 2 | left regular representation L_h | inv-direct | 3 |
| 2.14 | 2 | right regular representation R_h | unused-def | 3 |
| 2.15 | 2 | adjacency operator A=Σ L_e | inv-direct | 3 |
| 2.16 | 2 | self-adjointness of A | inv-direct | 2 |
| 2.17 | 2 | spectral decomposition of A | inv-direct | 4 |
| 2.18 | 2 | Hashimoto operator (directed-edge) | inv-direct | 4 |
| 2.2 | 2 | pointwise +,·,conj | inv-direct | 2 |
| 2.20 | 2 | bounded operators ℬ(L²) | inv-direct | 2 |
| 2.21 | 2 | compact operators 𝒦(L²) | unused-neg | 3 |
| 2.22 | 2 | trace-class ℬ_1(L²) | inv-direct | 3 |
| 2.23 | 2 | Hilbert-Schmidt ℬ_2(L²) | inv-direct | 3 |
| 2.24 | 2 | self-adjoint ℬ_sa | inv-direct | 2 |
| 2.25 | 2 | closed unbounded operators | inv-direct | 3 |
| 2.26 | 2 | trace Tr(T) | inv-direct | 2 |
| 2.27 | 2 | matrix elements ⟨g|T|h⟩ | inv-direct | 2 |
| 2.28 | 2 | orthogonal projection P_S | inv-direct | 3 |
| 2.29 | 2 | HS norm Tr(T*T) | inv-direct | 2 |
| 2.3 | 2 | counting (Haar) measure | inv-direct | 2 |
| 2.31 | 2 | functional calculus p(T) | inv-direct | 3 |
| 2.33 | 2 | resolvent R_λ(T) | inv-direct | 3 |
| 2.34 | 2 | determinant det(T) | inv-direct | 2 |
| 2.35 | 2 | algebraic tensor product | inv-indirect | 3 |
| 2.36 | 2 | Hilbert tensor product | inv-direct | 3 |
| 2.37 | 2 | tensor product of operators | inv-direct | 3 |
| 2.4 | 2 | sums Σ_g f(g) | inv-direct | 2 |
| 2.5 | 2 | L²(F_inv(E);𝔽) Hilbert space | inv-direct | 3 |
| 2.6 | 2 | orthonormal basis {δ_g} | inv-direct | 3 |
| 2.7 | 2 | Hilbert-space completeness | inv-direct | 2 |
| 2.8 | 2 | bounded linear operators | inv-direct | 3 |
| 2.9 | 2 | adjoints T* | inv-direct | 2 |
| 3.1 | 3 | one-parameter unitary group U(t) | inv-direct | 3 |
| 3.10 | 3 | discrete-time quantum walk U^n | inv-direct | 2 |
| 3.11 | 3 | discrete→continuum walk limit | inv-direct | 4 |
| 3.12 | 3 | continuum-limit Hamiltonian H | inv-direct | 3 |
| 3.13 | 3 | framework's specific continuum H | inv-direct | 3 |
| 3.2 | 3 | strong continuity of U(t) | inv-direct | 2 |
| 3.3 | 3 | continuous-time quantum walks | inv-direct | 3 |
| 3.4 | 3 | Stone (complex form) | inv-direct | 3 |
| 3.5 | 3 | Stone (real form) | inv-direct | 3 |
| 3.6 | 3 | self-adjoint H on ℂ-L² | inv-direct | 2 |
| 3.7 | 3 | skew-symmetric B on ℝ-L² | inv-direct | 2 |
| 3.8 | 3 | spectrum σ(H)⊂ℝ vs σ(B)⊂iℝ | inv-direct | 3 |
| 3.9 | 3 | Cayley transform V=(H−i)(H+i)⁻¹ | unused-neg | 3 |
| 4.1 | 4 | probability measure P | inv-direct | 2 |
| 4.10 | 4 | rate-distortion bound | inv-direct | 4 |
| 4.11 | 4 | discrete-time Markov chain | inv-direct | 3 |
| 4.12 | 4 | stationary distribution | inv-direct | 3 |
| 4.13 | 4 | continuous-time Markov process | unused-neg | 3 |
| 4.14 | 4 | correlation function C_n(s) | inv-direct | 3 |
| 4.15 | 4 | decay rate / correlation length | inv-direct | 3 |
| 4.16 | 4 | isotypic decomposition | inv-direct | 3 |
| 4.17 | 4 | Bloch decomposition | inv-direct | 4 |
| 4.18 | 4 | per-Brillouin-point fibers T(k) | inv-direct | 3 |
| 4.19 | 4 | symmetry-protected degeneracies | inv-direct | 3 |
| 4.2 | 4 | expectation E_P[f] | inv-direct | 2 |
| 4.20 | 4 | Alon-Boppana / Ramanujan bound | inv-direct | 3 |
| 4.21 | 4 | group quotient F_inv(E)/N (K_4) | inv-direct | 3 |
| 4.22 | 4 | quotient under equivalence | inv-direct | 3 |
| 4.23 | 4 | coarse-graining (lossy projection) | inv-direct | 3 |
| 4.24 | 4 | partial trace over subfactor | inv-direct | 4 |
| 4.25 | 4 | conditional expectation E[·|sub-σ] | unused-deriv | 4 |
| 4.3 | 4 | joint and marginal distributions | inv-direct | 3 |
| 4.30 | 4 | group representation ρ:G→𝒰(V) | inv-direct | 3 |
| 4.31 | 4 | character χ_ρ(g) | inv-direct | 2 |
| 4.32 | 4 | representation matrix elements | inv-direct | 2 |
| 4.33 | 4 | Schur orthogonality | inv-direct | 3 |
| 4.35 | 4 | Wigner d-matrices d^j_{mm´}(θ) | inv-direct | 4 |
| 4.36 | 4 | Clebsch-Gordan decomposition | inv-direct | 3 |
| 4.37 | 4 | Clebsch-Gordan coefficients | inv-direct | 3 |
| 4.38 | 4 | trace identities under reps | inv-direct | 2 |
| 4.39 | 4 | matrix Lie group | inv-direct | 3 |
| 4.4 | 4 | conditional probability / Bayes | inv-direct | 3 |
| 4.40 | 4 | Lie algebra | inv-direct | 3 |
| 4.41 | 4 | exponential map exp(X) | inv-direct | 2 |
| 4.42 | 4 | structure constants f^c_{ab} | inv-direct | 3 |
| 4.43 | 4 | Killing form K(X,Y) | inv-direct | 3 |
| 4.44 | 4 | one-parameter subgroup t↦exp(tX) | inv-direct | 2 |
| 4.48 | 4 | order parameter / phase diagram | inv-direct | 2 |
| 4.49 | 4 | critical exponents | inv-direct | 3 |
| 4.5 | 4 | Shannon entropy | inv-direct | 3 |
| 4.50 | 4 | mean-field approximation | inv-direct | 3 |
| 4.52 | 4 | renormalization group flow | inv-direct | 4 |
| 4.53 | 4 | Curie-Weiss mean-field model | inv-direct | 3 |
| 4.6 | 4 | KL divergence | inv-direct | 3 |
| 4.7 | 4 | mutual information I(X;Y) | inv-direct | 3 |
| 4.8 | 4 | description length L(M) | inv-direct | 3 |
| 4.9 | 4 | source coding (entropy) | inv-direct | 3 |
| 5.1 | 5 | imaginary unit i in op algebra | inv-direct | 2 |
| 5.10 | 5 | ℤ/2-grading by (-1)^F | inv-direct | 2 |
| 5.11 | 5 | Majorana operators γ | inv-direct | 3 |
| 5.12 | 5 | density matrix ρ | inv-direct | 3 |
| 5.14 | 5 | partial trace ρ_A=Tr_B(ρ_AB) | inv-direct | 3 |
| 5.15 | 5 | purification of ρ_A | inv-direct | 3 |
| 5.16 | 5 | Schmidt decomposition | unused-def | 3 |
| 5.17 | 5 | quantum tensor products w/ ent. | inv-direct | 3 |
| 5.18 | 5 | complex conjugation K (anti-linear) | inv-direct | 2 |
| 5.19 | 5 | anti-unitary V | inv-direct | 2 |
| 5.2 | 5 | Pauli σ^x,σ^y,σ^z | inv-direct | 3 |
| 5.20 | 5 | time-reversal symmetry | inv-direct | 3 |
| 5.21 | 5 | Schrödinger evolution e^{-iHt} | inv-direct | 3 |
| 5.22 | 5 | Heisenberg picture | unused-def | 3 |
| 5.23 | 5 | interaction picture | unused-neg | 4 |
| 5.24 | 5 | time-dependent perturbation | unused-neg | 4 |
| 5.25 | 5 | non-real algebraic eigenvalues | inv-direct | 3 |
| 5.26 | 5 | eigenvectors w/ complex phases | inv-direct | 3 |
| 5.27 | 5 | Berry / geometric phases | inv-direct | 3 |
| 5.28 | 5 | complex Lie groups Spin/SU/U/GL | inv-direct | 3 |
| 5.29 | 5 | spin reps of Spin(n) on Cl spinors | inv-direct | 3 |
| 5.3 | 5 | Hermitian (complex) operators | inv-direct | 2 |
| 5.30 | 5 | Pati-Salam embedding in Spin(6) | inv-direct | 3 |
| 5.31 | 5 | complex characters χ_ρ ∈ ℂ | inv-direct | 2 |
| 5.32 | 5 | complex Clebsch-Gordan SU(n) | inv-direct | 3 |
| 5.33 | 5 | Wick rotation t→-iτ | inv-direct | 3 |
| 5.36 | 5 | von Neumann entropy S(ρ) | unused-def | 3 |
| 5.37 | 5 | Schmidt rank of bipartite pure | unused-def | 3 |
| 5.38 | 5 | entanglement entropy | unused-def | 3 |
| 5.4 | 5 | anti-Hermitian operators | inv-direct | 2 |
| 5.5 | 5 | spectral decomp (real eig, complex evec) | inv-direct | 3 |
| 5.6 | 5 | Jordan-Wigner construction | inv-direct | 5 |
| 5.7 | 5 | CAR {c_i,c_j†}=δ_ij | inv-direct | 3 |
| 5.8 | 5 | complex Clifford Cl(n;ℂ) | inv-direct | 3 |
| 5.9 | 5 | spinor reps of Cl(n;ℂ) | inv-direct | 3 |
| 6.1 | 6 | smooth manifold M | inv-indirect | 3 |
| 6.10 | 6 | Lorentzian metric (-,+,+,+) | inv-indirect | 3 |
| 6.11 | 6 | Levi-Civita connection ∇ | unused-def | 3 |
| 6.12 | 6 | Christoffel symbols Γ | unused-def | 3 |
| 6.13 | 6 | Riemann curvature R^a_{bcd} | unused-def | 3 |
| 6.14 | 6 | Ricci R_{ab}, scalar R | unused-def | 3 |
| 6.15 | 6 | geodesics | inv-indirect | 3 |
| 6.16 | 6 | parallel transport | inv-indirect | 3 |
| 6.17 | 6 | Killing vector fields | unused-def | 3 |
| 6.18 | 6 | FLRW metric | inv-direct | 3 |
| 6.19 | 6 | Einstein equations | inv-indirect | 4 |
| 6.2 | 6 | tangent space T_p M | unused-def | 3 |
| 6.20 | 6 | Friedmann equations | inv-direct | 3 |
| 6.21 | 6 | Hubble parameter H(t) | inv-direct | 2 |
| 6.22 | 6 | cosmological scale factor a(t) | inv-direct | 2 |
| 6.23 | 6 | stress-energy tensor T_{ab} | inv-direct | 3 |
| 6.24 | 6 | causal structure / horizons | inv-direct | 3 |
| 6.3 | 6 | tangent / cotangent bundle | unused-def | 3 |
| 6.4 | 6 | tensor fields T^(p,q)(M) | inv-direct | 3 |
| 6.5 | 6 | differential forms Ω^k(M) | inv-indirect | 3 |
| 6.6 | 6 | exterior derivative d | inv-indirect | 2 |
| 6.7 | 6 | Lie derivative ℒ_X | unused-def | 3 |
| 6.9 | 6 | Riemannian metric g | inv-direct | 3 |
| A.10 | 7 | F_inv(E) as monoidal category | unused-def | 4 |
| A.11 | 7 | ZX-calculus diagrammatic reasoning | unused-def | 5 |
| A.12 | 7 | monoidal functors | unused-def | 5 |
| A.13 | 7 | Brownian motion as continuum limit | unused-def | 5 |
| A.14 | 7 | SDEs on L² | unused-def | 5 |
| A.15 | 7 | martingales, multiway filtration | unused-def | 5 |
| A.19 | 7 | quantum gravity operations | unused-def | 6 |
| A.2 | 7 | classifying space BF_inv(E) | unused-def | 5 |
| A.20 | 7 | TQFT operations | unused-def | 6 |
| A.21 | 7 | CFT operators (OPE, Virasoro) | unused-def | 6 |
| A.3 | 7 | K-theory K_*(C*_red(F_inv)) | unused-def | 6 |
| A.4 | 7 | Atiyah-Singer / graph Dirac index | unused-def | 6 |
| A.5 | 7 | reduced group C*-algebra | unused-def | 5 |
| A.6 | 7 | group von Neumann algebra L(F_inv) | unused-def | 5 |
| A.8 | 7 | free convolution of measures | unused-def | 5 |
| A.9 | 7 | free entropy / free Fisher info | unused-def | 5 |

## Full table (grouped by layer)

### Layer 0 (4 ops)

| op | name | verdict | template | Φ | L | Net |
|---|---|---|---|---|---|---|
| 0.1 | identity element id | inv-direct | `STRUCT` | 0.00 | 1 | -1.00 |
| 0.2 | generator T_e | inv-direct | `STRUCT` | 0.00 | 1 | -1.00 |
| 0.3 | sequential composition | inv-direct | `STRUCT` | 0.00 | 1 | -1.00 |
| 0.4 | involutive cancellation T_e²=id | inv-direct | `INVOL` | 2.37 | 2 | +0.37 |

### Layer 1 (13 ops)

| op | name | verdict | template | Φ | L | Net |
|---|---|---|---|---|---|---|
| 1.1 | group element g ∈ F_inv(E) | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 1.2 | group multiplication | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 1.3 | group inverse g⁻¹ | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 1.4 | group identity ε | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 1.5 | powers g^n | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 1.6 | left action L_h | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 1.7 | right action R_h | unused-def | `STRUCT` | 0.00 | 2 | -2.00 |
| 1.8 | conjugation c_h | unused-def | `CYCL` | 3.58 | 3 | +0.58 |
| 1.9 | subgroups, cosets | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 1.10 | quotient F_inv(E)/N (abelianization) | inv-negatively | `QUOT_ABEL` | 17.48 | 3 | +14.48 |
| 1.11 | Cayley graph | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 1.12 | word length ℓ(g) | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 1.13 | Cayley-graph distance d(g,h) | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |

### Layer 2 (33 ops)

| op | name | verdict | template | Φ | L | Net |
|---|---|---|---|---|---|---|
| 2.1 | functions f: F_inv(E) → 𝔽 | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 2.2 | pointwise +,·,conj | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 2.3 | counting (Haar) measure | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 2.4 | sums Σ_g f(g) | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 2.5 | L²(F_inv(E);𝔽) Hilbert space | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 2.6 | orthonormal basis {δ_g} | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 2.7 | Hilbert-space completeness | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 2.8 | bounded linear operators | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 2.9 | adjoints T* | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 2.10 | unitary/SA/skew classifications | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 2.11 | spectral content of bounded SA | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 2.13 | left regular representation L_h | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 2.14 | right regular representation R_h | unused-def | `STRUCT` | 0.00 | 3 | -3.00 |
| 2.15 | adjacency operator A=Σ L_e | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 2.16 | self-adjointness of A | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 2.17 | spectral decomposition of A | inv-direct | `BLOCH_SRS` | 3.00 | 4 | -1.00 |
| 2.18 | Hashimoto operator (directed-edge) | inv-direct | `BLOCH_SRS` | 3.00 | 4 | -1.00 |
| 2.20 | bounded operators ℬ(L²) | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 2.21 | compact operators 𝒦(L²) | unused-neg | `STRUCT` | 0.00 | 3 | -3.00 |
| 2.22 | trace-class ℬ_1(L²) | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 2.23 | Hilbert-Schmidt ℬ_2(L²) | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 2.24 | self-adjoint ℬ_sa | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 2.25 | closed unbounded operators | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 2.26 | trace Tr(T) | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 2.27 | matrix elements ⟨g|T|h⟩ | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 2.28 | orthogonal projection P_S | inv-direct | `PROJ_RANK2` | 2.00 | 3 | -1.00 |
| 2.29 | HS norm Tr(T*T) | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 2.31 | functional calculus p(T) | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 2.33 | resolvent R_λ(T) | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 2.34 | determinant det(T) | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 2.35 | algebraic tensor product | inv-indirect | `STRUCT` | 0.00 | 3 | -3.00 |
| 2.36 | Hilbert tensor product | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 2.37 | tensor product of operators | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |

### Layer 3 (13 ops)

| op | name | verdict | template | Φ | L | Net |
|---|---|---|---|---|---|---|
| 3.1 | one-parameter unitary group U(t) | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 3.2 | strong continuity of U(t) | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 3.3 | continuous-time quantum walks | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 3.4 | Stone (complex form) | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 3.5 | Stone (real form) | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 3.6 | self-adjoint H on ℂ-L² | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 3.7 | skew-symmetric B on ℝ-L² | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 3.8 | spectrum σ(H)⊂ℝ vs σ(B)⊂iℝ | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 3.9 | Cayley transform V=(H−i)(H+i)⁻¹ | unused-neg | `STRUCT` | 0.00 | 3 | -3.00 |
| 3.10 | discrete-time quantum walk U^n | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 3.11 | discrete→continuum walk limit | inv-direct | `COARSE_STRAUCH` | 3.32 | 4 | -0.68 |
| 3.12 | continuum-limit Hamiltonian H | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 3.13 | framework's specific continuum H | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |

### Layer 4 (49 ops)

| op | name | verdict | template | Φ | L | Net |
|---|---|---|---|---|---|---|
| 4.1 | probability measure P | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 4.2 | expectation E_P[f] | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 4.3 | joint and marginal distributions | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 4.4 | conditional probability / Bayes | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 4.5 | Shannon entropy | inv-direct | `ENTROPY_BERN` | 1.00 | 3 | -2.00 |
| 4.6 | KL divergence | inv-direct | `ENTROPY_BERN` | 1.00 | 3 | -2.00 |
| 4.7 | mutual information I(X;Y) | inv-direct | `ENTROPY_BERN` | 1.00 | 3 | -2.00 |
| 4.8 | description length L(M) | inv-direct | `ENTROPY_BERN` | 1.00 | 3 | -2.00 |
| 4.9 | source coding (entropy) | inv-direct | `ENTROPY_BERN` | 1.00 | 3 | -2.00 |
| 4.10 | rate-distortion bound | inv-direct | `ENTROPY_BERN` | 1.00 | 4 | -3.00 |
| 4.11 | discrete-time Markov chain | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 4.12 | stationary distribution | inv-direct | `ENTROPY_BERN` | 1.00 | 3 | -2.00 |
| 4.13 | continuous-time Markov process | unused-neg | `STRUCT` | 0.00 | 3 | -3.00 |
| 4.14 | correlation function C_n(s) | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 4.15 | decay rate / correlation length | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 4.16 | isotypic decomposition | inv-direct | `QUOT_C3` | 1.58 | 3 | -1.42 |
| 4.17 | Bloch decomposition | inv-direct | `BLOCH_SRS` | 3.00 | 4 | -1.00 |
| 4.18 | per-Brillouin-point fibers T(k) | inv-direct | `BLOCH_SRS` | 3.00 | 3 | +0.00 |
| 4.19 | symmetry-protected degeneracies | inv-direct | `PROJ_RANK2` | 2.00 | 3 | -1.00 |
| 4.20 | Alon-Boppana / Ramanujan bound | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 4.21 | group quotient F_inv(E)/N (K_4) | inv-direct | `QUOT_K4` | 2.00 | 3 | -1.00 |
| 4.22 | quotient under equivalence | inv-direct | `COARSE_BZ` | 3.00 | 3 | +0.00 |
| 4.23 | coarse-graining (lossy projection) | inv-direct | `COARSE_BZ` | 3.00 | 3 | +0.00 |
| 4.24 | partial trace over subfactor | inv-direct | `PT_DIRAC` | 3.00 | 4 | -1.00 |
| 4.25 | conditional expectation E[·|sub-σ] | unused-deriv | `STRUCT` | 0.00 | 4 | -4.00 |
| 4.30 | group representation ρ:G→𝒰(V) | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 4.31 | character χ_ρ(g) | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 4.32 | representation matrix elements | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 4.33 | Schur orthogonality | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 4.34 | Peter-Weyl decomposition | inv-indirect | `QUOT_S4` | 4.58 | 3 | +1.58 |
| 4.35 | Wigner d-matrices d^j_{mm´}(θ) | inv-direct | `STRUCT` | 0.00 | 4 | -4.00 |
| 4.36 | Clebsch-Gordan decomposition | inv-direct | `QUOT_C3` | 1.58 | 3 | -1.42 |
| 4.37 | Clebsch-Gordan coefficients | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 4.38 | trace identities under reps | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 4.39 | matrix Lie group | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 4.40 | Lie algebra | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 4.41 | exponential map exp(X) | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 4.42 | structure constants f^c_{ab} | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 4.43 | Killing form K(X,Y) | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 4.44 | one-parameter subgroup t↦exp(tX) | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 4.45 | partition function Z(β) | inv-direct | `THERMAL_SRS` | 6.00 | 3 | +3.00 |
| 4.46 | free energy F(β) | inv-direct | `THERMAL_SRS` | 6.00 | 2 | +4.00 |
| 4.47 | Boltzmann distribution | inv-direct | `THERMAL_SRS` | 6.00 | 2 | +4.00 |
| 4.48 | order parameter / phase diagram | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 4.49 | critical exponents | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 4.50 | mean-field approximation | inv-direct | `COARSE_BZ` | 3.00 | 3 | +0.00 |
| 4.51 | BZJ scaling v∝N^{-1/4} | inv-direct | `RG` | 3.32 | 3 | +0.32 |
| 4.52 | renormalization group flow | inv-direct | `RG` | 3.32 | 4 | -0.68 |
| 4.53 | Curie-Weiss mean-field model | inv-direct | `COARSE_BZ` | 3.00 | 3 | +0.00 |

### Layer 5 (38 ops)

| op | name | verdict | template | Φ | L | Net |
|---|---|---|---|---|---|---|
| 5.1 | imaginary unit i in op algebra | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 5.2 | Pauli σ^x,σ^y,σ^z | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 5.3 | Hermitian (complex) operators | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 5.4 | anti-Hermitian operators | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 5.5 | spectral decomp (real eig, complex evec) | inv-direct | `BLOCH_SRS` | 3.00 | 3 | +0.00 |
| 5.6 | Jordan-Wigner construction | inv-direct | `STRUCT` | 0.00 | 5 | -5.00 |
| 5.7 | CAR {c_i,c_j†}=δ_ij | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 5.8 | complex Clifford Cl(n;ℂ) | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 5.9 | spinor reps of Cl(n;ℂ) | inv-direct | `PROJ_RANK2` | 2.00 | 3 | -1.00 |
| 5.10 | ℤ/2-grading by (-1)^F | inv-direct | `PT_QUBIT` | 1.00 | 2 | -1.00 |
| 5.11 | Majorana operators γ | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 5.12 | density matrix ρ | inv-direct | `PT_DIRAC` | 3.00 | 3 | +0.00 |
| 5.13 | pure vs mixed state | inv-direct | `PT_DIRAC` | 3.00 | 2 | +1.00 |
| 5.14 | partial trace ρ_A=Tr_B(ρ_AB) | inv-direct | `PT_DIRAC` | 3.00 | 3 | +0.00 |
| 5.15 | purification of ρ_A | inv-direct | `PT_DIRAC` | 3.00 | 3 | +0.00 |
| 5.16 | Schmidt decomposition | unused-def | `PT_DIRAC` | 3.00 | 3 | +0.00 |
| 5.17 | quantum tensor products w/ ent. | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 5.18 | complex conjugation K (anti-linear) | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 5.19 | anti-unitary V | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 5.20 | time-reversal symmetry | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 5.21 | Schrödinger evolution e^{-iHt} | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 5.22 | Heisenberg picture | unused-def | `STRUCT` | 0.00 | 3 | -3.00 |
| 5.23 | interaction picture | unused-neg | `STRUCT` | 0.00 | 4 | -4.00 |
| 5.24 | time-dependent perturbation | unused-neg | `STRUCT` | 0.00 | 4 | -4.00 |
| 5.25 | non-real algebraic eigenvalues | inv-direct | `BLOCH_SRS` | 3.00 | 3 | +0.00 |
| 5.26 | eigenvectors w/ complex phases | inv-direct | `BLOCH_SRS` | 3.00 | 3 | +0.00 |
| 5.27 | Berry / geometric phases | inv-direct | `BLOCH_SRS` | 3.00 | 3 | +0.00 |
| 5.28 | complex Lie groups Spin/SU/U/GL | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 5.29 | spin reps of Spin(n) on Cl spinors | inv-direct | `PROJ_RANK2` | 2.00 | 3 | -1.00 |
| 5.30 | Pati-Salam embedding in Spin(6) | inv-direct | `QUOT_K4` | 2.00 | 3 | -1.00 |
| 5.31 | complex characters χ_ρ ∈ ℂ | inv-direct | `QUOT_C3` | 1.58 | 2 | -0.42 |
| 5.32 | complex Clebsch-Gordan SU(n) | inv-direct | `QUOT_C3` | 1.58 | 3 | -1.42 |
| 5.33 | Wick rotation t→-iτ | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 5.34 | quantum partition Z(β)=Tr e^{-βH} | unused-def | `THERMAL_SRS` | 6.00 | 3 | +3.00 |
| 5.35 | thermal density ρ(β) | unused-def | `THERMAL_SRS` | 6.00 | 3 | +3.00 |
| 5.36 | von Neumann entropy S(ρ) | unused-def | `ENTROPY_BERN` | 1.00 | 3 | -2.00 |
| 5.37 | Schmidt rank of bipartite pure | unused-def | `PT_DIRAC` | 3.00 | 3 | +0.00 |
| 5.38 | entanglement entropy | unused-def | `ENTROPY_BERN` | 1.00 | 3 | -2.00 |

### Layer 6 (24 ops)

| op | name | verdict | template | Φ | L | Net |
|---|---|---|---|---|---|---|
| 6.1 | smooth manifold M | inv-indirect | `STRUCT` | 0.00 | 3 | -3.00 |
| 6.2 | tangent space T_p M | unused-def | `STRUCT` | 0.00 | 3 | -3.00 |
| 6.3 | tangent / cotangent bundle | unused-def | `STRUCT` | 0.00 | 3 | -3.00 |
| 6.4 | tensor fields T^(p,q)(M) | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 6.5 | differential forms Ω^k(M) | inv-indirect | `STRUCT` | 0.00 | 3 | -3.00 |
| 6.6 | exterior derivative d | inv-indirect | `STRUCT` | 0.00 | 2 | -2.00 |
| 6.7 | Lie derivative ℒ_X | unused-def | `STRUCT` | 0.00 | 3 | -3.00 |
| 6.8 | de Rham cohomology H^k_dR | unused-def | `HOMOL_E2` | 6.00 | 3 | +3.00 |
| 6.9 | Riemannian metric g | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 6.10 | Lorentzian metric (-,+,+,+) | inv-indirect | `STRUCT` | 0.00 | 3 | -3.00 |
| 6.11 | Levi-Civita connection ∇ | unused-def | `STRUCT` | 0.00 | 3 | -3.00 |
| 6.12 | Christoffel symbols Γ | unused-def | `STRUCT` | 0.00 | 3 | -3.00 |
| 6.13 | Riemann curvature R^a_{bcd} | unused-def | `STRUCT` | 0.00 | 3 | -3.00 |
| 6.14 | Ricci R_{ab}, scalar R | unused-def | `STRUCT` | 0.00 | 3 | -3.00 |
| 6.15 | geodesics | inv-indirect | `STRUCT` | 0.00 | 3 | -3.00 |
| 6.16 | parallel transport | inv-indirect | `STRUCT` | 0.00 | 3 | -3.00 |
| 6.17 | Killing vector fields | unused-def | `STRUCT` | 0.00 | 3 | -3.00 |
| 6.18 | FLRW metric | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 6.19 | Einstein equations | inv-indirect | `STRUCT` | 0.00 | 4 | -4.00 |
| 6.20 | Friedmann equations | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 6.21 | Hubble parameter H(t) | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 6.22 | cosmological scale factor a(t) | inv-direct | `STRUCT` | 0.00 | 2 | -2.00 |
| 6.23 | stress-energy tensor T_{ab} | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |
| 6.24 | causal structure / horizons | inv-direct | `STRUCT` | 0.00 | 3 | -3.00 |

### Appendix (21 ops)

| op | name | verdict | template | Φ | L | Net |
|---|---|---|---|---|---|---|
| A.1 | group cohomology H^n(F_inv;ℤ) | unused-def | `HOMOL_E2` | 6.00 | 5 | +1.00 |
| A.2 | classifying space BF_inv(E) | unused-def | `CLASSIFYING` | 0.00 | 5 | -5.00 |
| A.3 | K-theory K_*(C*_red(F_inv)) | unused-def | `K_THEORY` | 6.00 | 6 | +0.00 |
| A.4 | Atiyah-Singer / graph Dirac index | unused-def | `ATIYAH_SINGER` | 3.00 | 6 | -3.00 |
| A.5 | reduced group C*-algebra | unused-def | `STRUCT` | 0.00 | 5 | -5.00 |
| A.6 | group von Neumann algebra L(F_inv) | unused-def | `STRUCT` | 0.00 | 5 | -5.00 |
| A.7 | KMS states on C*_red | unused-def | `THERMAL_SRS` | 6.00 | 5 | +1.00 |
| A.8 | free convolution of measures | unused-def | `STRUCT` | 0.00 | 5 | -5.00 |
| A.9 | free entropy / free Fisher info | unused-def | `ENTROPY_BERN` | 1.00 | 5 | -4.00 |
| A.10 | F_inv(E) as monoidal category | unused-def | `STRUCT` | 0.00 | 4 | -4.00 |
| A.11 | ZX-calculus diagrammatic reasoning | unused-def | `STRUCT` | 0.00 | 5 | -5.00 |
| A.12 | monoidal functors | unused-def | `STRUCT` | 0.00 | 5 | -5.00 |
| A.13 | Brownian motion as continuum limit | unused-def | `COARSE_STRAUCH` | 3.32 | 5 | -1.68 |
| A.14 | SDEs on L² | unused-def | `STRUCT` | 0.00 | 5 | -5.00 |
| A.15 | martingales, multiway filtration | unused-def | `STRUCT` | 0.00 | 5 | -5.00 |
| A.16 | modular forms (spectral) | unused-def | `MODULAR` | 23.48 | 6 | +17.48 |
| A.17 | automorphic L-functions | unused-def | `MODULAR` | 23.48 | 7 | +16.48 |
| A.18 | Selberg zeta function | unused-def | `MODULAR` | 23.48 | 6 | +17.48 |
| A.19 | quantum gravity operations | unused-def | `STRUCT` | 0.00 | 6 | -6.00 |
| A.20 | TQFT operations | unused-def | `TQFT` | 2.00 | 6 | -4.00 |
| A.21 | CFT operators (OPE, Virasoro) | unused-def | `STRUCT` | 0.00 | 6 | -6.00 |

## Compression-active ops grouped by Φ-template

(All ops with Φ > 0; gives a partition of the substrate's identical-state collapse.)

### `MODULAR` (Φ = 23.482 bits, 3 ops)

_modular-form Hecke-eigenvalue restriction_

- **A.16** (modular forms (spectral)) — unused-def, L=6, Net=+17.48
- **A.17** (automorphic L-functions) — unused-def, L=7, Net=+16.48
- **A.18** (Selberg zeta function) — unused-def, L=6, Net=+17.48

### `QUOT_ABEL` (Φ = 17.482 bits, 1 ops)

_abelianization F_inv → (Z/2)^|E|_

- **1.10** (quotient F_inv(E)/N (abelianization)) — inv-negatively, L=3, Net=+14.48

### `THERMAL_SRS` (Φ = 6.000 bits, 6 ops)

_thermal Z(β) on local Hilbert dim 2^|E|_

- **4.45** (partition function Z(β)) — inv-direct, L=3, Net=+3.00
- **4.46** (free energy F(β)) — inv-direct, L=2, Net=+4.00
- **4.47** (Boltzmann distribution) — inv-direct, L=2, Net=+4.00
- **5.34** (quantum partition Z(β)=Tr e^{-βH}) — unused-def, L=3, Net=+3.00
- **5.35** (thermal density ρ(β)) — unused-def, L=3, Net=+3.00
- **A.7** (KMS states on C*_red) — unused-def, L=5, Net=+1.00

### `HOMOL_E2` (Φ = 6.000 bits, 2 ops)

_cohomology classes (Z/2)^|E| ≅ |E| bits per degree_

- **6.8** (de Rham cohomology H^k_dR) — unused-def, L=3, Net=+3.00
- **A.1** (group cohomology H^n(F_inv;ℤ)) — unused-def, L=5, Net=+1.00

### `K_THEORY` (Φ = 6.000 bits, 1 ops)

_K_0 ⊃ Z + (Z/2)^|E|_

- **A.3** (K-theory K_*(C*_red(F_inv))) — unused-def, L=6, Net=+0.00

### `QUOT_S4` (Φ = 4.585 bits, 1 ops)

_S_4 cubic point-group quotient_

- **4.34** (Peter-Weyl decomposition) — inv-indirect, L=3, Net=+1.58

### `CYCL` (Φ = 3.584 bits, 1 ops)

_cyclic-rotation collapse_

- **1.8** (conjugation c_h) — unused-def, L=3, Net=+0.58

### `COARSE_STRAUCH` (Φ = 3.322 bits, 2 ops)

_discrete→continuum coarse-grain (Strauch 2006)_

- **3.11** (discrete→continuum walk limit) — inv-direct, L=4, Net=-0.68
- **A.13** (Brownian motion as continuum limit) — unused-def, L=5, Net=-1.68

### `RG` (Φ = 3.322 bits, 2 ops)

_scale-equivalence collapse over RG flow_

- **4.51** (BZJ scaling v∝N^{-1/4}) — inv-direct, L=3, Net=+0.32
- **4.52** (renormalization group flow) — inv-direct, L=4, Net=-0.68

### `BLOCH_SRS` (Φ = 3.000 bits, 8 ops)

_Bloch decomposition over srs primitive cell (~8 atoms)_

- **2.17** (spectral decomposition of A) — inv-direct, L=4, Net=-1.00
- **2.18** (Hashimoto operator (directed-edge)) — inv-direct, L=4, Net=-1.00
- **4.17** (Bloch decomposition) — inv-direct, L=4, Net=-1.00
- **4.18** (per-Brillouin-point fibers T(k)) — inv-direct, L=3, Net=+0.00
- **5.5** (spectral decomp (real eig, complex evec)) — inv-direct, L=3, Net=+0.00
- **5.25** (non-real algebraic eigenvalues) — inv-direct, L=3, Net=+0.00
- **5.26** (eigenvectors w/ complex phases) — inv-direct, L=3, Net=+0.00
- **5.27** (Berry / geometric phases) — inv-direct, L=3, Net=+0.00

### `COARSE_BZ` (Φ = 3.000 bits, 4 ops)

_BZ → isotypic / mean-field coarse-grain_

- **4.22** (quotient under equivalence) — inv-direct, L=3, Net=+0.00
- **4.23** (coarse-graining (lossy projection)) — inv-direct, L=3, Net=+0.00
- **4.50** (mean-field approximation) — inv-direct, L=3, Net=+0.00
- **4.53** (Curie-Weiss mean-field model) — inv-direct, L=3, Net=+0.00

### `PT_DIRAC` (Φ = 3.000 bits, 7 ops)

_partial trace over 8-dim Dirac auxiliary_

- **4.24** (partial trace over subfactor) — inv-direct, L=4, Net=-1.00
- **5.12** (density matrix ρ) — inv-direct, L=3, Net=+0.00
- **5.13** (pure vs mixed state) — inv-direct, L=2, Net=+1.00
- **5.14** (partial trace ρ_A=Tr_B(ρ_AB)) — inv-direct, L=3, Net=+0.00
- **5.15** (purification of ρ_A) — inv-direct, L=3, Net=+0.00
- **5.16** (Schmidt decomposition) — unused-def, L=3, Net=+0.00
- **5.37** (Schmidt rank of bipartite pure) — unused-def, L=3, Net=+0.00

### `ATIYAH_SINGER` (Φ = 3.000 bits, 1 ops)

_Dirac-index integer up to ker dim_

- **A.4** (Atiyah-Singer / graph Dirac index) — unused-def, L=6, Net=-3.00

### `INVOL` (Φ = 2.367 bits, 1 ops)

_involutive cancellation T²=id; reduces |E|^n→|E|·(|E|−1)^(n−1)_

- **0.4** (involutive cancellation T_e²=id) — inv-direct, L=2, Net=+0.37

### `PROJ_RANK2` (Φ = 2.000 bits, 4 ops)

_rank-2 projection on 8-dim Cl(6;ℂ) spinor_

- **2.28** (orthogonal projection P_S) — inv-direct, L=3, Net=-1.00
- **4.19** (symmetry-protected degeneracies) — inv-direct, L=3, Net=-1.00
- **5.9** (spinor reps of Cl(n;ℂ)) — inv-direct, L=3, Net=-1.00
- **5.29** (spin reps of Spin(n) on Cl spinors) — inv-direct, L=3, Net=-1.00

### `QUOT_K4` (Φ = 2.000 bits, 2 ops)

_K_4 quotient (srs ↔ free-product passage)_

- **4.21** (group quotient F_inv(E)/N (K_4)) — inv-direct, L=3, Net=-1.00
- **5.30** (Pati-Salam embedding in Spin(6)) — inv-direct, L=3, Net=-1.00

### `TQFT` (Φ = 2.000 bits, 1 ops)

_symmetric monoidal category classification_

- **A.20** (TQFT operations) — unused-def, L=6, Net=-4.00

### `QUOT_C3` (Φ = 1.585 bits, 4 ops)

_C_3 isotypic / cyclic-3 quotient_

- **4.16** (isotypic decomposition) — inv-direct, L=3, Net=-1.42
- **4.36** (Clebsch-Gordan decomposition) — inv-direct, L=3, Net=-1.42
- **5.31** (complex characters χ_ρ ∈ ℂ) — inv-direct, L=2, Net=-0.42
- **5.32** (complex Clebsch-Gordan SU(n)) — inv-direct, L=3, Net=-1.42

### `ENTROPY_BERN` (Φ = 1.000 bits, 10 ops)

_Bernoulli ±1 information measure (1 bit)_

- **4.5** (Shannon entropy) — inv-direct, L=3, Net=-2.00
- **4.6** (KL divergence) — inv-direct, L=3, Net=-2.00
- **4.7** (mutual information I(X;Y)) — inv-direct, L=3, Net=-2.00
- **4.8** (description length L(M)) — inv-direct, L=3, Net=-2.00
- **4.9** (source coding (entropy)) — inv-direct, L=3, Net=-2.00
- **4.10** (rate-distortion bound) — inv-direct, L=4, Net=-3.00
- **4.12** (stationary distribution) — inv-direct, L=3, Net=-2.00
- **5.36** (von Neumann entropy S(ρ)) — unused-def, L=3, Net=-2.00
- **5.38** (entanglement entropy) — unused-def, L=3, Net=-2.00
- **A.9** (free entropy / free Fisher info) — unused-def, L=5, Net=-4.00

### `PT_QUBIT` (Φ = 1.000 bits, 1 ops)

_partial trace over one-qubit subfactor_

- **5.10** (ℤ/2-grading by (-1)^F) — inv-direct, L=2, Net=-1.00

## Methodology notes

- **Reference scale n = 10 (srs girth), |E| = 6** (6 undirected edges per srs primitive cell). Different reference scales give different absolute Φ values but preserve the ranking.
- **Φ is not additive across ops** in this first-pass table. Two ops sharing a Φ-template double-count the same collapse. A marginal-Φ pass (each op's compression *given the previously-applied ops*) requires a topological order on the catalog and is the natural follow-up.
- **L estimates are coarse** (1–7 bits, increasing with layer/definitional depth). The discriminator between ops above the waterline is dominated by Φ; L matters only at the waterline boundary.
- **Φ = 0 ≠ useless.** Structural ops permit compression by other ops; their value is read off downstream, not on their own row. The Net = −L column for these ops gives their *spec-cost on the substrate's catalog*, which is itself a finite budget.

## Connection to gravity / dark-sector residual

- **Gravitational-DOF candidates** = configurations surviving all ops with Φ > 0. Algebraically: cyclically-reduced commutator words in F_inv(E). Geometrically: cycle content of the substrate Cayley-graph quotient. Per-op Φ here partitions the substrate's compressible content; the *complement* (residual after applying all positive-Net ops) is where the substrate Lichnerowicz curvature R_sub lives (D²_sub = n·I + R_sub, ‖R_sub‖² = n(n−1) = 30 for srs at n=6).
- **Cosmological round-trip readings** become per-cycle-class accounting: any Friedmann-level closure requires that the residual subset feeding (e.g.) Λ_CC and the residual subset feeding (e.g.) G_F overlap correctly. This table is the inventory side; per-cycle-class residual partition is a follow-up calculation.

## Cross-references

- `../operator_sweep/operator_sweep_from_A1.md` — source catalog.
- `docs/operator_sweep_audit_layer_*.md`, `../operator_sweep/operator_sweep_audit_appendix.md` — per-op audit (verdict, citations, ontology).
- `../framework/framework_axioms.md` §3 — A2-T (selective retention / MDL waterline).
