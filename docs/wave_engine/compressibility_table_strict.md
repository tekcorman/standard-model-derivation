# Operator Compressibility Table — Strict-Assumption Edition
**Date:** 2026-04-26 (snapshot pre-T1.1, pre-LORENTZ_SIG tag-split).
**Status:** Strict per-operator (Φ, L, Net) table with explicit assumption tags. Snapshot of catalog state on 2026-04-26 evening. **Subsequent corrections not reflected here:** (a) T1.1 template-dedupe (2026-04-27) removed 95 bits of Φ overcounting; (b) LORENTZ_SIG / CCLOSE tag-split (2026-04-27) split the 22-op blocked set into 15 CCLOSE-only + 1 LORENTZ_SIG-only + 6 both. Refer to `simulator.md`, `closure_experiment.md`, and `audit_pilot.md` for current canonical numbers. This table is preserved as a diff reference; absolute Φ totals are pre-T1.1 (overcount).
**Source:** identical-state counting on F_inv(E) (A2-T); operator catalog `../operator_sweep/operator_sweep_from_A1.md` and audits `docs/operator_sweep_audit_layer_*.md`.

## Why this exists

The first-pass table assigned a single Φ per op without flagging that many of those Φ values silently bake in framework-specific structure (srs lattice, k=3 trivalence, |E|=6, §F field-selection, §C smooth closure). This edition splits each op's Φ into:

- **Φ_lean** — Φ given ONLY {A1, E_FIN}. The substrate's *intrinsic* compression with no additional structure assumed.
- **Φ_strict** — Φ given the op's full assumption stack (the value from the first-pass table, now with the stack explicit).

Net is computed both ways. An op with positive Net_strict but Φ_lean = 0 is a *framework-specialization-dependent* compression — the substrate alone doesn't compress that way; only after specializing to srs/k=3/|E|=6/etc. does the bit-collapse appear.

## Assumption-tag taxonomy

| tag | always-present? | meaning |
|---|---|---|
| `A1` | ✓ | free product of involutions F_inv(E) (structural axiom) |
| `E_FIN` | ✓ | |E| finite |
| `ORDER` |  | total order on E (used by JW chain) |
| `E6` |  | |E| = 6 (six undirected edges per srs primitive cell) |
| `K3` |  | k = 3 trivalence at every Cayley-graph node |
| `SRS` |  | srs lattice (3D periodic K_4 quotient of F_inv(E)) |
| `K4Q` |  | K_4 normal-subgroup quotient |
| `CRYSTAL` |  | translation-invariance (allows Bloch decomposition) |
| `C3` |  | C_3 cyclic-3 substrate symmetry on primitive cell |
| `S4` |  | S_4 cubic point-group symmetry |
| `A2W` |  | A2-T selective-retention / MDL waterline |
| `A4` |  | A4 fermion anti-commutation (CAR via JW) |
| `A5M` |  | A5-mass downstream observable labeling |
| `P1` |  | P1' register-storability |
| `FF` |  | field selection §F (forces ℂ post-P1) |
| `STRAUCH` |  | discrete→continuum walk limit (Strauch 2006 + rapid decay) |
| `CCLOSE` |  | §C smooth-manifold closure (PARTIAL; not closed) |
| `COMPACT` |  | compact group/space (Peter-Weyl, Killing form) |
| `FIN_DIM` |  | finite-dimensional Hilbert sector / finite primitive cell |
| `LIE` |  | matrix-Lie-group structure |
| `THERM` |  | time-evolution + Hamiltonian + thermal Z(β) |
| `BZJ` |  | BZJ critical-scaling regime |
| `RGFL` |  | RG flow structure |
| `N_HUB` |  | cosmology-anchored N_hub |
| `C_REP` |  | complex-rep theory (post-§F) |

## Tag frequency across the 195-op catalog

| tag | # ops requiring it | % |
|---|---|---|
| `FF` | 44 | 22.6% |
| `SRS` | 30 | 15.4% |
| `LIE` | 27 | 13.8% |
| `A2W` | 27 | 13.8% |
| `E6` | 24 | 12.3% |
| `CCLOSE` | 22 | 11.3% |
| `K3` | 21 | 10.8% |
| `FIN_DIM` | 20 | 10.3% |
| `CRYSTAL` | 13 | 6.7% |
| `COMPACT` | 11 | 5.6% |
| `C_REP` | 11 | 5.6% |
| `THERM` | 11 | 5.6% |
| `STRAUCH` | 9 | 4.6% |
| `BZJ` | 6 | 3.1% |
| `ORDER` | 6 | 3.1% |
| `A4` | 6 | 3.1% |
| `C3` | 4 | 2.1% |
| `N_HUB` | 4 | 2.1% |
| `S4` | 3 | 1.5% |
| `RGFL` | 3 | 1.5% |
| `P1` | 2 | 1.0% |
| `K4Q` | 1 | 0.5% |

## Aggregate (strict accounting)

- **Total ops:** 195
- **Ops needing only {A1, E_FIN}** (no extras): 50 / 195 = 25.6%
- **Ops needing srs lattice (`SRS`):** 30 / 195 = 15.4%
- **Ops needing |E|=6 (`E6`):** 24 / 195 = 12.3%
- **Ops needing k=3 trivalence (`K3`):** 21 / 195 = 10.8%
- **Ops needing §C smooth closure (`CCLOSE`, PARTIAL):** 22 / 195 = 11.3%
- **Ops needing §F field selection (`FF`):** 44 / 195 = 22.6%
- **Ops needing A2-T waterline (`A2W`):** 27 / 195 = 13.8%

- **Net_strict > 0 (above waterline given full stack):** 17 / 195 = 8.7%
- **Net_lean > 0 (above waterline with A1+E_FIN ONLY):** 2 / 195 = 1.0%

**Headline:** of 17 positive-Net ops in the strict reading, **2 retain positive Net under {A1, E_FIN} alone**. The other 15 are framework-specialization-dependent compressions — they require srs / k=3 / |E|=6 / §F / etc. to deliver their bit collapse.

## Top 25 ops by Net_strict

| op | layer | name | template | Φ_strict | Φ_lean | L | Net_strict | extras |
|---|---|---|---|---|---|---|---|---|
| A.16 | 7 | modular forms (spectral) | `MODULAR` | 23.48 | 0.00 | 6 | **+17.48** | `CRYSTAL`, `E6`, `K3`, `SRS` |
| A.18 | 7 | Selberg zeta function | `MODULAR` | 23.48 | 0.00 | 6 | **+17.48** | `CRYSTAL`, `E6`, `K3`, `SRS` |
| A.17 | 7 | automorphic L-functions | `MODULAR` | 23.48 | 0.00 | 7 | **+16.48** | `CRYSTAL`, `C_REP`, `E6`, `K3`, `SRS` |
| 1.10 | 1 | quotient F_inv(E)/N (abelianization) | `QUOT_ABEL` | 17.48 | 0.00 | 3 | **+14.48** | `E6` |
| 4.46 | 4 | free energy F(β) | `THERMAL_SRS` | 6.00 | 0.00 | 2 | **+4.00** | `E6`, `FIN_DIM`, `K3`, `THERM` |
| 4.47 | 4 | Boltzmann distribution | `THERMAL_SRS` | 6.00 | 0.00 | 2 | **+4.00** | `E6`, `FIN_DIM`, `K3`, `THERM` |
| 4.45 | 4 | partition function Z(β) | `THERMAL_SRS` | 6.00 | 0.00 | 3 | **+3.00** | `E6`, `FIN_DIM`, `K3`, `THERM` |
| 5.34 | 5 | quantum partition Z(β)=Tr e^{-βH} | `THERMAL_SRS` | 6.00 | 0.00 | 3 | **+3.00** | `E6`, `FF`, `FIN_DIM`, `K3`, `THERM` |
| 5.35 | 5 | thermal density ρ(β) | `THERMAL_SRS` | 6.00 | 0.00 | 3 | **+3.00** | `E6`, `FF`, `FIN_DIM`, `K3`, `THERM` |
| 6.8 | 6 | de Rham cohomology H^k_dR | `HOMOL_E2` | 6.00 | 0.00 | 3 | **+3.00** | `CCLOSE`, `E6` |
| 4.34 | 4 | Peter-Weyl decomposition | `QUOT_S4` | 4.58 | 0.00 | 3 | **+1.58** | `COMPACT`, `S4` |
| 5.13 | 5 | pure vs mixed state | `PT_DIRAC` | 3.00 | 0.00 | 2 | **+1.00** | `A2W`, `FF` |
| A.1 | 7 | group cohomology H^n(F_inv;ℤ) | `HOMOL_E2` | 6.00 | 0.00 | 5 | **+1.00** | `E6` |
| A.7 | 7 | KMS states on C*_red | `THERMAL_SRS` | 6.00 | 0.00 | 5 | **+1.00** | `E6`, `K3`, `STRAUCH`, `THERM` |
| 1.8 | 1 | conjugation c_h | `CYCL` | 3.58 | 3.58 | 3 | **+0.58** | — |
| 0.4 | 0 | involutive cancellation T_e²=id | `INVOL` | 2.37 | 2.37 | 2 | **+0.37** | — |
| 4.51 | 4 | BZJ scaling v∝N^{-1/4} | `RG` | 3.32 | 0.00 | 3 | **+0.32** | `BZJ`, `RGFL`, `SRS` |
| 4.18 | 4 | per-Brillouin-point fibers T(k) | `BLOCH_SRS` | 3.00 | 0.00 | 3 | **+0.00** | `CRYSTAL`, `E6`, `K3`, `SRS` |
| 4.22 | 4 | quotient under equivalence | `COARSE_BZ` | 3.00 | 0.00 | 3 | **+0.00** | `CRYSTAL`, `SRS` |
| 4.23 | 4 | coarse-graining (lossy projection) | `COARSE_BZ` | 3.00 | 0.00 | 3 | **+0.00** | `A2W`, `SRS` |
| 4.50 | 4 | mean-field approximation | `COARSE_BZ` | 3.00 | 0.00 | 3 | **+0.00** | `BZJ`, `SRS` |
| 4.53 | 4 | Curie-Weiss mean-field model | `COARSE_BZ` | 3.00 | 0.00 | 3 | **+0.00** | `BZJ`, `SRS` |
| 5.5 | 5 | spectral decomp (real eig, complex evec) | `BLOCH_SRS` | 3.00 | 0.00 | 3 | **+0.00** | `CRYSTAL`, `E6`, `FF`, `K3`, `SRS` |
| 5.12 | 5 | density matrix ρ | `PT_DIRAC` | 3.00 | 0.00 | 3 | **+0.00** | `A2W`, `FF`, `FIN_DIM` |
| 5.14 | 5 | partial trace ρ_A=Tr_B(ρ_AB) | `PT_DIRAC` | 3.00 | 0.00 | 3 | **+0.00** | `A2W`, `FF`, `FIN_DIM` |

## Ops with positive Net_lean (substrate-intrinsic compression only)

These are the only ops whose compression survives stripping all framework specializations beyond {A1, E_FIN}. Everything else's Φ depends on framework-specific structure.

| op | layer | name | template | Φ_lean | L | Net_lean |
|---|---|---|---|---|---|---|
| 1.8 | 1 | conjugation c_h | `CYCL` | 3.58 | 3 | **+0.58** |
| 0.4 | 0 | involutive cancellation T_e²=id | `INVOL` | 2.37 | 2 | **+0.37** |

## Ops grouped by load-bearing extra assumption

(Each op is listed under its most restrictive extra tag; within group, sorted by Net_strict.)

### Load-bearing on `CCLOSE` — §C smooth-manifold closure (PARTIAL; not closed) (22 ops)

| op | name | verdict | template | Φ_strict | Net_strict |
|---|---|---|---|---|---|
| 6.8 | de Rham cohomology H^k_dR | unused-def | `HOMOL_E2` | 6.00 | +3.00 |
| 6.21 | Hubble parameter H(t) | inv-direct | `STRUCT` | 0.00 | -2.00 |
| 6.22 | cosmological scale factor a(t) | inv-direct | `STRUCT` | 0.00 | -2.00 |
| 6.1 | smooth manifold M | inv-indirect | `STRUCT` | 0.00 | -3.00 |
| 6.2 | tangent space T_p M | unused-def | `STRUCT` | 0.00 | -3.00 |
| 6.3 | tangent / cotangent bundle | unused-def | `STRUCT` | 0.00 | -3.00 |
| 6.4 | tensor fields T^(p,q)(M) | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 6.5 | differential forms Ω^k(M) | inv-indirect | `STRUCT` | 0.00 | -3.00 |
| 6.7 | Lie derivative ℒ_X | unused-def | `STRUCT` | 0.00 | -3.00 |
| 6.9 | Riemannian metric g | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 6.10 | Lorentzian metric (-,+,+,+) | inv-indirect | `STRUCT` | 0.00 | -3.00 |
| 6.11 | Levi-Civita connection ∇ | unused-def | `STRUCT` | 0.00 | -3.00 |
| 6.12 | Christoffel symbols Γ | unused-def | `STRUCT` | 0.00 | -3.00 |
| 6.13 | Riemann curvature R^a_{bcd} | unused-def | `STRUCT` | 0.00 | -3.00 |
| 6.14 | Ricci R_{ab}, scalar R | unused-def | `STRUCT` | 0.00 | -3.00 |
| 6.17 | Killing vector fields | unused-def | `STRUCT` | 0.00 | -3.00 |
| 6.18 | FLRW metric | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 6.20 | Friedmann equations | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 6.23 | stress-energy tensor T_{ab} | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 6.19 | Einstein equations | inv-indirect | `STRUCT` | 0.00 | -4.00 |
| A.19 | quantum gravity operations | unused-def | `STRUCT` | 0.00 | -6.00 |
| A.21 | CFT operators (OPE, Virasoro) | unused-def | `STRUCT` | 0.00 | -6.00 |

### Load-bearing on `SRS` — srs lattice (3D periodic K_4 quotient of F_inv(E)) (26 ops)

| op | name | verdict | template | Φ_strict | Net_strict |
|---|---|---|---|---|---|
| A.16 | modular forms (spectral) | unused-def | `MODULAR` | 23.48 | +17.48 |
| A.18 | Selberg zeta function | unused-def | `MODULAR` | 23.48 | +17.48 |
| A.17 | automorphic L-functions | unused-def | `MODULAR` | 23.48 | +16.48 |
| 4.51 | BZJ scaling v∝N^{-1/4} | inv-direct | `RG` | 3.32 | +0.32 |
| 4.18 | per-Brillouin-point fibers T(k) | inv-direct | `BLOCH_SRS` | 3.00 | +0.00 |
| 4.22 | quotient under equivalence | inv-direct | `COARSE_BZ` | 3.00 | +0.00 |
| 4.23 | coarse-graining (lossy projection) | inv-direct | `COARSE_BZ` | 3.00 | +0.00 |
| 4.50 | mean-field approximation | inv-direct | `COARSE_BZ` | 3.00 | +0.00 |
| 4.53 | Curie-Weiss mean-field model | inv-direct | `COARSE_BZ` | 3.00 | +0.00 |
| 5.5 | spectral decomp (real eig, complex evec) | inv-direct | `BLOCH_SRS` | 3.00 | +0.00 |
| 5.25 | non-real algebraic eigenvalues | inv-direct | `BLOCH_SRS` | 3.00 | +0.00 |
| 5.26 | eigenvectors w/ complex phases | inv-direct | `BLOCH_SRS` | 3.00 | +0.00 |
| 5.27 | Berry / geometric phases | inv-direct | `BLOCH_SRS` | 3.00 | +0.00 |
| 2.17 | spectral decomposition of A | inv-direct | `BLOCH_SRS` | 3.00 | -1.00 |
| 2.18 | Hashimoto operator (directed-edge) | inv-direct | `BLOCH_SRS` | 3.00 | -1.00 |
| 4.17 | Bloch decomposition | inv-direct | `BLOCH_SRS` | 3.00 | -1.00 |
| 4.19 | symmetry-protected degeneracies | inv-direct | `PROJ_RANK2` | 2.00 | -1.00 |
| 4.21 | group quotient F_inv(E)/N (K_4) | inv-direct | `QUOT_K4` | 2.00 | -1.00 |
| 5.30 | Pati-Salam embedding in Spin(6) | inv-direct | `QUOT_K4` | 2.00 | -1.00 |
| 6.6 | exterior derivative d | inv-indirect | `STRUCT` | 0.00 | -2.00 |
| 3.13 | framework's specific continuum H | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 4.15 | decay rate / correlation length | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 5.20 | time-reversal symmetry | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 6.15 | geodesics | inv-indirect | `STRUCT` | 0.00 | -3.00 |
| 6.16 | parallel transport | inv-indirect | `STRUCT` | 0.00 | -3.00 |
| A.4 | Atiyah-Singer / graph Dirac index | unused-def | `ATIYAH_SINGER` | 3.00 | -3.00 |

### Load-bearing on `E6` — |E| = 6 (six undirected edges per srs primitive cell) (12 ops)

| op | name | verdict | template | Φ_strict | Net_strict |
|---|---|---|---|---|---|
| 1.10 | quotient F_inv(E)/N (abelianization) | inv-negatively | `QUOT_ABEL` | 17.48 | +14.48 |
| 4.46 | free energy F(β) | inv-direct | `THERMAL_SRS` | 6.00 | +4.00 |
| 4.47 | Boltzmann distribution | inv-direct | `THERMAL_SRS` | 6.00 | +4.00 |
| 4.45 | partition function Z(β) | inv-direct | `THERMAL_SRS` | 6.00 | +3.00 |
| 5.34 | quantum partition Z(β)=Tr e^{-βH} | unused-def | `THERMAL_SRS` | 6.00 | +3.00 |
| 5.35 | thermal density ρ(β) | unused-def | `THERMAL_SRS` | 6.00 | +3.00 |
| A.1 | group cohomology H^n(F_inv;ℤ) | unused-def | `HOMOL_E2` | 6.00 | +1.00 |
| A.7 | KMS states on C*_red | unused-def | `THERMAL_SRS` | 6.00 | +1.00 |
| A.3 | K-theory K_*(C*_red(F_inv)) | unused-def | `K_THEORY` | 6.00 | +0.00 |
| 5.9 | spinor reps of Cl(n;ℂ) | inv-direct | `PROJ_RANK2` | 2.00 | -1.00 |
| 5.29 | spin reps of Spin(n) on Cl spinors | inv-direct | `PROJ_RANK2` | 2.00 | -1.00 |
| 5.8 | complex Clifford Cl(n;ℂ) | inv-direct | `STRUCT` | 0.00 | -3.00 |

### Load-bearing on `K3` — k = 3 trivalence at every Cayley-graph node (1 ops)

| op | name | verdict | template | Φ_strict | Net_strict |
|---|---|---|---|---|---|
| 4.20 | Alon-Boppana / Ramanujan bound | inv-direct | `STRUCT` | 0.00 | -3.00 |

### Load-bearing on `C3` — C_3 cyclic-3 substrate symmetry on primitive cell (3 ops)

| op | name | verdict | template | Φ_strict | Net_strict |
|---|---|---|---|---|---|
| 5.31 | complex characters χ_ρ ∈ ℂ | inv-direct | `QUOT_C3` | 1.58 | -0.42 |
| 4.16 | isotypic decomposition | inv-direct | `QUOT_C3` | 1.58 | -1.42 |
| 4.36 | Clebsch-Gordan decomposition | inv-direct | `QUOT_C3` | 1.58 | -1.42 |

### Load-bearing on `S4` — S_4 cubic point-group symmetry (1 ops)

| op | name | verdict | template | Φ_strict | Net_strict |
|---|---|---|---|---|---|
| 4.34 | Peter-Weyl decomposition | inv-indirect | `QUOT_S4` | 4.58 | +1.58 |

### Load-bearing on `FF` — field selection §F (forces ℂ post-P1) (30 ops)

| op | name | verdict | template | Φ_strict | Net_strict |
|---|---|---|---|---|---|
| 5.13 | pure vs mixed state | inv-direct | `PT_DIRAC` | 3.00 | +1.00 |
| 5.12 | density matrix ρ | inv-direct | `PT_DIRAC` | 3.00 | +0.00 |
| 5.14 | partial trace ρ_A=Tr_B(ρ_AB) | inv-direct | `PT_DIRAC` | 3.00 | +0.00 |
| 5.15 | purification of ρ_A | inv-direct | `PT_DIRAC` | 3.00 | +0.00 |
| 5.16 | Schmidt decomposition | unused-def | `PT_DIRAC` | 3.00 | +0.00 |
| 5.37 | Schmidt rank of bipartite pure | unused-def | `PT_DIRAC` | 3.00 | +0.00 |
| 5.32 | complex Clebsch-Gordan SU(n) | inv-direct | `QUOT_C3` | 1.58 | -1.42 |
| 3.6 | self-adjoint H on ℂ-L² | inv-direct | `STRUCT` | 0.00 | -2.00 |
| 5.1 | imaginary unit i in op algebra | inv-direct | `STRUCT` | 0.00 | -2.00 |
| 5.3 | Hermitian (complex) operators | inv-direct | `STRUCT` | 0.00 | -2.00 |
| 5.4 | anti-Hermitian operators | inv-direct | `STRUCT` | 0.00 | -2.00 |
| 5.18 | complex conjugation K (anti-linear) | inv-direct | `STRUCT` | 0.00 | -2.00 |
| 5.19 | anti-unitary V | inv-direct | `STRUCT` | 0.00 | -2.00 |
| 5.36 | von Neumann entropy S(ρ) | unused-def | `ENTROPY_BERN` | 1.00 | -2.00 |
| 5.38 | entanglement entropy | unused-def | `ENTROPY_BERN` | 1.00 | -2.00 |
| 3.4 | Stone (complex form) | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 3.8 | spectrum σ(H)⊂ℝ vs σ(B)⊂iℝ | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 3.9 | Cayley transform V=(H−i)(H+i)⁻¹ | unused-neg | `STRUCT` | 0.00 | -3.00 |
| 3.12 | continuum-limit Hamiltonian H | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 5.2 | Pauli σ^x,σ^y,σ^z | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 5.11 | Majorana operators γ | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 5.17 | quantum tensor products w/ ent. | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 5.21 | Schrödinger evolution e^{-iHt} | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 5.22 | Heisenberg picture | unused-def | `STRUCT` | 0.00 | -3.00 |
| 5.28 | complex Lie groups Spin/SU/U/GL | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 5.33 | Wick rotation t→-iτ | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 5.23 | interaction picture | unused-neg | `STRUCT` | 0.00 | -4.00 |
| 5.24 | time-dependent perturbation | unused-neg | `STRUCT` | 0.00 | -4.00 |
| 5.6 | Jordan-Wigner construction | inv-direct | `STRUCT` | 0.00 | -5.00 |
| A.11 | ZX-calculus diagrammatic reasoning | unused-def | `STRUCT` | 0.00 | -5.00 |

### Load-bearing on `A4` — A4 fermion anti-commutation (CAR via JW) (2 ops)

| op | name | verdict | template | Φ_strict | Net_strict |
|---|---|---|---|---|---|
| 5.10 | ℤ/2-grading by (-1)^F | inv-direct | `PT_QUBIT` | 1.00 | -1.00 |
| 5.7 | CAR {c_i,c_j†}=δ_ij | inv-direct | `STRUCT` | 0.00 | -3.00 |

### Load-bearing on `A2W` — A2-T selective-retention / MDL waterline (17 ops)

| op | name | verdict | template | Φ_strict | Net_strict |
|---|---|---|---|---|---|
| 4.1 | probability measure P | inv-direct | `STRUCT` | 0.00 | -2.00 |
| 4.2 | expectation E_P[f] | inv-direct | `STRUCT` | 0.00 | -2.00 |
| 4.5 | Shannon entropy | inv-direct | `ENTROPY_BERN` | 1.00 | -2.00 |
| 4.6 | KL divergence | inv-direct | `ENTROPY_BERN` | 1.00 | -2.00 |
| 4.7 | mutual information I(X;Y) | inv-direct | `ENTROPY_BERN` | 1.00 | -2.00 |
| 4.8 | description length L(M) | inv-direct | `ENTROPY_BERN` | 1.00 | -2.00 |
| 4.9 | source coding (entropy) | inv-direct | `ENTROPY_BERN` | 1.00 | -2.00 |
| 4.12 | stationary distribution | inv-direct | `ENTROPY_BERN` | 1.00 | -2.00 |
| 4.3 | joint and marginal distributions | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 4.4 | conditional probability / Bayes | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 4.10 | rate-distortion bound | inv-direct | `ENTROPY_BERN` | 1.00 | -3.00 |
| 4.11 | discrete-time Markov chain | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 4.13 | continuous-time Markov process | unused-neg | `STRUCT` | 0.00 | -3.00 |
| 4.14 | correlation function C_n(s) | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 4.25 | conditional expectation E[·|sub-σ] | unused-deriv | `STRUCT` | 0.00 | -4.00 |
| A.9 | free entropy / free Fisher info | unused-def | `ENTROPY_BERN` | 1.00 | -4.00 |
| A.15 | martingales, multiway filtration | unused-def | `STRUCT` | 0.00 | -5.00 |

### Load-bearing on `STRAUCH` — discrete→continuum walk limit (Strauch 2006 + rapid decay) (4 ops)

| op | name | verdict | template | Φ_strict | Net_strict |
|---|---|---|---|---|---|
| 3.11 | discrete→continuum walk limit | inv-direct | `COARSE_STRAUCH` | 3.32 | -0.68 |
| A.13 | Brownian motion as continuum limit | unused-def | `COARSE_STRAUCH` | 3.32 | -1.68 |
| 3.3 | continuous-time quantum walks | inv-direct | `STRUCT` | 0.00 | -3.00 |
| A.14 | SDEs on L² | unused-def | `STRUCT` | 0.00 | -5.00 |

### Load-bearing on `BZJ` — BZJ critical-scaling regime (2 ops)

| op | name | verdict | template | Φ_strict | Net_strict |
|---|---|---|---|---|---|
| 4.48 | order parameter / phase diagram | inv-direct | `STRUCT` | 0.00 | -2.00 |
| 4.49 | critical exponents | inv-direct | `STRUCT` | 0.00 | -3.00 |

### Load-bearing on `RGFL` — RG flow structure (1 ops)

| op | name | verdict | template | Φ_strict | Net_strict |
|---|---|---|---|---|---|
| 4.52 | renormalization group flow | inv-direct | `RG` | 3.32 | -0.68 |

### Load-bearing on `LIE` — matrix-Lie-group structure (12 ops)

| op | name | verdict | template | Φ_strict | Net_strict |
|---|---|---|---|---|---|
| 3.2 | strong continuity of U(t) | inv-direct | `STRUCT` | 0.00 | -2.00 |
| 4.38 | trace identities under reps | inv-direct | `STRUCT` | 0.00 | -2.00 |
| 4.41 | exponential map exp(X) | inv-direct | `STRUCT` | 0.00 | -2.00 |
| 4.44 | one-parameter subgroup t↦exp(tX) | inv-direct | `STRUCT` | 0.00 | -2.00 |
| 3.1 | one-parameter unitary group U(t) | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 3.5 | Stone (real form) | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 4.39 | matrix Lie group | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 4.40 | Lie algebra | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 4.42 | structure constants f^c_{ab} | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 4.43 | Killing form K(X,Y) | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 4.35 | Wigner d-matrices d^j_{mm'}(θ) | inv-direct | `STRUCT` | 0.00 | -4.00 |
| A.20 | TQFT operations | unused-def | `TQFT` | 2.00 | -4.00 |

### Load-bearing on `COMPACT` — compact group/space (Peter-Weyl, Killing form) (5 ops)

| op | name | verdict | template | Φ_strict | Net_strict |
|---|---|---|---|---|---|
| 4.31 | character χ_ρ(g) | inv-direct | `STRUCT` | 0.00 | -2.00 |
| 4.32 | representation matrix elements | inv-direct | `STRUCT` | 0.00 | -2.00 |
| 4.30 | group representation ρ:G→𝒰(V) | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 4.33 | Schur orthogonality | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 4.37 | Clebsch-Gordan coefficients | inv-direct | `STRUCT` | 0.00 | -3.00 |

### Load-bearing on `FIN_DIM` — finite-dimensional Hilbert sector / finite primitive cell (7 ops)

| op | name | verdict | template | Φ_strict | Net_strict |
|---|---|---|---|---|---|
| 2.28 | orthogonal projection P_S | inv-direct | `PROJ_RANK2` | 2.00 | -1.00 |
| 4.24 | partial trace over subfactor | inv-direct | `PT_DIRAC` | 3.00 | -1.00 |
| 2.26 | trace Tr(T) | inv-direct | `STRUCT` | 0.00 | -2.00 |
| 2.29 | HS norm Tr(T*T) | inv-direct | `STRUCT` | 0.00 | -2.00 |
| 2.34 | determinant det(T) | inv-direct | `STRUCT` | 0.00 | -2.00 |
| 2.22 | trace-class ℬ_1(L²) | inv-direct | `STRUCT` | 0.00 | -3.00 |
| 2.23 | Hilbert-Schmidt ℬ_2(L²) | inv-direct | `STRUCT` | 0.00 | -3.00 |

## Methodology notes

- **Φ_lean = 0 for any op needing extras.** Two ops escape this: 0.4 (involutive cancellation) and 1.8 (conjugation). Both work on the bare free product F_inv(E) without further structure. **Everything else's compression is downstream of framework specialization**, which is itself the load-bearing structural commitment of the project.
- **§C smooth-manifold closure is PARTIAL**, marked by tag `CCLOSE`. Any op tagged `CCLOSE` is contingent on an open structural problem; its Net_strict treats §C as already closed for accounting purposes but the column should be read with that caveat.
- **§F field selection (`FF`)** is forced by P1' under the framework's current axiom slate; it's always in scope when ℂ-Hilbert structure is used.
- **Tag conjunctions are AND, not OR.** An op with extras `{SRS, E6, K3, FF}` requires all four; the absolute minimum-assumption Φ for that op is 0 unless all four hold.
- **L unchanged from first-pass table.** Strict accounting affects Φ, not specification cost.

## Cross-references

- `docs/wave_engine/compressibility_table.md` — first-pass (no assumption column).
- `../operator_sweep/operator_sweep_from_A1.md` — source catalog.
- `../framework/framework_axioms.md` — A1, A2-T, A4, A5-mass, P1' definitions.
- `docs/operator_sweep_audit_*.md` — per-op verdict and ontology.
