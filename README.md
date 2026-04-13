# Standard Model from Two Axioms

A framework that derives Standard Model parameters from two axioms on a single graph.

## Axioms

1. **Toggle**: The fundamental operation is a binary state change on a graph node.
2. **MDL**: The system minimizes free energy F, where F = description length (Minimum Description Length).

One unit identification: G (Newton's constant) sets the Planck scale (one toggle = one Planck time).

## What follows

From these axioms, the MDL-optimal graph is the **srs crystal net** (space group I4₁32, coordination number k\*=3, girth g=10). Two algebraic structures on this graph encode all physics:

| Pillar | Object | What it encodes |
|--------|--------|-----------------|
| **Flavor** | h = (√3+i√5)/2 | Mixing angles, CP phases, mass hierarchies |
| **Gauge** | Cl(6) = Cl(4)⊗Cl(2) | Gauge group, couplings, fermion content |

h is the Hashimoto (non-backtracking) eigenvalue at the P point of the BCC Brillouin zone. Both pillars derive from k\*=3.

## Results

~40 theorem-grade + ~8 A-grade derivations = 48 parameters. Zero free parameters. Zero physical identifications beyond axioms. One open parameter (A_s, needs new mathematics). Most recent promotion: R = 228/7 (neutrino splitting) from A- to theorem via the closed-form Ihara/Chebyshev derivation (see `docs/R_theorem.md`).

| Quantity | Predicted | Observed | Accuracy |
|----------|-----------|----------|----------|
| Higgs mass m_h | 126.2 GeV | 125.25 GeV | 0.8% |
| Top mass m_t | 172.71 GeV | 172.69 GeV | 0.01% |
| Higgs VEV v | 245.64 GeV | 246.22 GeV | 0.24% |
| Baryon asymmetry η_B | 6.09×10⁻¹⁰ | 6.12×10⁻¹⁰ | 0.5% |
| PMNS mixing (4 obs) | χ²/dof = 0.22 | p ≈ 0.93 | all <0.6σ |
| CKM elements | V_us, V_cb, V_ub | PDG values | 0.08-3.5% |
| Neutrino splitting R | 228/7 = 32.571 | 32.576 (PDG) | 0.015% (<0.01σ, theorem) |
| Cosmological constant Λ | 3/N² | observed | <1% |
| Dark matter Ω_DM/Ω_m | 0.842 | 0.842 | <0.1% |

Full parameter scorecard: [`results/parameters.csv`](results/parameters.csv)

## Testable predictions

| Prediction | Experiment | Timeline |
|-----------|------------|----------|
| α₂₁ = 162.39° (Majorana phase) | nEXO, LEGEND-1000 | 2030+ |
| δ_CP = 249.85° (Dirac phase) | DUNE, Hyper-K | 2028+ |
| m_ν₁ = 0 (massless lightest neutrino) | KATRIN, Project 8 | 2027+ |
| m_ββ = 2.55 meV (0νββ amplitude) | nEXO | 2030+ |
| θ₂₃ = 48.72° (non-maximal) | DUNE | 2028 |
| R-parity violated | LHC RPV searches | ongoing |
| Gluino at 6970 GeV | FCC-hh (100 TeV) | 2040+ |
| No WIMP dark matter | LZ, XENONnT | ongoing |

## Repository structure

```
verify.py                          # Run all backbone proofs
results/parameters.csv             # Full parameter scorecard (grades, chains)
proofs/
  common.py                        # Shared lattice infrastructure
  foundations/                      # k*=3, srs, generations, H^2=k*I, Ramanujan
  flavor/                          # CKM, PMNS, CP phases, Ihara splitting
  masses/                          # Hierarchy, Koide, neutrinos, SUSY
  cosmology/                       # Lambda, Omega_DM, eta_B, n_s
  gauge/                           # Cl(6), Pati-Salam, R-parity
explorations/                      # Hypotheses tested (successes and failures)
  negative_results.md              # What didn't work and why
docs/
  derivations.md                   # Complete derivation chains (~500 lines)
  honest_assessment.md             # What's proven, what's not, what would falsify
  predictions.md                   # Testable predictions with experiments
```

## Requirements

Python 3.8+ with `numpy` and `matplotlib`.

```bash
pip install numpy matplotlib
```

## Quick start

```bash
python3 verify.py                                   # Run all 15 backbone proofs
python3 proofs/foundations/srs_generation_c3.py      # Generation definition at P point
python3 proofs/foundations/srs_p_point_algebra.py    # H²=k*I, Ramanujan, everything=k*
python3 proofs/flavor/srs_unified_mixing.py          # All PMNS angles from h
python3 proofs/foundations/srs_foundation_closure.py  # Verify all foundation theorems
```

## Key derivation chains

See [`docs/derivations.md`](docs/derivations.md) for complete chains. Highlights:

- **Generation definition**: C₃ irreps at BCC P point (`proofs/foundations/srs_generation_c3.py`)
- **CP phases**: Hashimoto h^n with chirality selection (`proofs/flavor/srs_hashimoto_seesaw_proof.py`)
- **Mass scale**: MDL mean-field + δ² Dyson (`proofs/masses/srs_mdl_meanfield_theorem.py`)
- **CKM**: Tree approximation at z\*=17/6 (`proofs/flavor/srs_ckm_tree_derivation.py`)
- **Pati-Salam**: Cl(6) → Spin(6) ≅ SU(4), 105/105 structure constants (`proofs/gauge/srs_so10_embedding.py`)
- **Baryon asymmetry**: Laplace concentration at P (`proofs/cosmology/srs_eta_b_exact.py`)
- **Foundation closures**: z\*, M_R, η_B -- no identifications needed (`proofs/foundations/srs_foundation_closure.py`)

## Honest assessment

This framework derives 48 of 50 quantities (45 SM + 5 cosmological) from two axioms. The derivations range from theorem-grade (39 items: complete proof from axioms) to A-grade (9 items: complete chains with computational precision as the only gap). One parameter (A_s) is genuinely open.

Every script is a computational verification, not a fit. No parameters are adjusted. See [`docs/honest_assessment.md`](docs/honest_assessment.md) for detailed caveats and [`docs/predictions.md`](docs/predictions.md) for falsifiable predictions.

## License

MIT
