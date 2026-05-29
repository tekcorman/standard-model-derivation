# Testable Predictions

Every prediction below is derived from one structural axiom (A1: binary self-inverse toggle) plus a definitional commitment (P1') plus an empirical labeling rule (A5-mass) with zero fitted constants. Values and σ are pulled from the auto-generated `predicted_parameters.md` at the repo root (run `python3 run_predictions.py` to regenerate). Grades follow [`parameter_linter.md`](parameter_linter.md) Clauses 5–9.

---

## Near-Term Predictions (2026–2032)

| Prediction | Value | Current status | Experiment | Timeline |
|-----------|-------|---------------|------------|----------|
| **δ\_CP (PMNS, Dirac phase)** | **180°** (arccos(−1) via V_{−1}–T_{B-L} identity) | NuFIT 6.0 IC19: 177°⁺¹⁹₋₂₀ (+0.16σ) | DUNE, Hyper-K | 2028–2032 |
| θ\_23 non-maximal | 48.72° | 49.2° ± 1.3° (−0.37σ) | DUNE | 2028 |
| m\_ν1 = 0 (massless lightest neutrino) | 0 eV | < 0.8 eV (KATRIN) — consistent | KATRIN, Project 8 | 2027–2030 |
| Normal ordering (NH) | NH | ~3σ preference | JUNO, DUNE | 2027–2030 |
| **Cosmic birefringence β** | **0.354°** = sin(arg h)·α\_EM(M\_Z) | 0.342° ± 0.094° (+0.13σ) | LiteBIRD (±0.05°) | ~2032 |
| **CMB hemispherical A** | **1/15 = 0.0667** | 0.07 ± 0.02 (−0.17σ) | CMB-S4 high-resolution | 2030+ |
| **β hard cap \|β\| ≤ α\_EM** | **≤ 0.418°** | observed at ~85% of bound | any future birefringence measurement | ongoing |
| Per-ℓ CMB parity amplitudes | −7/48 at ℓ=4, −13/64 at ℓ=6, +297/1024 at ℓ=8 | consistent at current precision | Planck, LiteBIRD reanalysis | ongoing |
| R-parity violated | RPV signatures (I4₁32 has no inversion) | no RPV signal yet | LHC Run 3 | ongoing |
| No WIMP dark matter | null direct-detection (DM = uncompressed multiway branches, gauge-decoupled) | no signal | LZ, XENONnT, PandaX | ongoing |

> **δ\_CP^PMNS provenance.** The 180° prediction comes from the parameter-free V_{−1}–T_{B-L} symmetry-breaking identity (graduated 2026-05-05; THEOREM-GRADE-STRUCTURAL-CONDITIONAL on Need-D-3). The earlier Hashimoto-phase route (g−1)·arg(h*) ≈ 249.85° was **falsified at +3.83σ vs NuFIT 6.0 IC19** in 2026-05-02 and is preserved at `predictions/retracted/delta_CP_PMNS.py` as honest history. The same V_{−1}–T_{B-L} identity also fixes δ\_CP^CKM = arccos(1/3) = 70.53° (+0.68σ vs PDG-2024), an independent corroboration.

> **β cosmic birefringence provenance.** The 0.354° value uses the framework's RG-run α\_EM(M\_Z) ≈ 1/127.93 with c=1·sin(arg h) = √(5/8) (Theorem 4 + amplitude-class dark correction). Earlier publications quoted 0.331° from observed α\_EM = 1/137.036 — that was a Clause-9 smuggle (observed α(0) is not framework-derived without the Δα low-energy hadronic threshold, which is Clause-9-blocked / Move-1-out-of-scope). The downgrade is preserved as a named gap; the structural form remains theorem-grade.

---

## Medium-Term Predictions (2030–2040)

| Prediction | Value | Experiment | Timeline |
|-----------|-------|------------|----------|
| α\_21 (Majorana phase) | 162.39° = g·arg(h) | nEXO, LEGEND-1000 | 2030+ |
| α\_31 (Majorana phase) | 324.78° = 2g·arg(h) | nEXO, LEGEND-1000 (currently unconstrained) | 2030+ |
| m\_ββ (0νββ amplitude) | ≈ 2.55 meV (from m\_ν₂ + α\_21 chain) | nEXO, LEGEND-1000 | 2030+ |

α\_21 and α\_31 currently carry STRUCTURAL-DERIVATION-CONDITIONAL grade (re-graded 2026-05-12 from earlier UNIQUE-THEOREM-GRADE-CONDITIONAL, which had been inflated); they are framework predictions but their structural derivation has named gaps tracked in [`parameter_uniqueness_ledger.md`](parameter_uniqueness_ledger.md) Rows P35 / P36.

---

## Long-Term Predictions (2040+)

| Prediction | Value | Experiment | Timeline |
|-----------|-------|------------|----------|
| η\_lattice (dim-6 LIV at ~147 PeV) | 1/12 | SWGO, future UHE γ | below current sensitivity |
| Universe transparency onset | ~147 PeV | SWGO; tentative GRB 221009A signal | ongoing |

---

## SUSY Spectrum — honest-conditional on literal-particle interpretation

> **Status (2026-05-27, post-SUSY-load-bearing audit):** the framework's substrate-derived matter content is 3 PS generations + 2 Higgs doublets, with NO superpartners (Cl(6) Fock all-fermionic per Path-E recheck 2026-05-12; A1 thermal-apparatus closure 2026-05-27). The MSSM β-coefficient values (33/5, 1, −3) the framework predicts are derived by algebraic inversion under α_GUT⁻¹ = 24 ([`theorem_beta_coefficients_derived.md`](../theorems/theorem_beta_coefficients_derived.md), mathematically complete), NOT by counting literal sparticles. The gap between substrate-derived 2HDM β (b_2 = −3) and observation-imposed MSSM β (b_2 = +1) is precisely characterized as Δb_2 = +4 ([R-19](../audits/registers/structural_residue_register.md)).
>
> The values below are **honest-conditional predictions IF the literal-particle interpretation of [ADOPTED-MSSM-Sb](../audits/registers/adoption_register.md) holds** (i.e., if literal sparticles physically realize the observed β-coefficient values via m\_{3/2} = M\_P / √(N^(1/2)) + gravity mediation from the srs lattice structure). The framework does NOT commit to this interpretation. None of these rows have shipped `predictions/*.py` files; they sit at ❌ in [`target_parameters.md`](target_parameters.md) and at RETIRED-conditional in [`parameter_uniqueness_ledger.md`](parameter_uniqueness_ledger.md) Row P58.
>
> Per the SUSY-load-bearing audit, no framework prediction in the live numerical comparison ([`../../predicted_parameters.md`](../../predicted_parameters.md)) depends on these values.

| Particle | Mass (GeV) — conditional | Detection channel |
|----------|--------------------------|-------------------|
| Gluino | ~6970 | FCC-hh: gluino pair → jets + MET |
| Wino (chargino / neutralino) | ~2960 | FCC-hh: disappearing tracks |
| Bino (lightest neutralino) | ~1280 | indirect: RPV decays |
| Gravitino | ~1732 | cosmological: not stable DM |
| Stop | ~5000 | FCC-hh |
| Heavy Higgs (H, A) | ~8000 | FCC-hh: τ⁺τ⁻ resonance |
| tan β | 44.73 (documented) / 60.07 (live RGE; Row P46 reconciliation pending) | b–τ Yukawa unification + GJ = 3 |

R-parity is structurally VIOLATED in the framework regardless of literal-particle interpretation (I4₁32 has no inversion symmetry). Proton stability is maintained by Z₃ triality (the generation quantum number), not R-parity. **Absence of sparticle signals at HL-LHC / FCC-hh is CONSISTENT with the framework**, not falsifying — substrate-derived matter content is 2HDM, not MSSM.

---

## Qualitative predictions

These follow structurally from the framework:

1. **Dark matter is not a particle.** DM = uncompressed multiway branches of the graph, outside the Cl(6) Fock states the gauge sector reads — no gauge channel, no direct-detection coupling. No WIMP, no axion, no sterile neutrino. See `theorem_dark_sector_multiaxial_waterfilling_candidate.md` §§9–10 for the gauge-vs-gravity asymmetry.

2. **The cosmological constant is not fine-tuned.** Λ\_substrate = 1/N² is forced by Margolus-Levitin on a coasting toggle graph; the observed factor-of-2 vs ΛCDM-fit is the parametric-class translation Λ\_LCDM = 3·Ω\_Λ\_LCDM·Λ\_substrate (Row P24-sibling), which is structurally accounted for, not an unsolved fine-tuning. There is no coincidence problem; Λ decreases as N grows.

3. **w\_DE = −1 exactly.** The dark-energy equation of state is exactly −1 because the toggle graph has no dynamics at the Λ scale; rate²-scaling cancels (16/15)² in the ratio. UNIQUE-THEOREM-GRADE.

4. **Proton stability via Z₃ triality.** Despite R-parity violation, the proton is stabilized by Z₃ triality (the generation quantum number). Baryon-number violation requires simultaneous change of all three generation labels — exponentially suppressed at framework scale.

5. **The Hubble tension is a structural prediction, not an anomaly.** H₀^substrate (CMB-side) = 68.18 km/s/Mpc and H₀^observer (SH0ES-side) = (16/15)·H₀^substrate = 72.72 km/s/Mpc; the (16/15) rate-gap = ε_toggle·(1/k*) = (1/5)(1/3) = 1/15 is the predicted offset between the two measurement channels.

---

## What would falsify the framework

Listed in order from most to least decisive. Each row names what the falsification would kill and what would survive — the framework's structural skeleton is not a single-point-of-failure construction.

| Observation | What it would kill | What survives |
|------------|--------------------|---------------|
| θ\_23 = 45.00° ± 0.3° | dark-correction mechanism (tan² arg h amplitude rule) | gauge, generations, CKM |
| δ\_CP^PMNS far from 180° (e.g., < 150° or > 210° at 5σ) | V_{−1}–T_{B-L} identity for the PMNS channel | gauge, CKM (same identity gives δ\_CP^CKM ≈ 70°) |
| m\_ν1 > 0 established (KATRIN positive signal) | W45 trivial-mode walker closure (h^g ≡ +1 ⇒ no ν\_R Majorana mass) | most else; would force a 3rd ν\_R |
| β > α\_EM ≈ 0.418° | Theorem 4 (c\_1 = 0 on all 2D slices) + amplitude-class dark correction | gauge, flavor |
| β ≠ 0.354° ± 0.01° once measured at LiteBIRD precision | the specific amplitude-class form sin(arg h)·α\_EM | Theorem 4, gauge, flavor |
| A (CMB hemispherical) ≠ 1/15 at high precision | Theorem 2 (ε\_CP · 1/k\* = 1/5 · 1/3) | flavor sector |
| Per-ℓ parity amplitudes ≠ predicted rationals | Theorem 1 (cubic-moment formula) at higher n | the n = 1 case (= A) and gauge |
| WIMP dark matter found | DM = uncompressed multiway branches | all particle physics |
| 4th generation found | C₃-on-observer generation mechanism | would require revision |
| Direct DM detection signal | gauge-decoupled DM (multi-axial waterfilling §§9–10) | gauge, flavor |
| No SUSY below ~10 TeV at FCC-hh | MSSM mass hierarchy pathway | gauge, flavor (would need different mass mechanism) |
| Proton decay observed (at rate inconsistent with Z₃-suppressed channels) | Z₃ triality protection | partial revision |
| w\_DE ≠ −1 measured | Λ\_substrate = 1/N² coasting closure | particle-physics sector unaffected |

---

## How to verify

```bash
python3 verify.py                              # backbone proof suite (25 proofs, ~10s)
python3 run_predictions.py                     # regenerate predicted_parameters.md
python3 predictions/_validate_dag.py           # check predictions/ DAG self-containment
```

Every value in this document is one Python run away. The canonical numerical authority is `predicted_parameters.md` at the repo root, regenerated from the live `predictions/*.py` files via the SECTORS manifest in `run_predictions.py`. Grades and conditionals are tracked in [`target_parameters.md`](target_parameters.md) and [`parameter_uniqueness_ledger.md`](parameter_uniqueness_ledger.md).
