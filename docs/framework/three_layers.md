# The three layers — physics is the triple (D, ω, {A(O)})

One page. The WHY behind the master-object discipline, for every future handoff.

## The triple

Physics in this framework is **(D, ω, {A(O)})**: the operator, the state, and the net of local
algebras. Three layers, built in order:

| Layer | Object | Master file | Status |
|------|--------|-------------|--------|
| **1** | **D — the operator's global spectrum.** Traces, eigenvalues, characters, resolvent channels. State-independent. | `derivation_topdown/bridge/the_run.py` | **BUILT** — ~95 ✅ live here (gauge couplings, CKM/PMNS, Koide masses, oblique forms, η_B, Ω-ratios, θ_QCD…). Untouchable. |
| **2** | **ω — the state's GLOBAL structure.** The run as KMS state of the tick; global functionals of ω. | (M0/M2/MC/MG results) | **BUILT** — κ = h/t_P, KMS/tick 2π, clock map, frame identity, coasting spine + eras at form level, Hubble-tension sign, c_s²=1/3, fluctuation spectrum. |
| **3** | **ω — the state's LOCAL structure: the net O ↦ (A(O), ω\|_O).** | `derivation_topdown/state/the_net.py` | **BUILDING** — the ONE unbuilt layer. ML-0 built the net skeleton; ML-1+ extend it. |

Only two degenerate region-shapes of Layer 3 existed before the ML-track: region = **one cell**
(M0's C-projector) and region = the **global tick** subalgebra (M0-2R's U(1)/2π). Both are the two
regression anchors of `the_net.py`.

## Every open parameter is one region-shape of the net

The ~48 non-✅ rows of `target_parameters.md` collapse to the ONE Layer-3 object evaluated at
different regions (see `parameter_bins_and_local_net_throughline_2026-07-08.md`):

- **Bin L-metric** — causal-**diamond** modular flow (ML-1) → Newton's G 2π, era exponents, native
  z_eq, θ_* (booked ~9× pressure), r_*, r_drag, θ_MC, z_*.
- **Bin L-sector** — the **DHR sector category** (ML-2) → m_e −70 ppm, m_μ, m_ν scale, B1 hadron
  anchoring → Y_p, D/H, ³He, ⁷Li.
- **Bin L-response** — local **density response** (downstream) → n_s, σ_8, S_8, f σ_8. Needs Bin
  L-metric's eras as input; do not attack first.
- **Bin S** (Layers 1–2, ~95 ✅) is restriction-independent — Layer-3 work **cannot** move those
  numbers. **Bin X** rows (τ, z_reion, SUSY block) are framework-external; declare out-of-scope.

## The discipline (why this file exists)

Layer 1 converged because it had a **master object file** and every session's deliverable was "add
a forced read to the one object." The M/MC/MG stations instead built scratch probes against
observable-framed targets; the composable object never accreted, so each fresh session needed a
fresh diagnosis to see the whole.

**Rule:** Layer-3 math **accretes in `the_net.py`**. Extend it every station; never a scratch probe.
Frame each station as *"add a region-class / forced read to the net,"* never *"attack observable X."*
The two regression anchors (cell projector, tick 2π) must always hold. This extends the
"one master object, no regression" architecture rule to Layer 3.

Poisons (standing): no targets before blind confronts; ħ/G/sector goal-seek forbidden; an open miss
stays open; θ_* stays OPEN until ML-4; the species-lift may terminate as a PRICED adoption.
