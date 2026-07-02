# Wave Simulator — Usage and Results

The wave simulator is implemented in two complementary scripts:

- `proofs/wave_engine/simulator_lean.py` — minimal Layer 0+1 simulator validating the mechanism on the framework's lean (substrate-intrinsic) catalog.
- `proofs/wave_engine/simulator.py` — full 195-op simulator with tag-establishment dynamics.

## Lean simulator results

Configuration: |E|=6 (six undirected edges per srs primitive cell), n=10 (girth scale).

Initial wave-state: `{A1, E_FIN}` with 60,466,176 raw configurations.

Cascade trace (12 ops fire, in catalog order):

```
tick  op    name                           refs                Φ_marg  classes
  3   0.4   involutive cancellation        reduced             +2.37   11,718,750
  8   1.8   conjugation                    cyclic+reduced      +3.58      976,887
  9   1.10  abelianization                 abel+cyc+red       +13.90           64
```

Halts at **64 abelianization classes** with **total Φ = 19.85 bits** (944,784× compression of raw configurations). 12 derived objects emitted.

**Three findings on the lean run:**

1. **The mechanism halts coherently** at a recognizable substrate structure (the abelianization image (Z/2)^6).
2. **Total Φ is path-independent** — greedy mode fires 1 op for 19.85 bits; cascade fires 12 ops for the same 19.85. The final compression is a structural property of the lean catalog.
3. **Net under cascade is −4.15 bits** — the lean ontology costs more spec than it compresses on its own. The framework BUYS spec cost up front in exchange for unlocking high-Φ ops downstream once tags are established.

## Full simulator results

Configuration as lean. Initial tags: `{A1, E_FIN, A2W, P1, A5M, E6, K3, ORDER}`.

### Cascade order

The wave propagates in roughly 5 phases, visible in the trace:

| ticks | phase | what fires |
|---|---|---|
| 1–3 | Lean compression | 0.4 → 1.8 → 1.10 (substrate identical-state collapse, +19.85 bits) |
| 4–8 | Tag-unlock core | 2.22 → 3.1 → 3.5 → 3.11 establish FIN_DIM, LIE, FF, STRAUCH |
| 9–17 | Information theory + symmetry | 4.5–4.16 (entropy/MI/MDL); 4.16 establishes C3; 4.21 establishes K4Q+SRS+CRYSTAL |
| 18–35 | SRS-locked content | Bloch decomp, Hashimoto, Pati-Salam fragments; 4.45 establishes THERM; 4.51 establishes BZJ+N_HUB |
| 36–67 | Quantum + appendix | 5.1 establishes C_REP; 5.6 establishes A4; modular forms A.16/17/18 fire at +17.48 each |

The order of tag-establishment matches the framework's actual derivation history.

### Two reading modes (post T1.1 template-dedupe, 2026-04-27)

| mode | gate rule | ops fired | Φ_total | L_total | Net |
|---|---|---|---|---|---|
| Strict A2-retention | fire if Φ > 0 OR establishes tag | ~67/195 | 94.15 | ~230 | **−135.85** |
| Fire-every-firable | fire if extras ⊆ tags | 173/195 | 94.15 | 522 | **−427.85** |

Strict A2 is the A2-faithful reading. Fire-every-firable shows reachability. Pre-T1.1 numbers had Φ = 183.34 (95-bit overcount; see `cost_methodology.md` and `mechanism.md`).

### Ops that did NOT fire

22 ops blocked. After the LORENTZ_SIG / CCLOSE tag-split (2026-04-27):

- 15 ops blocked by `CCLOSE` only (smooth-manifold ops without signature dependence)
- 1 op blocked by `LORENTZ_SIG` only (6.10 Lorentzian metric)
- 6 ops blocked by BOTH (cosmology cluster: 6.18 FLRW, 6.19 Einstein, 6.20 Friedmann, 6.21 Hubble, 6.22 a(t), 6.23 stress-energy)

This includes all of Layer 6 except 6.6 (exterior d on chain complex), 6.15 (graph geodesics), 6.16 (discrete parallel transport), 6.24 (causal structure intrinsic to multiway). Plus Appendix A.19 (quantum gravity) and A.21 (CFT/Virasoro).

`LORENTZ_SIG` is BLOCKED at substrate level by the bounded-D² obstruction (`memory/project_lorentzian_signature_route_c_blocked_2026-04-26.md`). `CCLOSE` is research-level open. See `closure_experiment.md` for the 4-way scenario diff.

### Top compression contributors

| op | template | Φ contribution | note |
|---|---|---|---|
| 1.10 | QUOT_ABEL | 13.90 | abelianization given involutive+cyclic refinements |
| A.16, A.18 | MODULAR | 23.48 each | modular forms / Selberg zeta — currently `unused-deferred` |
| A.17 | MODULAR | 23.48 | automorphic L-functions |
| 6.8 | HOMOL_E2 | 6.00 | de Rham cohomology (CCLOSE-blocked in baseline) |
| 4.45–4.47, 5.34–5.35, A.7 | THERMAL_SRS | 6.00 each | partition function / thermal cluster |

The MODULAR cluster contributes ~52 bits — ~28% of the entire wave's Φ budget — and these ops are currently unused-deferred per the audit. This is the framework's largest unrealized compression gain from the modular workstream.

## Running the simulator

```bash
# Lean run (sanity check on Layer 0+1)
python3 proofs/wave_engine/simulator_lean.py

# Full run (195 ops)
python3 proofs/wave_engine/simulator.py
```

Both scripts print a per-tick trace and a halting summary. Output is deterministic given the catalog.

## Modifying the catalog

The 195-op catalog is hard-coded in `proofs/wave_engine/simulator.py` as the `CATALOG` list. Each entry is a tuple `(id, layer, name, template, L, extras, refinement)`. To add an op:

1. Add a new tuple to `CATALOG` with appropriate fields
2. If the op establishes new tags, add to `ESTABLISHES`
3. If the op uses a new template, add to `PHI_TEMPLATE`
4. If the op references a new tag, add to `ALL_TAGS` and decide initial vs cascade-established
5. Re-run; the simulator picks up the changes automatically

This is the entry point for Pass A (coverage audit) — adding missing ops to the catalog and re-running to see what unlocks.
