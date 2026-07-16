# The Bridge Kit (CB-1)

`bridge_vectors.json` is the **one canonical, machine-generated** JSON containing every
value/structure the the upstream engine bridge work (`~/projects/the upstream engine`) consumes from this repo, with
per-value provenance and a `--check` regression mode. It kills the prose-copy failure mode
across the repo boundary: the upstream engine never hand-types a number from a doc here; it vendors this file
and re-derives nothing.

Built by `scripts/export_bridge_vectors.py`. Pre-registered FROZEN in
internal research notes (contracts BK-0..BK-7). Schema frozen in
internal research notes sec CB-1 (normative).

**This station moved no physics number.** Every value below is a *read* or a *re-expression*
of something the engine (`derivation_topdown/`) already computes; nothing here was derived,
tuned, or fit.

## Refresh protocol

**the upstream engine never hand-edits its vendored copy.** Refresh means: re-export in THIS repo at a named
commit, then copy the file across.

```
cd .
python3 scripts/export_bridge_vectors.py --export      # writes bridge_kit/bridge_vectors.json
python3 scripts/export_bridge_vectors.py               # bare = --check: recompute + diff, twice
python3 scripts/export_bridge_vectors.py               # (run it twice; both must be green)
git add bridge_kit/bridge_vectors.json
git commit -m "..."                                      # note the resulting commit hash
```

Then, on the the upstream engine side, vendor the file tagged with `meta.repo_commit` from the JSON itself —
that is the authoritative "as-of" pointer, not a doc reference. `--check` mode (the bare
invocation) is what `verify.py`'s BACKBONE entry runs at INTEGRATION; it recomputes every
field fresh and diffs against the committed JSON at `rtol 1e-12`, so any drift between the
engine and the committed export is caught automatically on every `verify.py` run.

## Schema (`schema: 1`)

### `meta`
| field | meaning | provenance |
|---|---|---|
| `repo_commit` | `git rev-parse HEAD` at export time | shell, computed at export time |
| `export_utc` | UTC export timestamp (ISO 8601) | `datetime.now(timezone.utc)` |
| `exporter` | this file's own repo-relative path | literal |
| `schema` | schema version (frozen at 1 by the pre-reg) | literal |

**Not gated in `--check`.** `export_utc` differs on every run by construction (wall clock).
`repo_commit` is printed and compared for information only, never gated — this repo runs an
auto-sync cron (per the standing env note), so a commit can legitimately land between an
`--export` and a later bare `--check` in the same session without any payload value changing.
Gating on either field would produce failures unrelated to the actual bridge vectors.

### `substrate` — the srs crystal (`derivation_topdown/dirac_srs_mdl/srs.py`)
| field | value | provenance |
|---|---|---|
| `K` | 3 | `srs.DEG`, `srs.py:12` |
| `GIRTH` | 10 | `the_run.GIRTH` = `read_girth()`, `the_run.py:78` (off the Hashimoto renewal sequence — not a typed exponent) |
| `NV` | 4 | `srs.NV`, `srs.py:11` |
| `NE` | 6 | `len(srs.EDGES)`, `srs.py:14-15` |
| `b1` | 3 | `NE - NV + 1` (Euler formula for the Z^3 deck rank); cross-checked against `the_run.read_geometry()`'s own `b1`, `the_run.py:94` |
| `edges` | `[[i, j, [tx,ty,tz]], ...]` | `srs.EDGES` verbatim, `srs.py:14-15` — the K4-cell maximal-abelian-cover presentation (tail, head, Z^3 homology vector) |

BK-1 gate: `check_bk1()` re-derives every one of these from `srs` directly and asserts
identity (not just presence) against the JSON block.

### `run` — the master run object (`derivation_topdown/bridge/the_run.py`)
| field | value | provenance |
|---|---|---|
| `alpha_1` | ≈0.039018442310623 | `the_run.U_RUN`, `the_run.py:324` — `= ρ^(GIRTH-2)`, `ρ=(K-1)/K` |
| `rho_survival` | 2/3 | `the_run.RHO`, `the_run.py:323` — `Fraction(K-1, K)`, computed off `K` |
| `rho_step` | ≈0.078036884621247 | `the_run.read_run()[0]`, `the_run.py:481-486` |
| `arrow` | `true` | `the_run.read_run()[1]` — forward convergence; backward is ill-posed |

BK-2 gate: `check_bk2()` asserts `alpha_1 == the_run.U_RUN` exactly, cross-checks it
**READ-ONLY** against `predictions/_value_locks.json["values"]["alpha_1"]` (rtol 1e-12; this
script never opens that file for writing, and never calls `scripts/value_lock.py --freeze`),
and asserts `arrow is True`.

### `thermo` — the tick-sector Gibbs/KMS temperature
| field | value | provenance |
|---|---|---|
| `u_c` | 0.5 | `1.0/(K-1)` — REUSE MAP: `derivation_topdown/adapters/thermal_time.py:151` (`u_c = 1.0/q`, `q=k-1`). Re-expressed, never imported (`thermal_time.py` is a flat script that `sys.exit()`s on import). |
| `beta_eff` | ≈5.101147368611 | `2*math.log(u_c/alpha_1)` — REUSE MAP: `thermal_time.py:209`. Matches the KMS-2 station's own derived `beta_eff=5.1011473686` (G5a adapter, 2026-07-08). |
| `ln2` | ≈0.693147180560 | `math.log(2)` — the Landauer-lock constant (`β·κ = ln2`) |

BK-3 gate: `check_bk3()` recomputes `beta_eff` from the JSON's own `u_c`/`alpha_1` fields and
asserts exact agreement — this is an *internal consistency* check, not a re-derivation.

### `clock` — the observer's clock read (`the_run.read_clock()`, `the_run.py:83-90`)
| field | value | meaning |
|---|---|---|
| `eps` | 0.2 (=1/5 exactly) | the toggle-disconfirmation `ε` |
| `clock` | ≈1.066666666667 (=16/15 exactly) | `1 + ε/K` |

### `spectral` — the adjacency spectrum
| field | value | provenance |
|---|---|---|
| `perron` | 3.0 | `the_run.adjacency_energies()[0]`, `the_run.py:71-76` — the Perron–Frobenius dominant adjacency eigenvalue at Γ |
| `irrep3` | -1.0 | `the_run.adjacency_energies()[1]` — the 3-irrep (generation-multiplicity) eigenvalue |
| `gap_additive` | ≈0.267949192431 | `(K-1) - sqrt(K)` — see note below |
| `adjacency_eigs_at_k` | `{label: [4 eigenvalues]}` | `srs.adjacency(k)` (`srs.py:17-22`), `np.linalg.eigvalsh`, at the frozen k-points (below) |

**`gap_additive` provenance note (a FINDING, not a silent renumbering):** the pre-reg names
the literal quantity from `proofs/foundations/srs_ramanujan_theorem.py:676` ("the spectral gap
lambda_1 = 2 - sqrt(3)") and explicitly forbids adding a `read_gap()` getter to the engine.
This exporter computes it as `(K-1) - sqrt(K)` using the already-structural constant
`K = srs.DEG`, rather than typing the bare floats `2` and `3` — numerically **identical** to
`2 - sqrt(3)` for this crystal's `K=3`, but satisfying the no-hand-typed-floats poison. No
getter was added anywhere; the formula lives only in `build_spectral()` in the exporter.

### `zeta` — the graph-zeta / matter-weighted determinant
| field | meaning | provenance |
|---|---|---|
| `det_coeffs_at_k` | `{label: [[re,im], ...]}`, ascending powers of `u`, `det(I - u·B(k)) = Σ c_n u^n` | built from `np.linalg.eigvals(srs.hashimoto(k))` (`srs.py:42-49`) by polynomial convolution `Π(1 - λ_i u)` — this exporter's own math, cross-checked (BK-4 self-test) against `np.linalg.slogdet(I - uB(k))` at 5 random `u` |
| `trW_INT` | `[[re,im], ...]` for `L=1..20` | `Tr(W_INT^L)` via 20 successive matmuls; `W_INT` REUSE MAP below |
| `loop_identity_residual_at_alpha1` | ≈1.18e-17 | `|-log det(I-u·W_INT) - Σ_{L≤40} (u^L/L)·Tr(W_INT^L)|` at `u=alpha_1` |

`W_INT` REUSE MAP (verbatim block-assignment logic, re-expressed — never imported, since
`zeta_gauge.py` is a flat script that `sys.exit()`s on import):
- `derivation_topdown/adapters/zeta_gauge.py:522-541` — the `W_INT` build itself (dart-block
  assignment over `B_GAMMA = srs.hashimoto(Γ).real`).
- `zeta_gauge.py:522-524` — the Cl(6) generators come from
  `simulator.srs_engine.utils.AlgebraicUtility.cl6_generators()` (6 generators, 8×8 complex
  each; `simulator/srs_engine/utils/algebraic.py`). This is a proper importable package (has
  `__init__.py` at every level), **not** one of the flat adapters — importing it is not the
  "never import an adapter" poison.
- `W_INT` is `8·ND × 8·ND` with `ND = 2·len(srs.EDGES) = 12` → 96×96.
- The `loop_identity_residual_at_alpha1` gate itself is BK-4's own tolerance (`< 1e-9`,
  expected ~1e-17 — matches exactly what `zeta_gauge.py`'s own ZG-4(b) contract measures at
  `u=alpha_1`: `1.179e-17` when that adapter is run standalone).

### `mdl` — the description-length comparison (`proofs/foundations/dl_comparison.py`)
| field | meaning | provenance |
|---|---|---|
| `dl_table` | `{net_name: DL_bits}` for srs/ths/eta/utj (crystal_3d), honeycomb (crystal_2d), Petersen/K_{3,3} (finite), random(N=100/1000) (finite) | `dl_comparison.dl_srs()` / `dl_ths()` / `dl_eta()` / `dl_utj()` / `dl_honeycomb_2d()` / `dl_petersen()` / `dl_k33()` / `dl_random(N)` — **imported directly** (BK-5 import route: the file has an `if __name__ == "__main__":` guard around its only side-effecting code, verified programmatically by this exporter at every run, not just once by inspection) |
| `srs_breakdown` | `{space_group, n_orbits, wyckoff, coordinates, edges, chirality}` (bits) | the second return value of `dl_comparison.dl_srs()` |

BK-5 gate: `srs (Laves)`'s total DL is asserted strictly less than every other `crystal_3d`
entry (`ths`, `eta net`, `utj net`) — a regression of the existing locked theorem
(`dl_comparison.py`'s own `main()` proof), not a new numeric target.

## Frozen k-point and seed choices

Two **independent** `np.random.default_rng(0)` streams are used (documented explicitly here
so the ordering is never a surprise — this is a deliberate deviation from `zeta_gauge.py`'s
own convention of drawing both `u`-samples and `k`-points sequentially off *one* shared
`np.random.default_rng(0)` object; here they are two separately-seeded draws):

1. **k-points** (`frozen_kpoints()`): `rng = np.random.default_rng(0)`; the point set is
   `Γ = (0,0,0)` plus `rng.random(3)` drawn three times in a fresh loop — i.e. the first three
   3-vectors off a `seed(0)` generator. These four k-points are shared by BOTH
   `spectral.adjacency_eigs_at_k` and `zeta.det_coeffs_at_k` (same labels, same order), so the
   two blocks are directly comparable point-for-point.
2. **BK-4 self-test u-values** (`build_zeta()`): a *separate* `rng_u = np.random.default_rng(0)`
   generates 5 complex values `1.2·(rng.random(5)-0.5) + 1.2j·(rng.random(5)-0.5)i` — the same
   functional form `zeta_gauge.py:149` uses for its own `ZG-1` polynomial-identity samples.

Labels in `adjacency_eigs_at_k` / `det_coeffs_at_k`: `"Gamma"` for `(0,0,0)`, else
`"k(%.10f,%.10f,%.10f)"` on the three components — reproducible byte-for-byte across runs
because the generator is re-seeded at `0` every invocation (this is what makes the `--check`
diff meaningful: the "random" points are not actually random run-to-run).

## Never hand-edit this file

`bridge_vectors.json` is a build artifact. If a value looks wrong, the fix is upstream (in
`the_run.py`, `srs.py`, or one of the two adapters whose recipes are re-expressed here) or in
the exporter's re-expression of that recipe — never in the JSON itself. `--check` mode exists
specifically to make a hand-edit or a stale export immediately visible (`git diff` after a
`--export` re-run is the fast local signal; the `verify.py` BACKBONE entry is the CI signal).
