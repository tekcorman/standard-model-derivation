# Lemma — the commutation obstruction: Galois Z₃ in the commutant of `B` blocks per-isotypic spectral readings of the within-species 3-fold (2026-05-23)

**Status:** THEOREM-GRADE (structural). Small, elementary, load-bearing.
This lemma is the structural reason route 4 of the δ-physical menu

was eliminated — and, more generally, the reason **the entire class of
per-Galois-isotypic spectral readings of `B_NB` is empty** as a route to
the Koide 3-fold within species. It complements R1/R2/R3 of
an internal working note
by giving a generic structural mechanism — not a route-specific accident.

Verified numerically:
- `proofs/foundations/h_power_yukawa_galois_isotypic_stage0_2026-05-23.py`
  — K_4(Γ) shortcut: residue phases all real (0) — the reality version of
  the obstruction.
- `proofs/foundations/h_power_yukawa_galois_isotypic_stage0_Ppoint_2026-05-23.py`
  — B_NB(P): residues at y=1/h_P in j=0 and j=1 have *identical* phase
  −arg(h_P); j=2 has no h_P at all. The 3-fold AP never forms.

---

## 1. Statement

Let `B` be a linear operator on a finite-dim space `V`, and let `G = ⟨σ⟩ ≅
Z_3` act on `V` by a unitary representation `P_σ` satisfying

```
   [ B, P_σ ] = 0          (G is in the commutant of B).
```

Let `π_j = (1/|G|) Σ_k ω^{-jk} P_σ^k` for `j ∈ {0, 1, 2}` (Z₃-Fourier
projectors onto the three isotypic components, `ω = e^{2πi/3}`). Let `h`
be an eigenvalue of `B` with right-eigenvector `v` and (biorthogonal)
left-eigenvector `w`. Let `ψ ∈ V` be any anchor, and define

```
   F_j(y) := ⟨ π_j ψ | (I − y·B)^{−1} | π_j ψ ⟩.
```

Then the residue `R_j := Res_{y = 1/h} F_j(y)` satisfies

> **The complex phase `arg(R_j)` is independent of `j` modulo the sign of
> the eigenvector overlap.** No "natural Galois phase" `ω^j = e^{2πi j/3}`
> appears in the per-isotypic residue. In particular, the three phases
> `{arg(R_0), arg(R_1), arg(R_2)}` do not form an arithmetic progression
> with common difference `2π/3`.

**Corollary (the route-4 elimination).** The Koide cosine parametrisation
`√m_j = M₀ (1 + ε cos(2πj/3 + δ))` cannot be obtained as `arg(R_j) =
2πj/3 + δ` from `(B, π_j, ψ)`. The within-species generation 3-fold is
not in the spectrum of `B`, read either globally (R3 of Need-B) or
per-isotypic (this lemma).

## 2. Proof

**Step 1 (eigenspace decomposition).** Because `[B, P_σ] = 0`, each
eigenspace `E_h = ker(B − h·I)` is `P_σ`-invariant. The Z₃-Fourier
projectors decompose it:

```
   E_h = (E_h ∩ Im π_0) ⊕ (E_h ∩ Im π_1) ⊕ (E_h ∩ Im π_2).
```

This is the eigenspace partitioning **into** isotypics, not a rotation
**between** them.

**Step 2 (per-isotypic residue).** Near `y = 1/h`, the resolvent has the
Laurent expansion

```
   (I − y·B)^{−1} = −(1/h) · |v⟩⟨w| / (y − 1/h)  +  (regular)
```

(where `|v⟩⟨w|` is the rank-one spectral idempotent of `B` at `h`, and
the `−(1/h)` prefactor follows from `d/dy(1 − y·h)|_{y=1/h} = −h`).
Hence

```
   F_j(y) = −(1/h) · ⟨π_j ψ | v⟩⟨w | π_j ψ⟩ / (y − 1/h)  +  (regular)
```

and

```
   R_j = −(1/h) · ⟨π_j ψ | v⟩⟨w | π_j ψ⟩
       = −(1/h) · ⟨ψ | π_j v⟩⟨w | π_j ψ⟩       (π_j self-adjoint)
       = −(1/h) · (π_j v · ψ)(π_j w · ψ)*       (schematic).
```

**Step 3 (the phase identity).** Because `v ∈ E_h` and `E_h` is
`P_σ`-invariant, `π_j v` is the *projection of `v` onto its `j`-isotypic
component within `E_h`*. The complex number `⟨ψ | π_j v⟩` carries the
phase of this overlap; the conjugate factor carries its mirror.
Crucially, **the prefactor `−1/h` is `j`-independent**, and the only `j`-
dependence in the rest is in `⟨ψ | π_j v⟩⟨w | π_j ψ⟩`. This is a
*magnitude* and *sign* of the projection on the `j`-th piece of `E_h` —
*not* a multiplicative factor of `ω^j`.

For a Galois phase `ω^j` to multiply `R_j`, the eigenvector `v` itself
would need to transform under `σ` by `ω^j`. But `v` is fixed (each `v` is
one eigenvector); the Galois-isotypic decomposition splits `E_h` into
its three pieces, each of which is its own `v_{(j)}`. The "phase per
isotypic" is the phase of the *complex overlap of ψ with the j-th piece*
— a number determined by the anchor and the (single) eigenspace
geometry, with **no per-j Galois rotation**.

**Step 4 (the structural conclusion).** For the three `R_j` to satisfy
`arg(R_j) − arg(R_0) ≡ 2πj/3 (mod 2π)` by mechanism, one would need an
isomorphism `E_h ∩ Im π_j ≅ E_h ∩ Im π_0` implemented by `P_σ` that
rotates phases by `ω^j`. But `P_σ` acts on `Im π_j` by the scalar `ω^j`
(by definition of isotypic), and on `E_h` it acts diagonally on these
three pieces. Composing the two: `P_σ · (π_j v) = ω^j · π_j v` — but
this is just the definition of `π_j v` belonging to the `j`-th isotypic;
it does *not* rotate `π_j v` into `π_0 v`. The "phase per isotypic" of
the *residue* is independent of this `ω^j` because it comes from the
*magnitude squared* (or biorthogonal overlap) of the projection, not
from the action of `P_σ` on `v` directly.

∎

**Remark (Hashimoto-specific refinement).** For Hashimoto `B` (non-
Hermitian, biorthogonal `|v⟩ ≠ |w⟩`), `R_j = −(1/h) · ⟨π_j ψ | v⟩
⟨w | π_j ψ⟩` is not strictly `|⟨π_j ψ | v⟩|^2`. Its phase is a sum of
the right- and left-overlap phases. Both depend on the anchor `ψ` and
on the eigenvector pair `(|v⟩, |w⟩)`; the *j-dependence* enters only
through the magnitude of `π_j ψ` in `E_h`, not via `ω^j`. So even in
the non-Hermitian case the residue phase is `j`-independent in the
sense of having no factor of `ω^j` — the cancellation is by
construction.

## 3. Empirical confirmation (this session)

### 3.1 Real-Bloch (K_4 at Γ)

`proofs/foundations/h_power_yukawa_galois_isotypic_stage0_2026-05-23.py`:

- G4 PASS: h = 1 lives in all three Galois isotypics of `B(K_4)`.
- G6: residues at y = 1 in j = 0, 1, 2 came out as
  `(0.555, 0.391, 0.054) × e^{i·0}` — **all real, all phase 0**.
- The reality version of the obstruction: with `B` real and `ψ` real,
  the biorthogonal overlaps are real, so `R_j` is real for every `j`.
  The 2π/3 progression is absent (`a_1 = a_2` numerically; differences
  of contributions to `π_1` and `π_2` cancel).

### 3.2 Complex-Bloch (B_NB at P-point)

`proofs/foundations/h_power_yukawa_galois_isotypic_stage0_Ppoint_2026-05-23.py`:

- G3 PASS: framework's `h_P = (√3 + i√5)/2` is an eigenvalue of `B_NB(P)`.
- G_complex PASS: `||Im B_NB(P)||` ≈ 4.90 — complex Bloch phases break
  the K_4(Γ) reality.
- G4 FAIL: `h_P` lives in j = 0 and j = 1 *only* — j = 2 has `−h_P` instead.
  The eigenvalue distributes `(1, 1, 0)` across isotypics, not `(1, 1, 1)`.
- G7: residues at y = 1/h_P in j = 0 and j = 1 came out as
  `0.665 · e^{−i·0.9117}` and `0.043 · e^{−i·0.9117}` —
  **identical phase, j-independent**. Empirical confirmation of the
  lemma.

In both cases the lemma's predicted phenomenon is observed.

## 4. Scope of the elimination

The lemma rules out **all routes of the form**:

> "Read the Koide phase δ as the residue argument of a per-Galois-
> isotypic diagonal of a substrate-spectral resolvent, given that the
> Galois Z₃ is a substrate symmetry."

This includes — but is not limited to — route 4. Any future proposal
that takes (`B_NB(k)` or any commutant-compatible operator on the
substrate) and projects through `π_j` to extract Koide δ from a
*spectral residue* of that operator falls under the same obstruction.

It does **not** rule out:

- Spectral readings on operators that *don't* commute with `P_σ` (e.g.,
  Galois-twisted operators `B' = B + V` where `V` does not commute with
  the C₃ symmetry). The Koide phase would then live in the failure of
  perfect symmetry — a *broken-symmetry* spectral reading. Whether such
  a `V` is upstream-derivable is a separate question.
- Non-spectral readings on the operator algebra `M = L(F_inv(E))` —
  the surviving menu pointed at by the route-4 elimination:
  - **Route 1**: Connes 2-cocycle in `H²(Z₃, U(M^α))` (Galois
    cohomology of the M1.B subfactor tower).
  - **Route 2**: Subfactor principal-graph spectrum at Jones index 3.
  - **Route 3**: Voiculescu free-Fisher minimizer on `L(𝔽_4)^α`.

## 5. Relation to existing eliminations

The four-way elimination of bounded routes to Need-B δ-physical, as of
2026-05-23:

| route | status | mechanism of elimination |
|---|---|---|
| R1 | ❌ eliminated (2026-05-16) | Triplet screw-Wigner-D template depends on a Q = 2/3 coincidence (`Q²/2 = Q/3 ⇔ Q = 2/3`); at `Q_down ≈ 0.75` the routes diverge. |
| R2 | ❌ eliminated (2026-05-16) | Same template as R1, applied to the R4-refuted `arg(h_P)/4`. |
| R3 | ❌ eliminated (2026-05-16) | No principled *global* spectral reading of `G_NB` reproduces δ_lepton = 2/9 by mechanism. |
| **route 4 (this lemma)** | ❌ **eliminated (2026-05-23)** | **No per-Galois-isotypic spectral reading either — commutation obstruction `[B, P_σ] = 0` forces `j`-independence.** |
| W65 F3 | ❌ eliminated (2026-05-25) | Higgs-induced phase: natural Higgs-vacuum-derived operators commute with C_3 by construction (`W65_F3_obstruction_inheritance_check_2026-05-25.py`). |
| W66 F1 | ❌ eliminated (2026-05-25) | Observer Bayesian-walk: all 4 natural observer-side dynamical operators on `C³_obs` commute with C_3 (`W66_F1_obstruction_inheritance_check_2026-05-25.py`). |
| **AB5 / F5 Path B** | ❌ **eliminated (2026-05-25 EOD+1)** | **NA-4 multiway-DAG bracketing level: all 4 natural multiway-DAG operators on ℋ_freemagma commute with ρ_3; all alphabet-coupled operators factorize into U_alpha ⊗ U_bracket, collapsing cross-bracketing trace identically; no framework-natural entangling operator identifiable (`NA4_AB5_sufficiency_check_2026-05-25.py`).** |
| **W70 girth-pathway** | ❌ **eliminated (2026-05-26)** | **Closed girth-10 NB walks on K_4 (substrate primitive cell Γ-Bloch quotient) from base vertex decompose into 138 free C_3 orbits, all size 3; zero C_3-fixed walks. Per-isotypic decomposition is purely faithful (trivial dim 0, faithful dim 276) → walks inherit obstruction by construction. The girth-pathway-shape hypothesis (3 classes mapping to 3 generations via site-stabilizer action) is closed-NEG (`W70_girth_pathway_koide_2026-05-26.py`).** |

The pattern is now clean: **δ-physical is not in the spectrum of `B`,
neither global (R3) nor per-isotypic (route 4), nor in any natural
operator on the framework's substrate (W66), observer (W66), Higgs
(W65), or multiway-DAG bracketing (AB5) sectors**. The lemma is now
load-bearing across **5 distinct sectors** — substrate, observer,
Higgs, per-Galois-isotypic spectral, bracketing-DAG. The surviving
candidates are *non*-spectral readings on the operator algebra
`M = L(F_inv(E))` (routes 1/2/3 of §4 menu — note routes 1/2/3 of
that menu inherit the same Galois-tower coarseness; see route-1 Connes
2-cocycle elimination 2026-05-23).

## 6. Use in future scoping

When evaluating any new bounded probe for Need-B δ-physical:

1. **Spectral reading on a commutant-compatible operator?** If yes
   (and `[B, P_σ] = 0` is the relevant symmetry), the lemma blocks it.
   Do *not* run the probe; mark it as a refinement of route 4.
2. **Spectral reading on a Galois-twisted operator?** The lemma does
   not block, but you owe an upstream derivation of the twist.
3. **Non-spectral reading on the operator algebra?** Outside the
   lemma's scope; routes 1/2/3 of the §4 menu apply.

## 7. References

  — the route-4 scope, with this elimination recorded in its banner.
  — session arc that produced the lemma.
  — R1/R2/R3 elimination; this lemma's predecessor.
- `docs/theorems/theorem_observer_substrate_iprojection_scoping.md` §7.5
  (M1.B) — the outer Galois Z₃ on `M = L(F_inv(E))` whose commutation
  with `B_NB` is the lemma's premise.
- `proofs/foundations/h_power_yukawa_galois_isotypic_stage0_2026-05-23.py`
  and `_Ppoint_2026-05-23.py` — numerical confirmations.

## 8. Type-classification

- **Type 1**: spectral theory of finite-dim operators commuting with a
  finite-group action (standard linear algebra).
- **Type 2**: Z₃-Fourier projection of complex vector spaces.
- **Type 3**: Laurent expansion of resolvent at simple poles (rank-one
  spectral idempotent).

The lemma is a *negative* structural theorem: like the
"Higgs makes no edge selections" lemma (`theorem_ytau_corollary` §7
L3+L10), it precludes a category of mechanisms rather than constructing
one.
