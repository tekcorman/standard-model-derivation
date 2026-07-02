# PRE-REGISTRATION — Family-D c_F fermion-leg residue (W1)

**Committed before any residue number is evaluated.** Binding once committed.
Anchor: W1 of the pre-publication cleanup; c_F = −α₁²/(N·k*) = −α₁²/12 is the
one weak point of Family D (c_H structurally derived via Route H; δρ sound).
Goal: determine c_F by computing the per-fermion-leg residue in the SAME
B_NB(srs) resolvent formalism that already closed δ_r, with channel,
projector, normalization, and decision rule frozen here. Strengthen or kill;
both accepted. No grading branch.

## A. Frozen inputs (zero freedom)

1. **Operator:** `B_NB(srs)` on the 2|E|=12 directed-edge primitive cell,
   reused verbatim from `proofs/foundations/nb_two_vertex_generations_probe.py`
   (`directed_edges`, `nb_operator`, `rev_index`). Not rebuilt.
2. **Resolvent / projector:** `G_NB(u)=(I−uB_NB)⁻¹=Σ_λ P_λ/(1−uλ)`,
   `P_λ=|r_λ⟩⟨l_λ|/⟨l_λ|r_λ⟩`. Perron `P_P=|1⟩⟨1|/⟨1|1⟩` (proven
   `B_NB·1=(k*−1)·1`); Ramanujan `h_P=(√3+i√5)/2`.
3. **Fermion-leg coupling vector:** the CAR single-directed-edge unit basis
   vector `|e_a⟩∈ℂ¹²`, forced by `docs/theorems/theorem_car_local_jordan_wigner.md`
   §1 (one CAR mode ↔ one directed edge mode at the vertex). Not tunable.
4. **JW sign:** −1 (proven F3/F4 closed-fermion-line sign, CAR thm §7). Reused.
5. **Joint walker amplitude:** α₁² (Route H, accepted structurally derived).

## B. Frozen structural arguments (committed; the entire weakness, reduced to two)

**B1 — Channel.** §3.1 rule of `theorem_unified_oblique.md`:
species-conserving / C₃-diagonal → Perron `P_P` (λ_P=k*−1);
species-changing / off-diagonal → `h_P`. The charged-lepton Yukawa
`y_τ φ ψ̄ψ` has a **generation-diagonal** fermion bilinear (τ mass
eigenstate: same generation in and out). A generation-diagonal bilinear is
species-conserving ⇒ **the fermion leg projects onto the Perron channel
P_P.** (Committed. The h_P reading would apply only to a generation-OFF-diagonal
coupling, which the charged-lepton Yukawa is not. If, on inspection, the
fermion leg is provably neither cleanly diagonal nor off-diagonal, that
triggers the channel_select sub-clause in C.)

**B2 — Normalization.** The per-leg residue weight is the IDENTICAL formula
used for the proven δ_r coefficient,
`unified_oblique_one_resolvent_2026-05-16.py:160`:

```
w = ⟨V̂ | P_channel | V̂⟩ / (2|E|)
```

with `V̂` the **unit-normalized** coupling vector and `P_channel` the B1
projector. For δ_r the gauge-singlet unit vector is `ŝ=1/√(2|E|)·1`, giving
`⟨ŝ|P_P|ŝ⟩=1` ⇒ `c_S=1/(2|E|)=1/12`. For the fermion leg the CAR structure
fixes `V̂_F=|e_a⟩` (a unit standard basis vector). **The same formula is
applied verbatim, no fermion-specific modification.** (Committed: we do NOT
introduce a separate fermion normalization; the residue convention is
whatever δ_r uses, applied identically.)

## C. Decision rule (thresholds fixed now; exact arithmetic in K=ℚ(√2,√3,√5))

`c_F = (−1)·α₁²·w_F`, `w_F = ⟨e_a|P_channel|e_a⟩/(2|E|)` per B1+B2.
"Match" = **exact equality**, not "within X%".

| Computed `w_F` | Verdict | Action |
|---|---|---|
| **exactly 1/12** | **STRENGTHEN** | c_F derived by the δ_r-class Perron residue; delete the fake F-1/F-2 "two routes"; Family-D fermion sector graduates; m_τ −0.17σ stands as a derivation |
| **any other exact value** (1/144, 1/9, h_P-value, channel_select mix, …) | **REJECT** | Family-D fermion sector falsified by its own claimed mechanism; recompute m_τ/m_e/m_μ with the residue-true `w_F`; report honest σ_PDG; document the clean negative |

Channel_select sub-clause (only if B1 is provably ambiguous): `w_F` =
spectral-measure-weighted superposition of the available channels
(waterfilling, **not** argmin); the resulting single exact `w_F` is then run
through the table above.

## D. Anti-gaming clauses (what makes it binding)

1. This file is git-committed **before** the residue is numerically evaluated.
2. m_τ / m_e / m_μ values are **not consulted** until after `w_F` and the
   verdict are determined.
3. `|e_a⟩` must give an identical `w_F` for all 12 directed-edge choices
   (srs edge-regularity). If not, the object is ill-defined → **REJECT**.
4. No denominator menu — `w_F` is whatever the one pre-registered residue
   gives; we never enumerate {9,12,144,…} and pick.
5. B_NB reused verbatim from the existing probe — no freshly-built graph.
6. δ_r reproduction sanity check: the probe must reproduce `c_S=1/12` for the
   gauge-singlet via the same code path before computing `w_F` (proves the
   formula path is the δ_r one, not a re-derivation).

## E. Honest pre-analysis (does NOT relax the registration)

δ_r's formula has `⟨ŝ|P_P|ŝ⟩=1` because the gauge-singlet **is** the unit
Perron direction. A single-edge vector has `⟨e_a|P_P|e_a⟩=|⟨e_a|1⟩|²/⟨1|1⟩
=1/(2|E|)` (its squared overlap with the uniform Perron eigenvector). The
same formula then gives `w_F = (1/(2|E|))/(2|E|) = 1/(2|E|)² = 1/144 ≠ 1/12`.
**On this reading the prereg REJECTS the Family-D c_F=−α₁²/12 claim.** The
STRENGTHEN outcome (1/12) would require the fermion leg to skip the explicit
`/(2|E|)` that δ_r's formula carries — a fermion-specific normalization that
is derived nowhere and that B2 explicitly does NOT grant. I expect REJECT.
We commit and compute regardless; the mechanism decides, and a clean,
self-found negative is an accepted, valuable outcome.

— committed 2026-05-18, prior to evaluation.
