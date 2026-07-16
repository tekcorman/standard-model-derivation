# Theorem — two of the three generation "homes" are ONE forced object; the third is consistent-pending-a-datum

**Date:** 2026-07-15 · **Station:** GEN-HOMES (freeze
internal research notes) · **Grade:** theorem (sub-target 1),
adjudicated-open (sub-targets 2, 3).
**Receipts:** implementation pass internal research notes; independent
verification internal research notes; runnable
verification `proofs/foundations/gen_homes_reconciliation_check_2026-07-15.py` (19/19 PASS).
Supersedes the prose `[RESOLVED 2026-07-01]` claim in
internal research notes:63-67` (which was partly false — see §3).

---

## Statement (sub-target 1 — the theorem)

The two generation "homes"
- **(i)** the three C₃-Fourier isotypes (windings) of the 12-dart Hashimoto run — eigenspaces of
  `Pc3[t] = (1/3)Σ_m ω^{−tm} Pperm^m`, t=0,1,2, on ℂ¹² (`derive_generation_spectrum.py:48-58`);
- **(ii)** the m=−1,0,+1 bands of the λ=−1 spin-1 Weyl cone — the λ=−1 eigenspace of the K₄
  adjacency `A(Γ)` (`the_run.py:read_matter_row`);

**are the SAME object, forced, not fitted.** The load-bearing algebraic fact: K₄'s adjacency is
`J − I`, so its λ=−1 eigenspace is exactly the sum-zero subspace of ℂ⁴ = **ρ₃, the honest A4
3-irrep** (spectrum {3,−1,−1,−1}; the λ=−1 eigenspace verified equal to the sum-zero subspace to
≤1e-13). Home (ii)'s carrier is therefore *literally* ρ₃. Home (i)'s C₃-isotype grading, carried
to ρ₃ by the induced-A4 intertwiner (`induced_A4_action_lemma_2026-07-14.md`, `S_{:,i}=φᵢ[:,d0]`),
is exactly the C₃-eigengrading of the deck element σ acting on ρ₃:

> **isotype `t` (eigenvalue-`ω^t` space of `Pperm`) ↦ the `ω^t`-eigenspace of ρ₃(σ)**, for t=0,1,2.

Verified numbers: `M(σ)=S⁻¹ρ₃(σ)S` to 6.7e-16, `S†S=I/12` to 5.6e-17, the isotype↔eigenspace
correspondence to 8.1e-16.

## Why it is FORCED, not a basis fit (the §2 guard, adjudicated by the verification)

An abstract order-3 isomorphism between two 3-dim reps with eigenvalues {1,ω,ω²} is automatic and
therefore **vacuous** (BOOTCAMP §5). The station survives that guard by the following distinction,
established by re-running the reconciliation on **three mutually unrelated bases of ρ₃** (QR-style,
explicit Gram-Schmidt, random-U(3)-rotated):

- The explicit matrices **S and M(σ) are gauge-dependent** — they change with the chosen basis of
  ρ₃. They are *coordinate exhibitions* of the map, not invariants. (This corrects the
  implementation pass's phrase "no free U(3)/GL choice entered anywhere," which overstates S's status.)
- The **reconciliation claim itself is gauge-invariant**: the correspondence "winding-isotype `t`
  ↔ `ω^t`-eigenspace of ρ₃(σ)" holds in every basis, because it follows from equivariance alone
  (the intertwiner conjugates the C₃-action; eigenspaces of order-3 elements are matched by
  eigenvalue, and the eigenvalue labels are gauge-invariant). It is pinned by BOTH constructions —
  the basepoint-dart evaluation is canonical on the (i) side, the λ=−1 eigenspace is canonical on
  the (ii) side — with no free choice.
- **Rejection control** (the W↔A4 pattern): the correct isotype↔eigenvalue pairing beats the two
  wrong cyclic assignments and a random non-isomorphic order-3 control by, independently across the
  three bases, margins **4.0e14 / 1.5e15 / 7.0e14** (implementation pass: 2.04e15) — 9+ orders past the
  ≥1e5 bar. The match is not a coincidence of alignment.

**Conclusion (i)≡(ii):** the C₃-windings and the spin-1/λ=−1 bands are two readings of one A4
3-irrep, related by a construction-pinned, gauge-invariant correspondence. **VERDICT 1-A, sealed.**

---

## Sub-target 2 — (i)↔(iii), the observer factor ℂ³_gen: CONSISTENT, NEEDS A DATUM (2-C)

The claim that ℂ³_gen's generation-Z₃ is the SAME C₃ as the substrate deck screw (via "M1.B's
Galois tower") is **consistent but not construction-forced**:

- **Provenance corrected:** M1.B is NOT prose-only (the freeze's own §0/T2.0 premise was wrong, and
  so was the `[RESOLVED 2026-07-01]` doc-lag note). Three real scripts exist, committed 2026-05-28:
  `proofs/foundations/m1b_observer_substrate_iprojection_attempt.py`, `m1b_c_basis_match.py`,
  `m1b_d_iprojection_structural_map.py`.
- **But the identification is not forced:** `M1.B.b` is genuinely substrate-grounded (a real outer
  automorphism, a Jones-index-3 subfactor). `M1.B.c` and `M1.B.d` run on **admitted finite-dim toy
  models** that trivialize the real infinite-dim operator algebra and reproduce only the vacuous
  abstract order-3-unitary conjugacy (R3's own L2). Machine-confirmed: any random order-3 unitary
  is U(3)-conjugate to σ_shift (residual 9.1e-16, non-discriminating), and the conjugating matrix
  carries a genuine unresolved U(1)² freedom. The residual S₃ label ambiguity is broken only by the
  framework's **external Koide/mass-ordering datum** — which was correctly **never read** in this
  station (goal-seek guard honored).

**Conclusion (i)↔(iii):** consistent up to U(3); the basis-pinning is an EXTERNAL input, named. This
is not a defect — it is the honest location of the missing read. **VERDICT 2-C.** It routes
directly into the II.4 ppm identification wall: the observer-factor basis-pinning IS the
isotype↔generation identification the ppm miss is missing, now stated precisely as "needs datum X"
rather than as a vague soft spot.

---

## Sub-target 3 — the documentation contradiction: 3-B

`framework_architecture.md` literally carries the "orthogonal" framing — at **line 70** ("separate
tensor factor orthogonal to the gauge rep factor") **and line 144** (a second, independent
occurrence). The doc-lag note's assertion that framework_architecture "carries no 'orthogonal'
framing (checked)" is a **factual mis-quote**. Per the freeze's routing rule, and consistent with
2-C: "orthogonal" stands as the current status until the external datum is supplied. The doc-lag
note's §4 `[RESOLVED 2026-07-01]` entry is corrected accordingly (see the note, struck-and-kept).

---

## Net effect on the generation-mechanism soft spot

The 2026-06-29 "biggest conceptual soft spot" (THREE unreconciled homes) is now:
- **(i) ≡ (ii)** — ONE forced object, theorem-grade, with an explicit gauge-invariant
  correspondence. Half the soft spot CLOSES.
- **(iii)** — consistent with (i)/(ii) up to U(3); its identification with them is an external
  datum, precisely named. The remaining edge is no longer "three inconsistent homes"; it is "one
  forced two-reading object plus a consistent third whose basis-pinning is the ppm-wall datum."

Nothing here reads, fits, or references any mass/ppm/mixing value (top-down law honored). The one
free datum remains the run-endpoint s (no-go theorem; no second datum introduced).

## Regression anchor / verify wiring

The durable regression artifact is `proofs/foundations/gen_homes_reconciliation_check_2026-07-15.py`
(19/19 PASS). It anchors the gauge-INVARIANT correspondence and the rejection-control margins — NOT
a specific S (which is gauge-dependent and must not be anchored as if canonical). verify.py wiring
is QUEUED for the next L9 hygiene batch (thread-capped, quiet box). `the_run.py` is deliberately NOT
modified — Layer-1 spectrum stays untouched; the reconciliation lives as a theorem + check, not a
new spectrum read.
