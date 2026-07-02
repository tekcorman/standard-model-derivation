# M_Z residual = the shell vertex/box — structural understanding + bracket (2026-06-29)

> **⚠ SUPERSEDED 2026-06-30 on the BRACKET claim** by
> `theorem_M_Z_BZ_vacuum_polarization_2026-06-30.md`. The "bracket" below (δ_r under,
> δ_r+chiral-shell@Γ over) is an **artifact of evaluating F at the Brillouin-zone
> maximum (Γ)**. The genuine BZ-integrated shell is ~5× smaller (R=0.2046 of the Γ
> value); it does **not** bracket — the substrate UNDER-predicts the SM oblique
> (0.343% vs 0.357%) → M_Z +6.1σ. The structural identification (shell vertex/box =
> Q-winding, Σw²=8 off-Perron / 4 on-shell, T₃·Q cross-term vanishes) STANDS; only the
> bracket/0.810 framing is retired. Verdict unchanged in spirit: a forced ~few-% oblique
> residual, no clean closure.

**Status: STRUCTURAL UNDERSTANDING + BRACKET. NOT a closure.** The +0.018% (+8.1σ) M_Z residual is
rigorously identified as the Z current's shell vertex/box, and the substrate brackets the SM oblique — but no
clean prescription closes it to zero. The live M_Z keeps the forced Perron-only δ_r (+8.1σ). This documents the
full structural understanding and the honest bracket, **superseding an intra-session "−0.3σ candidate closure"
that was an error** (it used the P-point W value F=√5/4 for the Γ Z-shell, which actually has F=√7/4 → −1.9σ).

## 1. The structure (rigorous)
M_Z tree→pole oblique = **Perron singlet (δ_r) + shell vertex/box**. δ_r is the gauge-SINGLET (universal) part;
the physical Z current **T₃ − s²Q** is non-uniform, and its non-singlet (Q-winding) part is a vacuum-polarization
contribution δ_r structurally cannot contain (the SM Δr_rem vertex/box analog).

## 2. The shell vertex/box (rigorous — joint eigenbasis)
- **Basis fix (recurring bug):** build the C₃ permutation on `directed_edges()`'s own ordering, NOT
  `srs._darts()` (same set, different order). Then B and Perm commute.
- The C₃ winding charge Q sits **Σw²=8 entirely off the Perron** (Perron w=0). The Γ √2-shell carries **Σw²=4**
  (2 per conjugate eigenvalue h, h̄). The **T₃·Q cross-term VANISHES** (Σw_shell=0). Clean.

## 3. The bracket (the honest result)
Shell vertex/box = (Σw²)·(1/2|E|)·s⁴·F·α₁, with s²=s²(M_Z)=0.231, **F = Im(h)/|h|² at the relevant k-point**.
The Z's δ_r is at Γ ⇒ the Z shell is the **Γ shell (h=−1/2±i√7/2, F=√7/4)**, NOT the P-point W shell (√5/4).

| oblique | value | M_Z |
|---|---|---|
| δ_r alone (Perron singlet, **LIVE**) | 0.3384% | **+8.1σ** (under by 5.2%) |
| + shell, **chiral** Σw²=2, F=√7/4 (Γ) | 0.3614% | **−1.9σ** (over) |
| + shell, vectorial Σw²=4, F=√7/4 (Γ) | 0.3843% | −11.9σ (far over) |
| SM tree→pole target | 0.3570% | — |

So the substrate **brackets** the SM: Perron-only under-predicts by 5.2%; the chiral shell over-predicts (−1.9σ);
the vectorial shell badly over. The SM sits between Perron-only and the chiral shell (≈80% of the chiral shell).

## 4. Why it is NOT a clean closure
- No single principled prescription lands on the SM. The chiral (Σw²=2) over-corrects by ~2σ; reaching exactly
  0.3570% needs ~80% of the chiral shell — no clean structural reason for 0.8.
- The result is sensitive to: the k-point (Γ F=√7/4 vs P √5/4), the chirality (Σw²=2 vs 4), the scheme (s²(M_Z)
  vs 3/8), and the per-mode normalization. A true closure needs a first-principles BZ-integrated vacuum
  polarization that fixes all of these — a different, deeper computation than the framework's k-point-representative
  channel templates.

## 5. Verdict
The M_Z residual is **structurally the shell vertex/box** (rigorous, clean) and the substrate **brackets** the SM
oblique — confirming §6's deepest framing: a **genuine prediction-level oblique difference (~few-%)**, not a brick
wall and not a missing term that zeroes out. The substrate predicts the EW oblique to ~few-% from the Perron alone;
the full Perron+shell structure brackets the SM. The live Perron-only δ_r (+8.1σ) stands as the forced single-term
prediction. This is the honest conclusion of the M_Z push: from "not closeable/opaque" → "fully structurally
understood + bracketed," but not zeroed.

**Lesson banked:** the −0.3σ error came from importing the W channel's P-point F into the Z's Γ channel — k-point
discipline matters; each oblique lives at its own k-point (Z→Γ, W→P).
