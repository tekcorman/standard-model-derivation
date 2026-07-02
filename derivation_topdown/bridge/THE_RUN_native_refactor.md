# the_run.py — the zero-information refactor (governance doc, 2026-06-23 PM)

**Purpose.** Before driving `the_run.py` to *zero information injection* line by line, this doc locks the two
halves so neither is lost in the discipline pass:
- **CONTENT** (A) — the physics/mechanisms/results we built. These must be PRESERVED (every read keeps reproducing
  its value).
- **FORM** (C) — the zero-information standard. Every read must become a CHOSEN-INPUT-FREE invariant of the one
  object, using the native generators (B) that already exist in the repo.

The refactor is the FINAL form of the unification: not "one object, many reads," but **"one object, zero-input
invariant reads."** A read that still carries a choice of mine is an *unfinished derivation*, not a native read.

---

## A. THE CONTENT — physics threads that must survive the refactor (do NOT lose)

The one object: **D = B(srs ⊗ srs‑z) ⊗ ∂_N** — the joint mirror-cover Hashimoto B, dressed by the run ∂_N
(= the walker STEPPING: one NB step = one observation = one tick of N = time). Mechanisms, all reproduced:
1. **Masses = recurrence-under-running** = the dressed resolvent G_NB=(I−uB)⁻¹, u=α₁=ρ^(g−2). The SPECTRUM is the
   masses (whole-spectrum read, no channel pick — the target form).
2. **Dark = the dressed spectrum** (first-girth-return); the three live forms (α₁/h, α₁/(1−α₁), α₁²) = ONE
   expansion. Value wins: **m_b +2.99→+0.22σ, m_t +4.71→−0.95σ** (zero adoption).
3. **Obliques δ_r/δρ/S = cavity-GF reads** of G_NB: g(z)=1/(z−k·f(z)); the **discriminant sign (off/on McKay cut)
   DERIVES the resummed-vs-leading form** (replaced my hand form-selection). δ_r +0.338%, δρ +1.091%, S +0.253%.
4. **Gauge running = the run's ZERO-MODE**; the +4 (b₁=+12/5) = the 4D completion D₄=D₃⊗1+γ_t⊗∂_N (γ_t = the
   even-triple grading; clean split D₄²=D₃²+∂_N²; KO 2→6). **β-CONTENT (12/5,4,4) in hand**; the OPERATOR derivation
   (ζ_{D₄}(0) from D₄) is the one research-level open piece (continuum Dirac-cone + KO-6; lattice heat-kernel = dead end).
5. **δ (generation phase) = the chiral run-phase 2π/√7** (the band-edge ℚ(√−7) root).
6. **Species map = Cl(6) Hamming weight** n∈{0,1,2,3}→ν,d,u,e (the number operator N̂=Σaᵢ†aᵢ eigenvalue — a READ).
7. **Scale = the ONE free axis** (time, G_F-pinned). The EW gauge self-consistency is SCALE-FREE (v cancels: both
   sides ∝ v²) ⇒ the obliques do NOT re-pin the scale; the M_Z/m_W residuals are gauge-internal (the sub-leading S),
   not scale. an internal working note
8. **R-19 de-escalated**: the MSSM-shaped β is the forced content's 4D completion, not an external import.

## B. THE NATIVE GENERATORS — already in the repo (use these; do NOT reinvent)

| native generator | gives (with NO selection) | where |
|---|---|---|
| **Ihara/Bass zeta** `det(I−uB) = (1−u²)²(1−u)(1−2u)(1+u+2u²)³` | the CHANNELS (the factors' roots: Perron h=2, h=1, √−7 shell); poles=spectrum | `dirac_srs_mdl/srs.py:51` ihara_zeta_inv; `zeta_factorization_srs_srsz_2026-06-10.py` |
| **`N_L = ∫_BZ Tr B(k)^L dk`** | the GIRTH (N_L=0 for L<10), the moments | `zeta_channel_dictionary_probe_2026-06-10.py` |
| **cavity GF** `g(z)=1/(z−k·f(z))`, qf²−zf+1=0 | the OBLIQUES (values/flows; the cut derives the form) | `theorem_unified_oblique.md §7.5` |
| **Tr(D_F²) = 24 = α_GUT⁻¹** ; **sin²θ_W = ΣT₃²/ΣQ²** ; **Λ²=6** (max eigval) | gauge boundary as pure traces/moments | `4d_dirac_operator_construction_probe.py`; `sin2_theta_W_derivation.md`; `interaction/explore_i03` |
| **Cl(6) Hamming weight** `N̂=Σaᵢ†aᵢ` | the SPECIES labels (which mode = which fermion) | `theorem_charge_before_color.md §9`; `matter_bridge/explore_m05` |
| **spectral moments / heat-kernel** Tr(Dⁿ) | gauge/gravity/cosmology (a₀,a₂,a₄) | `phase4_2_heat_expansion_sectors_2026-06-11.py` |

## C. THE STANDARD — zero information injection (the bar; the file is currently FAR from it)

**★ THE DEEPEST RULE (user, 2026-06-23 PM — the load-bearing one): NO RANGE, NO BOUND, NO `min`/`first`/
`argmax`/`[i]` — the object emits its COMPLETE structure and the MATHEMATICS forces which feature is used; I
never select it.** A `range(1,14)` smuggles "the answer is the minimum, and it's in 1–14"; a `next(first nonzero)`
smuggles "take the minimum." Both PRESUME. A diagnostic read may *describe* a feature (e.g. "girth = shortest
cycle"); but the moment a feature FEEDS a downstream quantity (girth → α₁), "take the minimum" stops being a
description and becomes a presumption — the feature must be FORCED (e.g. BZ-averaging annihilates every
net-translating walk, so the first nonvanishing term IS the first zero-net cycle *by force of the lattice*, not by
my `min`). **We do not even include the minimum unless the mathematics leaves nothing else.** The mass-spectrum
read therefore emits the WHOLE spectrum with no pick — no "the Perron," no "the shell," no `argmax`, no `[0]`,
no range; the resolvent/renewal forces which eigenvalues carry which masses. (My current `read_girth` VIOLATES
this — `range(1,14)` + `next(min)`; it must become the lowest nonvanishing renewal/zeta coefficient, no bound.)

A read is native iff it contains **none of the following** of mine:
- **ranges/bounds/selections** (`range(...)`, `min`/`first`/`max`/`argmax`/`[i]` → the object emits everything,
  the math forces what's used — see THE DEEPEST RULE above);
- **chosen k-points** (use BZ-integrated / k-invariant functionals — the zeta `det`, `N_L`);
- **selections** (which-eigenvalue → use traces/determinants over the WHOLE operator);
- **constructed operators/directions** (the singlet, the grading, T₃, the C₃ 3-cycle, the cavity branch, the
  copy⊗Weyl⊗dart tensor — these must BE srs's operators, not built by me);
- **typed constants** (`−2` in ρ^(g−2), `p=2`, the EW `½`, `2π`, `c_v=5/12`, the whole gauge β);
- **computational choices** (grid sizes `6`, ranges `range(1,14)`, thresholds `1e-6`);
- **output choices** (the `__main__` prints — what/how to show).

**Every one of these is an unfinished derivation.** "Not native" ⇒ keep working; never flag-and-move-on.

**★ FORCED vs SELECTED (refinement, 2026-06-23 PM — what survives the DEEPEST RULE).** A FORCED invariant —
the Perron–Frobenius dominant mode, the girth (lattice forces the zero prefix), the non-Perron distinct
eigenvalue (mult = generations) — is NATIVE even though a `max`/`first` extracts it, because the mathematics
leaves exactly one answer; the extraction is not a choice. A SELECTED feature — `range(1,14)` (presumes the
answer's location), "use the √3 channel," "this eigenvalue is m_b" — is an injection. Test: *would any other
answer be mathematically admissible?* If no → forced (native). If yes → I'm choosing (unfinished). Applied:
the channel ENERGIES in read/`[4'dark]` are forced (k*=PF Perron, −1=forced non-Perron, √k*=√Perron); the
channel ASSIGNMENT (Perron→down, shell→ν, band-edge→generation) is SELECTED by me — the selection map
(species→channel, forced by chir/color concentration) must be built as a read to retire those labels.

PROGRESS (2026-06-23 PM, line-by-line): ✅ girth (renewal sequence, forced-0 prefix, no range/min) · ✅ ρ
native (k−1)/k · ✅ p_toggle read (darts/edge) · ✅ read_clock inputs read (form = forced MDL disconfirmation)
· ✅ species = Hamming-weight N̂ read · ✅ obliques via cavity-GF (cut derives form) · ✅ **SELECTION MAP =
read_selection** — the [4'dark] hand-labels RETIRED: species n→(channel h, walker L) forced; the d/u split is
COMPUTED off the Cl(0,2) handedness (down-Higgs grade-1 odd ⇒ ω flips ⇒ L=g, Perron-rank1; up conj-Higgs
iσ2 H* even ⇒ no flip ⇒ L=0, Perron-rank2), verified doublet-independent over 2000 cases; darks reproduce
0.980491 (down/m_b) & 0.990245 (up/m_t). Both quarks dress at the SAME Perron h_P=2 (L sets the rank, not a
different root — the agent's "h=1" was the anchor q_NB^0, caught before wiring).
RESIDUAL typed/flagged: the −2 (n_fixed) in U_RUN ; the ν↔chir-7 / e↔chir-5/3 chir-match (A5 import) ; the
down↔H / up↔H̃ coupling (hypercharge, encoded not computed).

★ GENERATIONS (read_generation, 2026-06-24): the per-species 3-mass spectrum is now a NATIVE read of the run
∂_N — moduli 4:2:2 FALL OUT (per-winding PF-Perron at Γ, NOT typed); phase = directed velocity φ=2π/√7
(read_mass's rate) × s; **the stale 2/9 is RETIRED — the Koide phase φ·s=0.2222220 falls out, not hardcoded**;
s = the one free axis. Reproduces the leptons (m_μ/m_e exact, m_τ/m_e +61 ppm). **★ KEY REFRAME (user-directed,
verified): the m_e/m_μ −70 ppm is NOT the observer scale-slice — the scale CANCELS in the ratio (proven). It is
the NEXT-ORDER ∂_N (the winding-dressing asymmetry), an UN-WORKED operator calculation, NOT a free axis. The
generation_phase_delta_irreducible doc conflated scale-slice with running-phase and relabeled an un-computed gap
as "settled." The −70 ppm is in-reach (a defined ∂_N dressing calc); the naive full B(s·axis) dressing OVER-
applies it (4% off) ⇒ needs the correct next-order magnitude.**

GAUGE β (read_gauge_running, 2026-06-24): LAYER 1 now COMPUTED (Dynkin sums over the forced content → MSSM
33/5,1,−3, no typed coeffs). LAYER 2 (the one-loop formula = ζ_{D₄}(0)) research-level. NOTE: removing layer-1
injection does NOT move g_2 (the β was already MSSM); g_2's residual lives in layer 2 (isolated, not the content).

NEXT: the next-order ∂_N winding-dressing (m_e/m_μ −70 ppm, in-reach) ; the chir-match + hypercharge coupling ;
then adjacency_energies, gauge_singlet_projection (argmax→power-iter), the k-grids, the 5/12 + leg counts.

## D. INJECTION INVENTORY + refactor order (the line-by-line target)

Current `the_run.py` injects at every level (audited 2026-06-23 PM). Remove in this order, reproducing each value
as a pure invariant BEFORE the choice is allowed to leave:
1. **k-choices** → BZ-integrated / k-invariant (the zeta `det`, `N_L`, BZ-traces). [read_geometry, read_run,
   read_dirac4_lift, adjacency_energies, read_obliques eval points]
2. **selections** (max/argmax/which-eigenvalue) → traces/determinants. [adjacency_energies max/min,
   gauge_singlet_projection argmax, the read(λ) channel picks → the whole spectrum + Hamming-weight species]
3. **constructed operators** → srs's own operators. [the singlet, the grading, T₃, the C₃ 3-cycle, cover_B,
   the tensor structure]
4. **typed constants** → read off the object. [−2 (n_fixed), p=2, ½ (EW), 2π, c_v=5/12, the β = ζ_{D₄}(0)]
5. **output** → emit invariants, not chosen prints.

**Acceptance test (per the framework's own north-star):** the over-determination — the SAME one object read for
masses (poles), mixings (off-diagonal), gauge (moments), obliques (cavity flows) — agrees with zero new input. A
choice that breaks this agreement is wrong; the forced reads hold it.

---
*This doc governs the refactor of `derivation_topdown/bridge/the_run.py`. Content (A) is preserved; form (C) is
enforced using generators (B); progress is measured by injections removed (D). Verify every claim against srs.py /
the cited theorems — never trust prose, including this doc.*
