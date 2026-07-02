# The unified fermion mass operator — M_persistence

> **UPDATE 2026-05-21 (W47–W54).** The "Need-D-3 (dynamics tier)" conditional
> below was worked over the W47–W54 arc. Net: the
> Need-D-3 *obstruction* is dissolved (W49 — it was a symmetric-phase artifact),
> and m_ν₁ = 0 — the operator's kernel — is DERIVED (W44/W45). But the CKM
> fold-in (mixings into M_persistence) is NOT closed: the W54 confrontation is
> an honest negative. The 12-mass spectrum claims here stand; the CKM is open,
> with a corrected 4-probe build path in an internal note.
>
> **UPDATE 2026-05-21 (Need-D-3 closed).** The "Need-D-3 (dynamics tier)"
> conditional is now discharged. `theorem_selection_map_2026-05-21.md` derived
> the species→walker-type selection map (forced bijection), and
> `theorem_updown_split_conjugate_higgs_2026-05-21.md` closed its last entry —
> the up/down split (the up-type couples to the even-grade conjugate Higgs ⇒
> cannot flip handedness ⇒ walk length L=0; down-type odd-grade ⇒ L=g). The
> dynamics-tier conditional below should be read as discharged at
> THEOREM-GRADE-STRUCTURAL; the absolute-scale anchors remain the open tier.

**Date:** 2026-05-21
**Status:** SYNTHESIS-GRADE. Assembles the framework's fermion-mass results
into a single operator. The *operator framing* and the *kernel identification*
are the new content; the 12 channel values are individually theorem-grade or
theorem-grade-conditional (cited). The whole operator inherits exactly two open
conditionals — Need-D-3 (dynamics tier) and the absolute scale anchors —
explicitly tiered below.
**Companion probe:** `proofs/foundations/W46_mass_operator_assembly_2026-05-21.py` (7/7 gates PASS) — assembles M_persistence as an explicit 12×12 block operator and verifies block-diagonality, the dim-1 kernel = ν₁, the shape∘dynamics factorisation, and the trivial-holonomy kernel criterion.
**Supersedes:** an internal working note (2026-04-17) —
which named the target (a 3×3 Hermitian M per species) and whose attempts
collapsed (they assumed C₃-on-generation, B6-forbidden). This document assembles
the operator the framework has since actually built.
**Reframes:** `theorem_yukawa_master_theory_synthesis_2026-05-20.md` as the
*off-shell content* of the operator stated here.

---

## 1. Statement

> **The fermion mass operator.** Every Standard-Model fermion mass is an
> eigenvalue of one operator,
>
>   **M_persistence = the holonomy of a self-sustaining left↔right chirality
>   oscillation on the srs↔srs-z double cover.**
>
> A fermion is *massive* iff its substrate walker, completing its
> self-sustaining circuit, returns to the *opposite* mirror sheet — a
> non-trivial holonomy — and must therefore oscillate to persist. A fermion is
> *massless* iff that holonomy is trivial: it never flips, never oscillates,
> and lies in **ker M_persistence**.

The 12 SM fermion masses are the spectrum of M_persistence; the one massless
fermion (the lightest neutrino) is its kernel.

## 2. The central identity — three views of one operator

The framework's "mass = persistent multiway structure" (axiom A5(a): the
substrate's spectral gap at criticality *is* physical fermion mass) and the
textbook chirality picture of the Higgs mechanism are **not two facts to
reconcile — they are one operator seen from three sides:**

| View | What it says | Domain |
|---|---|---|
| **Persistence** | a massive particle is a pattern that *survives* on the substrate — waterfilling-retained, self-sustaining | time |
| **Chirality flip** | a massive particle *survives by oscillating* L↔R; the flip is the Higgs interaction, the flip-rate is the mass | mechanism |
| **Holonomy** | the oscillation is a walker circuit on the srs↔srs-z double cover; its girth-ring holonomy h^g measures the flip | geometry |

The unification is the identity **persistence ⟺ chirality oscillation ⟺
double-cover holonomy.** A massive particle persists *because* it flips; the
thing that persists *is* the standing L↔R oscillation. A massless particle does
not flip, carries no internal persistent structure, and simply streams.

## 3. The operator

### 3.1 Acting space and block structure

M_persistence acts on the substrate fermion mode space and is **block-diagonal
in species**:

$$M_{\text{persistence}} \;=\; \bigoplus_{s\,\in\,\{\nu,\,e,\,u,\,d\}} M^{(s)},
\qquad M^{(s)} : \mathbb{C}^3_{\text{gen}} \to \mathbb{C}^3_{\text{gen}}$$

— four species sectors (neutrino, charged lepton, up-type, down-type), each a
3×3 operator over the three generations. Its 12 eigenvalues are the 12 SM
fermion masses. (This is exactly the target named by the 2026-04-17 scoping
doc — now assembled rather than scoped.)

The **mass** operator and the **Yukawa** operator are the same object up to the
Higgs vev: $M_{\text{persistence}} = v \cdot Y_{\text{persistence}}$, with
$v = 246.22$ GeV (`predictions/v_higgs.py`).

### 3.2 M = shape ∘ dynamics

Each species block factorises into a **shape** layer and a **dynamics** layer:

$$M^{(s)} \;=\; \underbrace{A^{(s)} \cdot R^{(s)}}_{\text{shape (volcano, srs)}}
\;\cdot\; \underbrace{\big(1 - c_s\,\alpha_1/(1-\alpha_1)\big)}_{\text{dynamics (mirror, srs-z)}}$$

- **A^(s)** — the generation-3 *anchor*: the §3 selection-rule scalar
  `chir · Q^L / k*^edge_sel` (§4). The depth of the species' volcano vent.
- **R^(s)** — the within-generation 3×3 structure, normalised so its largest
  eigenvalue is 1: the Koide rotation for the cycle-walker species, the
  representation split for the spectral-walker neutrino (§6).
- the **dark correction** `(1 − c_s·α₁/(1−α₁))` — the srs↔srs-z interaction,
  the "lava": the mirror's contribution to the flip
  (`theorem_substrate_feshbach_dark_corrections_master.md`).

### 3.3 Eigenvalues and kernel

- **spectrum(M_persistence) = the 12 fermion masses.**
- **ker(M_persistence) = the massless fermion modes** — a mode lies in the
  kernel iff its girth-ring holonomy is trivial (h^g = +1): no oscillation, no
  persistence, no mass.

## 4. The shape layer — the volcano (srs)

The static, achiral branching substrate. As the universe cools from the GUT
scale, its compressible branching substructure freezes out; that frozen shape
is the **§3 selection rule** (`theorem_yukawa_master_theory_synthesis` §3):

$$y_X \;=\; {\rm chir}(\text{species}) \cdot Q^{\,L(\text{species})}
/ k_*^{\,{\rm edge\_sel}(\text{species})}$$

The shape supplies the *channels* — which Bloch concentration point, which of
the four §4(D) walker types, which chirality input. The four walker types are
the four ways the substrate supports a persistent walker:

| Walker type (§4(D)) | Species | Within-generation structure |
|---|---|---|
| I — spectral asymptotic | neutrino | representation split (4,2,2) + R = 228/7 |
| II — saturation (L=0) | up-type | Koide rotation |
| III — lepton cycle (L=g−2) | charged lepton | Koide rotation, ε²=2 |
| IV — Perron walker (L=g) | down-type | Koide rotation |

## 5. The dynamics layer — the mirror (srs-z)

srs-z is the *directed* lift of srs: the directed arcs are precisely what the
achiral srs lacks — **srs-z is where chirality lives**. The mirror layering
onto the volcano "with a tiny catch" is the Higgs broken phase (W20: Higgs
vacuum ↔ edge qubit f₁ ↔ mirror Z₂ ↔ involution σ on srs-z).

The srs↔srs-z interaction *is* the chirality-flipping dynamics — the "lava."
Its strength is α₁; it surfaces as the universal dark correction
`g_phys = g_bare·(1 − c·α₁/(1−α₁))`. The volcano shape says *how* the lava
flows down each channel; the mirror says *that there is* a flow at all.

A fermion's mass is then the **girth-ring holonomy** of its walker on the
srs↔srs-z double cover: a circuit returning to the mirror sheet (non-trivial
h^g) forces a standing L↔R oscillation — mass; a circuit returning to the same
sheet (h^g = +1) does not — masslessness.

## 6. The spectrum — 12 channels as one operator

| Sector | Block M^(s) | gen-3 anchor A^(s) | within-gen R^(s) | eigenvalues |
|---|---|---|---|---|
| **charged lepton** | 3×3 | y_τ = (5/3)Q⁸/k*² = 7.2165×10⁻³ | Koide ε²=2, δ=2/9 | m_τ, m_μ, m_e (1776.84 / 105.65 / 0.5110 MeV) |
| **up-type** | 3×3 | y_t = 1 (saturation) | Koide ε²_up (Row P37) | m_t, m_c, m_u |
| **down-type** | 3×3 | y_b = Q^g = 1.734×10⁻² | Koide ε²_down (R4 band) | m_b, m_s, m_d |
| **neutrino** | 3×3 | m_ν3 spectral (Type-I) | rep split + R = 228/7 | m_ν3, m_ν2, **0** (50.57 / 8.86 / 0 meV) |

All 12 SM fermion masses are now *one operator's spectrum*. The charged-lepton
block is theorem-grade (W43). The quark blocks are theorem-grade-conditional
(ε² bands + Need-D-3). The neutrino block: m_ν3, m_ν2 theorem-grade-conditional
(Need-D-3 + R = 228/7); the third eigenvalue is the kernel (§7).

## 7. The kernel — m_ν1 = 0

The lightest neutrino is **ker M_persistence**, and this is the one entry of
the spectrum *computed from first principles* (W44 reframe + W45,
`proofs/foundations/W45_nu_R_modecount_holonomy_2026-05-21.py`, 7/7):

On the framework's Hashimoto operator B(P), the spectrum splits into 8
Ramanujan walker modes (|h|²=2) and 4 trivial modes (|h|=1). The right-handed
Majorana mass *is* a girth-ring walker holonomy, M_R = |M_R|·h^g. Computing
h^g (g=10): the Ramanujan modes carry **non-trivial** holonomy (the live
α_21 = 162.39° / δ_CP-channel = 197.61° phases); the 4 trivial modes carry
**h^g = +1 identically** — trivial holonomy, no walker dynamics.

So the trivial-C₃ generation hosts no dynamical Majorana ν_R; the substrate
makes exactly **2**. A Type-I seesaw with 2 right-handed and 3 left-handed
neutrinos is rank-2 ⇒ exactly one massless light neutrino. **m_ν1 = 0.**

This is the proof-of-concept that M_persistence is a real operator with a real
spectrum: its kernel was *located by computation*, not assumed. (Conceptually,
the photon and gluon are massless for the same reason — trivial double-cover
holonomy — though they are eigenmodes of the gauge sector, not of this
fermion-mass operator.)

## 8. Honest grade

**SYNTHESIS-GRADE.** What this document delivers: the *operator framing* —
the 12 channels assembled as one block operator M_persistence, the
shape∘dynamics factorisation, and the kernel identification. What it does NOT
deliver: an unconditional, fully-computed spectrum. The honest conditional
structure, tiered by layer:

- **Dynamics-tier conditional — Need-D-3.** 11 of the 12 channels inherit the
  open Need-D-3 / V_Ram ≅ Cl(6)-Fock piece (the eigenbasis distinguishing the
  Y_u from Y_d "lava channels"). This is the framework's named multi-session
  block.
- **Shape-tier — the kernel.** y_ν1 = 0 is theorem-grade-conditional on A5(a)
  + the Probe-B Re-sign-lock — and is the *one* channel carrying **no**
  Need-D-3 dependency (W44/W45).
- **Scale anchors.** The shape layer gives mass *ratios* cleanly; the absolute
  scales (v, the M_R scale, the y_ν = 1 anchor) remain partly adopted — the
  master dark-correction doc's "single hard residue."

**What "completing the operator" means.** Two named jobs, one per layer: close
Need-D-3 (graduates the 11 dynamics-tier entries) and discharge the scale
anchors. Neither is required for the *framing* — M_persistence is already a
well-defined operator with all 12 modes placed and its kernel computed.

## 9. Cross-references

- `theorem_yukawa_master_theory_synthesis_2026-05-20.md` — the §3 selection
  rule and §4(A)–(D) sub-theorems; the off-shell content of this operator.
- `theorem_walker_length_MDL_waterline_2026-05-21.md` — §4(D), the 4 walker
  types.
- `proofs/foundations/W45_nu_R_modecount_holonomy_2026-05-21.py` +
  an internal working note — the
  kernel computation.
- `proofs/foundations/W46_mass_operator_assembly_2026-05-21.py` — assembles
  M_persistence as an explicit block operator and verifies its structure.
- `theorem_substrate_feshbach_dark_corrections_master.md` — the dynamics-layer
  dark correction.
  named the target this document assembles.
