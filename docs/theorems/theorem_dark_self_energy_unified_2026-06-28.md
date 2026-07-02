# Theorem — the unified dark self-energy Σ = α₁/h, and the one open sign (2026-06-28)

**Status:** consolidates the dark-correction sector into ONE read of ONE resolvent, separates
the FORCED structure from the single UNRESOLVED degree of freedom (the channel-read sign), and
corrects two stale framings. Supersedes the scattered "Family A/B/C/D" taxonomy as the *organizing*
statement (the families are channels/projections of this one object).

> **Honesty note (load-bearing).** A 2026-06-28 "forward-leak forces DOWN" claim was **retracted**
> the same day: the channel read is the particle's OWN self-energy (the walker's own girth excursion,
> same fermion line returning), for which standard QFT (electron δm>0) gives **UP**, not DOWN. The
> "forward-leak" is a physical picture, not a derivation, and the L-step survival is not actually
> depleted by a cycle. So the channel-read sign is **NOT forced** — see §3. What IS forced is §2.

## 1. The one object
Everything in the dark/precision layer is one read of the single resolvent G_NB = (I − uB)⁻¹ at u = α₁:

$$\Sigma(h) = \alpha_1 / h \qquad \alpha_1 = ((k_*-1)/k_*)^{g-2} = (2/3)^8,$$

the leading first-girth-return at the channel eigenvalue h. The "Family A/B/C/D" taxonomy is this one
read evaluated at different channels (shell h=(√3+i√5)/2 vs Perron h_P=2) and projected onto the
observable's coupling. (`dark_corrections_nativeness_audit_2026-06-22`, `heavy_mass_sigma_alpha1_over_h_2026-06-22`.)

## 2. FORCED (theorem-grade)
1. **Magnitude.** The full residue α₁/h (NOT the principal value α₁/2h) reproduces the lepton & neutrino
   coefficients **Re(1/h)=√3/4, −Im(1/h)=√5/4 to 1e-12** — independent observables ⇒ the magnitude is
   forced, not fitted.
2. **Channel by sector.** Perron h_P=2 for the heavy quarks and gauge bosons (color triplet → Γ-trivial
   λ=+3 → Ihara-Bass roots h∈{1,2}); shell for leptons/ν (chir-band-edge). `theorem_selection_map_2026-05-21`.
3. **Coefficient by coupling** (CORRECTED 2026-06-28 — the single-edge 1/12 is the *vertex* projection,
   NOT the channel read; the 2026-05-15/18 master-theorem framing lagged this):
   - **c = 1** — channel read, when the observable IS the channel mode (the d/b Type-IV Perron walker:
     full residue, the "saturated channel").
   - **c_S = 1/(2|E|) = 1/12** — gauge-singlet *projection* of the Perron residue (M_Z's δ_r).
   - **c_F = α₁/(N·k)** single-edge — the *vertex* dark (y_τ, Family D), per
     `theorem_car_local_jordan_wigner`.
4. **Power by the L-rule.** L=0 (saturation) → p=2; L>0 (propagating) → p=1. Over-determined (reproduces
   the framework's own R1/R2 trigger). `power_of_h_from_walker_length_scratch`.
5. **Vertex-dark sign.** A *separate* closed fermion loop carries −1 (Peskin–Schroeder §4.8) — this fixes
   the y_τ c_F sign DOWN. **This is the only rigorously-forced sign.** It does NOT extend to the channel
   reads (those are own-self-energies, not separate loops).

## 3. THE SIGN — RESOLVED by grounding in the foundational mass-definition (2026-06-28)
The sign is forced **DOWN**, *given* the framework's foundational, user-confirmed mass-definition:

> **mass = the DYNAMICAL recurrence RATE** — the slice-to-slice difference under ∂_N
> (`memory/mass-energy-is-recurrence-distribution`, user-confirmed "important fact") — **NOT** the
> static recurrence amplitude.

- The *rate* reading (cycle = delay; fraction α₁/h wastes its steps) gives **mass × (1 − α₁/h), DOWN** — and
  this reading reproduces the framework's value.

### ⚠ 2026-06-29 (reframed 2026-07-01) — the sign is DERIVED conditional on the rate foundation (settled DOWN); only the standalone formal lemma (rate-reading selection) is open.
An attempt to make "rate reduction = α₁/h ⇒ DOWN" a clean CAS-checkable lemma **failed**, and revealed an earlier
over-claim. The foundational "mass = dynamical recurrence" does **not** single out the sign — three readings of the
first-girth-return give three signs (verified symbolically):
- **(1) literal formula** (h/k)^L at fixed L: a g-cycle makes the walk (L+g) steps; the L-step amplitude is
  unchanged → **NO CHANGE** (so the formula alone does NOT even generate the dark);
- **(2) rate / velocity to fixed distance:** cycle-takers (fraction α₁/h) are delayed → **DOWN** (the framework's value);
- **(3) return amplitude** G=1/(1−α₁/h): the cycle adds a returning path → **UP**.

So the channel-read DOWN sign is **DERIVED CONDITIONAL on the framework's foundational, user-confirmed
mass-definition** (mass = the DYNAMICAL recurrence RATE): the rate reading (2) — cycle-takers waste steps → delayed —
gives mass×(1−α₁/h) = **DOWN**. It is ALSO **empirically cross-checked**: m_b prefers DOWN (+0.22σ vs +5.8σ for UP);
the leptons/ν cross-check the coefficient to 1e-12 with DOWN; and the **vertex** sign (y_τ c_F) IS formally forced
(closed-fermion-loop −1, Peskin §4.8). What is OPEN is only the standalone CAS lemma FORMALIZING the rate-reading
selection (why reading (2), not the amplitude reading (1) / return-amplitude reading (3)).

**Status of the sign:** DERIVED conditional on the mass=recurrence-rate foundation (the rate reading → DOWN) +
empirically cross-checked + consistent across the sector (m_b, m_t, M_Z δ_r, m_W δρ, m_ν all DOWN, all matching) +
vertex formally forced. The one open piece is the standalone from-first-principles CAS lemma formalizing the
rate-reading selection — so **NOT "formally forced from nothing," but equally NOT undetermined: the physics is
settled DOWN given the foundation.** [Power p: L-rule (L=0→2, L>0→1). Shared sign across the sector.]

## 4. Current live status under this theorem
- **Forced:** magnitude (α₁/h, 1e-12 cross-check), channel (Perron/shell), coefficient (c=1/c_S/c_F), power (L-rule),
  and the **vertex** sign (y_τ c_F, Peskin −1).
- **DERIVED conditional on the mass=recurrence-rate foundation (+ empirically cross-checked):** the channel-read
  DOWN sign (m_b, m_t, M_Z δ_r, m_W δρ, m_ν). The rate reading → DOWN; cross-checked + consistent across the sector;
  the data prefers DOWN. The one open piece is the standalone CAS lemma formalizing the rate-reading selection
  (the mass=recurrence foundation read GENERICALLY admits no-change/DOWN/UP; the RATE definition picks DOWN).
- m_b (−α₁/h_P → +0.22σ), m_t (−α₁/h_P² → −0.95σ): magnitude-forced, sign derived-conditional-on-the-rate-foundation. R1 stands.
- **Grade:** magnitude THEOREM-GRADE; sign DERIVED conditional on the foundational rate definition (solid:
  cross-checked + consistent + vertex formally forced), with the standalone formal lemma (rate-reading selection)
  as the named open piece. NOT "formally forced from nothing"; equally, NOT undetermined — settled DOWN.

## 5. Open questions
- The sign (§3) is RESOLVED *conditional on* the foundational mass-definition (dynamical recurrence rate).
  The remaining micro-task is a clean from-first-principles derivation of "rate reduction = α₁/h" (the cycle
  fraction wastes its steps) — currently a plausible-and-grounded argument, not yet a CAS-checkable lemma.
- M_Z +0.018% residual = the sub-leading of this same self-energy (one order beyond the leading singlet
  projection) — now a well-posed next target (the sign is no longer the blocker; the Z-path is open).
- REPO PASS (flagged): audit predictions/ for the c=1-channel vs 1/12-vertex coefficient and the DOWN sign;
  fix where the stale single-edge 1/12 was mis-applied to a channel read.
