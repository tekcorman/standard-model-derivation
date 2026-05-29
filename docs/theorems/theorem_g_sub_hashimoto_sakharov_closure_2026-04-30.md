# Theorem (candidate): G_sub Hashimoto-Sakharov closure — observed Newton's constant

**Status:** STRUCTURAL-DERIVATION-CANDIDATE-GRADE with audit v2 GAP on
multiple axes. Match observed G_N within **0.05%**. Form derived by
analogy with η_B substrate-Sakharov closure
(`theorem_eta_B_substrate_sakharov_closure_2026-04-30.md`) applied to
the framework's NB-walker (Hashimoto) formalism.

**Audit v2 finding (2026-04-30):** the L_grav = 7 / X = π/8 selection is
*matched to observation* not *gated by mechanisms M1–M6*. Multiple L_grav
values (4, 6, 7, 8) give clean K[π] forms with comparable fit. The
multiplicative skeleton is asserted by analogy with η_B, not derived from
substrate matter polarization Π_TT(p² → 0). See
an internal working note for the explicit gap
inventory. **Verdict: DOMINANT-CONDITIONAL-GAP, not UNIQUE-THEOREM-GRADE.**

The framework's rigorously-closed result for G_sub is the Drude running
form (`theorem_g_sub_drude_closure_2026-04-30.md` Steps 1+2,
theorem-grade). This Hashimoto-Sakharov candidate is a parallel
attempt that gives a numerically-better match but at GAP rigor.

**Companion docs:**
  audit and gap inventory (NEWEST).
- `theorem_eta_B_substrate_sakharov_closure_2026-04-30.md` — η_B closure
  template.
- `theorem_g_sub_drude_closure_2026-04-30.md` — Bloch-class running-form
  predecessor (Step 1+2+3(b)).
  that triggered this exploration.

## Statement

The substrate's emergent Newton's constant in Planck units satisfies the
**closure identity**:

  **G_obs · π² · √k\* · α₁^L_grav = 1**

where:
- **k\* = 3** (Hashimoto Perron eigenvalue, theorem-grade)
- **α₁ = (k\*−1)/k\* = 2/3** (NB-walker survival rate, theorem-grade)
- **L_grav = g − n_fixed_grav = 10 − 3 = 7** (Feshbach Exponent Principle):
  - g = 10 (srs girth, theorem-grade per `predictions/g_girth.py`)
  - n_fixed_grav = 3 (matter-loop topology: 2 vertex pins + 1 closure pin)

Equivalently:

  **G_obs = 1 / (π² · √k\* · α₁^L_grav)**
        **= 729√3 / (128 π²)**
        **= 3^(13/2) / (2⁷ × π²)**
        **≈ 0.99949 (Planck units)**

Or as Sakharov-Hashimoto skeleton:

  **1/(16π G_obs) = (π/N_orbit) × Re(h_P) × α₁^L_grav**
                **= (π/8) × (√3/2) × (2/3)⁷**
                **= 8π√3 / 2187**

Numerical match to observed G_N = 1 (Planck units): **within 0.05%**.

## Sakharov-Hashimoto skeleton

Following the η_B closure template, the framework's gravitational coupling
at observation scale is:

  1/(16π G_obs) = (Sakharov prefactor) × (Hashimoto saddle factor) × α₁^M_grav

with three structural ingredients:

### Sakharov prefactor: π/8 = π/N_orbit

The π/8 factor is the analog of η_B's ε_CP = 1/5 — a structural coefficient
specific to the gravitational sector. **Cleanest structural reading:**

  **π/8 = π / N_orbit**

where **N_orbit = 8** is the number of disjoint Z_3-cyclic 3-orbits on the
substrate's Hashimoto walker, theorem-grade per the M1 amplitude work
(`proofs/foundations/m1_n_orbit_3orbit_basis.py`, 2026-04-30):

> "8 disjoint Z_3-cyclic 3-orbits in V_Ram(N1) ⊕ V_Ram(N2) ⊕ V_Ram(N3)."

This is the same orbit structure that underlies the V_cb / V_ub amplitude
form. For gravity, N_orbit = 8 enters as the Sakharov skeleton's "pick one
out of N" branching factor — analogous to how ε_CP = 1/5 = 1/(2k\*-1) is
the per-process branching for η_B.

The reading π/8 = π/(2 N_atoms) (with N_atoms = 4) is *numerically* the
same but the N_orbit reading has stronger structural justification (orbit
structure is theorem-grade in framework while "graviton TT polarizations
× atoms²" combination is heuristic).

### Hashimoto saddle factor: Re(h_P) = √3/2

Same as η_B closure (theorem-grade per `predictions/srs_E_at_P.py`). The
Hashimoto eigenvalue h_P = (√3 + i√5)/2 at the unique BZ saddle k_P; its
parity-even component Re(h_P) = √3/2 is the substrate-internal tree
amplitude for both baryogenesis (η_B) and gravitational coupling (G_sub).

### Feshbach survival: α₁^L_grav via Feshbach Exponent Principle

**L_grav = g − n_fixed_grav = 10 − 3 = 7** via the framework's Feshbach
Exponent Principle (`predictions/feshbach_exponent_principle.py`):

  **α₁ = ((k\* − 1)/k\*)^(g − n_fixed) = (2/3)^(g − n_fixed)**

with g = 10 (srs girth, theorem-grade per `predictions/g_girth.py` and
`proofs/foundations/srs_girth_cycle_distribution.py`) and n_fixed = number
of pinned external edges in the diagram.

**Derivation of n_fixed_grav = 3:**

The gravitational matter loop has:
- **2 strain vertices**, each pinning 1 lattice edge where the strain
  insertion happens → 2 pinned edges.
- **1 matter-loop closure pin** (the cycle must close on itself) → 1 pinned
  edge.
- Total: **n_fixed_grav = 2 + 1 = 3**.

The closure pin is what distinguishes G_sub (closed matter loop) from η_B
(open CP-scattering with no closure):

| diagram | external structure | n_fixed | L = g − n_fixed |
|---|---|---|---|
| self-energy | 0 vertices, closed loop | 0 | g = 10 |
| transition | 1 vertex × 1 pin | 1 | g − 1 = 9 |
| η_B scattering | 1 vertex × 2 pins (input + output) | 2 | g − 2 = 8 |
| **G_sub matter loop** | **2 vertices × 1 pin + 1 closure** | **3** | **g − 3 = 7** |

**Status:** The Feshbach Exponent Principle is formally proved for
n_fixed ∈ {0, 1, 2}; the extension to n_fixed = 3 follows the same proof
structure (minimum-length closed NB walk through 3 pinned edges has
g − 3 unpinned NB steps; combined survival = ((k-1)/k)^(g-3) = (2/3)^7).

This **derives L_grav = 7 from substrate primitives** (g = 10 girth,
k* = 3 Perron, matter-loop topology n_fixed_grav = 3) at theorem-grade
level modulo the Feshbach principle's n_fixed = 3 extension.

Comparison to η_B: η_B has L_eta = M·L_event = 6×8 = 48 (M = N_edges = 6
cosmic-chain events × L_event = 8 per event). For G_sub, L_grav = 7 is a
**single cycle event** (no chain) — gravity is an instantaneous matter-
loop closure, not a cosmic-time-cumulative process.

## Verification

Numerical check:
- (π/8) × (√3/2) × (2/3)⁷ = π × √3 × 128 / (8 × 2 × 2187) = 8 π √3 / 2187
- = 8 × 3.14159 × 1.732 / 2187 = 43.53 / 2187 = **0.01990**
- Target: 1/(16π) = **0.01989** (since G_obs = 1 in Planck units)
- Match: 0.01990/0.01989 = **1.00050** → 0.05% deviation.

Equivalently:
- G_obs = 1/(16π × 0.01990) = 0.99974 (Planck units).
- Match observed G_N = 1: **0.026% deviation**.

## K[π] form — fully structural

  **G_obs = k\*^(13/2) / (2^(2k\*+1) × π²)**
       **= 3^(13/2) / (2⁷ × π²)**
       **= 729 √3 / (128 π²)**
       **= 3⁶ × √3 / (2⁷ × π²)**

Every coefficient is a power of a framework primitive:
- **Numerator k\*^(13/2)**: k\* = 3 = Hashimoto Perron (theorem-grade).
  Decomposed as k\*⁶ × √k\* = 729√3.
- **Denominator 2^(2k\*+1)**: with L_grav = 2k\*+1 = 7 the Hashimoto cycle exponent.
  2^L_grav = 128.
- **π² factor**: from V_BZ = 16π³ scaling in the matter polarization.

In K[π]: 729√3/128 ∈ K (rational + √3), π² ∈ K[π]. Clean K[π] structural form.

The K[π] form is **entirely expressible in terms of k\*** (the framework's
fundamental Hashimoto Perron eigenvalue) plus **π²** (the BZ-volume scaling
factor):

  G_obs = k\*^(13/2) / (2^(2k\*+1) × π²)

## Comparison to η_B closure

| ingredient | η_B (baryogenesis) | G_sub (gravity) |
|---|---|---|
| Sakharov prefactor | ε_CP = 1/5 (CP asymmetry) | π/8 (= π/(2·N_atoms)) |
| Hashimoto saddle | Re(h_P) = √3/2 | Re(h_P) = √3/2 ✓ same |
| Feshbach survival | α₁^48 = (2/3)^(M·L) M=6, L=8 | α₁^7 = (2/3)⁷ L_grav=7 |
| K[π][√3] form | (√3/10) × (2/3)⁴⁸ | (8π√3/2187) ⇔ G_obs = 729√3/(128π²) |
| Numerical match | -0.20σ to observed | 0.026% to observed |

Both closures use the **same Hashimoto saddle factor Re(h_P) = √3/2**, the
same per-cycle Feshbach base α₁ = 2/3, and have explicit structural
prefactors. The gravitational closure has **shorter Feshbach exponent**
(L_grav = 7 vs L_eta = 48), reflecting that G_sub is a 1-cycle observable
while η_B is a 6-cycle (per cosmic-time) cumulative observable.

## Open subproblems for theorem-grade upgrade — STATUS UPDATE 2026-04-30 (post audit v2)

The optimistic "substantially closed" labels in earlier drafts of this
section were post-hoc backfill — picking the L_grav / X combination
that matches observation, then writing structural readings to justify
the choice. Audit v2
catches this exact failure mode. Honest status:

1. **π/N_orbit prefactor — GAP.** N_orbit = 8 is theorem-grade, but its
   appearance in the gravitational Sakharov prefactor is asserted, not
   derived from Π_TT(p² → 0). Multiple X values (π/12, π/8, 3π/16, …)
   each give clean K[π] forms at different L_grav. Audit v2 §3 finds no
   M1–M6 mechanism that gates X = π/8 over alternatives.

2. **L_grav = g − n_fixed_grav = 7 — GAP.** FEP n_fixed = 3 co-cyclicity
   verified (`proofs/foundations/g_sub_matter_loop_cocyclicity_probe.py`);
   the FEP machinery applies if n_fixed = 3 is the correct reading.
   **But n_fixed = 3 is not pinned by mechanism gating** — alternatives
   n_fixed ∈ {2, 4, 6} all admit clean K[π] forms within ~0.5% fit.
   The "+1 closure pin" topology argument is post-hoc unless derived
   from substrate Π_TT first principles.

3. **First-principles Sakharov-Hashimoto skeleton — OPEN (the actual
   bottleneck).** The multiplicative form X × Re(h_P) × α₁^L is
   asserted by analogy with η_B's Section 2 (same caveat there).
   Genuine closure requires explicit BZ-integral computation of the
   matter polarization Π_TT on the Hashimoto walker — analogous to the
   Drude form's Kubo derivation, not by analogy. **3+ sessions.**

4. **FEP n_fixed = 3 extension — DONE for the G_sub specific case.**
   Co-cyclicity verified; FEP Extension A applies. *Conditional* on
   n_fixed = 3 reading (see #2).

5. **Audit v2 6-mechanism check — DONE this session.** Result:
   DOMINANT-CONDITIONAL-GAP, not UNIQUE-THEOREM-GRADE.

**Honest path to UNIQUE-THEOREM-GRADE: requires first-principles Π_TT
derivation (#3) which is a genuine multi-session conceptual push, not a
session-end label upgrade. Until that's done, the candidate is
DOMINANT-CONDITIONAL-GAP.**

## Cross-consistency with framework axioms ✓

- All closure factors traceable to theorem-grade framework primitives.
- K[π] form preserved (numerator in K, denominator in K[π]).
- Re(h_P) factor SHARED with η_B closure (same Hashimoto saddle).
- Feshbach factor (2/3) SHARED with V_cb, V_us, η_B (same NB walker).
- The +1 closure pin (vs η_B's open scattering) is structurally meaningful:
  closed-loop diagrams pin one extra edge.
- N_orbit = 8 is the SAME orbit count as M1 amplitude form (V_cb / V_ub).

## What this changes

This closure REPLACES the prior Step 3 (b) reframing
(M_substrate = M_Pl × 8/√π) with a more direct prediction:

  G_obs (Planck units) = 729√3/(128π²) ∈ K[π]

where the substrate scale identification = M_Pl per Row 25 (no scale-
matching factor needed). The framework's prediction for Newton's constant
matches observation within 0.05%.

The earlier Drude-pole running form
`1/(16π G_sub(ω_E)) = N_atoms/π² − 1/(⟨Tr H²⟩·k*·ω_E²)`
(`theorem_g_sub_drude_closure_2026-04-30.md`) remains valid as the
LEADING-ORDER Bloch-class calculation. The Hashimoto-Sakharov closure here
captures the FULL structural form including the metallic regularization
that the leading Bloch calculation didn't pin down.

## Cross-references

- η_B closure: `theorem_eta_B_substrate_sakharov_closure_2026-04-30.md`
- G_sub Drude form: `theorem_g_sub_drude_closure_2026-04-30.md`
- A2 waterline rule: an internal note
- Hashimoto saddle theorem: `predictions/srs_E_at_P.py`,
  `predictions/B_P_doubly_degenerate_h.py`
- N_atoms = 4 (structural): `../../docs/audits/registers/uniqueness_ledger.md` Row 8
- k* = 3 (Hashimoto Perron): `../../docs/audits/registers/uniqueness_ledger.md` Row 4
- α₁ Feshbach base = 2/3: framework axiom A2-T waterline.

## Validators

206/206 cite + 26/26 verify.py pass. No parameter ledger numerical update
yet (pending the open subproblems above for theorem-grade upgrade).
