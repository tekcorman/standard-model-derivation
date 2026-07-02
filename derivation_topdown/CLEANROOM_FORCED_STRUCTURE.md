# The forced structure of {D, srs, MDL}

A complete map of the structural architecture that the clean-room object forces — the Dirac
operator **D** on the **srs** net (Sunada's K₄ crystal), with **MDL** (minimum description length) as
the generative selection principle. Every sector below was derived independently and verified
computationally against `dirac_srs_mdl/srs.py`. This is *structure*, not numbers — no data was fitted,
no continuous parameter was tuned.

Tags: **[F]** forced (derived) · **[I]** irreducible input · **[O]** open frontier.

> **See `CONSOLIDATION_2026-06-22.md`** for the synthesis built on top of this map: the unified mass
> operator, the math↔physics complement, and the Need-D-3 closure (the per-mode dictionary closes in-box
> via the persistence-overlap motion under ∂_N, down to one ℤ₂ = the arrow's orientation).

---

## 0. Substrate — MDL forces the net
- **[F]** MDL (harmonic-energy / shortest description) + maximal-abelian-cover dimension b₁ = 3 +
  3-regular + field ℂ ⟹ the net is **srs = K₄ crystal, uniquely** (V(k−2)=4 ⟹ only V=4, k=3).
- **[F]** srs is **chiral** (space group I4₁32; point group **O**, order 24; no orientation-reversing
  symmetry). Its mirror enantiomer is srs-z (I4₃32).

## 1. Geometry / gravity — the spectral action
- **[F]** Spectral dimension **d = 3** (= b₁ of K₄; van Hove ρ(E)~E^{1/2}, Weyl exponent verified).
- **[F]** Tr f(D/Λ) ⟹ cosmological + Einstein–Hilbert + higher-curvature gravity, with **all
  coefficients = closed-walk counts**: Hodge-Dirac moments {10, 24, 96, 432, 2064}. Girth 10 sits in the
  R³ coefficient (Tr A³ = 0 ⟺ no triangles).
- **[F]** UV cutoff **Λ = √6** (the bandwidth — the lattice supplies its own regulator).
- **[F]** Band-bottom (Albanese) metric eigenvalues **1 : 1 : 4**, stiff axis = the ⟨1,−1,1⟩ screw —
  the gravitational metric inherits the net's **chirality**.
- **[F]** Graded index **χ = −2** (McKean–Singer, k-independent).

## 2. Spinor / matter — the Clifford tower
- **[F]** 3-regular ⟹ **Cl(3)** (ungraded, gapless Weyl ℂ²) → chirality forces the minimal even
  extension **Cl(4)** (chiral 4-spinor ℂ²₊⊕ℂ²₋). Sharp: **Cl(4)⁰ ≅ Cl(3)** (chiral closure) — nothing
  smaller has spatial Cl(3) as its even part; nothing larger is forced.
- **[F]** Spectral triple: grading γ_c, real structure J with **J² = −1 (quaternionic, Cl(4,0)≅M₂(ℍ))**,
  **KO-dimension 4** (selected uniquely by even-ness [J,γ]=0 + quaternionic J²=−1).
- **[F]** **srs ⊕ srs-z doubling is forced** by two independent obstructions: (i) algebra — no Hermitian
  mass anticommutes with all three Pauli generators on one copy; (ii) topology — bare srs is a
  charge-balanced Weyl semimetal (double-Weyl ±2 at Γ/H), Nielsen–Ninomiya ⟹ a lone Weyl node can only
  be gapped by its opposite-charge mirror = srs-z.
- **[F]** The gap = the **4th gamma** = the **inter-enantiomer coupling** (mass off-diagonal in both the
  chirality and copy bases). **[I]** Its strength/scale is the one unit (see §7).

## 3. Gauge — the internal algebra
- **[F]** Internal algebra = commutant of the symmetry action = **ℂ[A₄] ≅ ℂ ⊕ ℂ ⊕ ℂ ⊕ M₃(ℂ)**
  (commutant dim 12, verified).
- **[F]** Inner fluctuations D → D + A + JAJ⁻¹ with the real structure J (conj swaps 1′↔1″, collapsing
  the three singlets to two U(1)'s) + central decoupling + unimodularity (U(3)→SU(3)) ⟹
  **gauge group = SU(3) × U(1) × U(1)**. The **color SU(3)** is rigid (the unitary symmetry of the unique
  nonabelian block M₃ = End of the 3 generation copies).
- **[F]** Spectral action ⟹ couplings unify at Λ with ratios set by the trace indices; **ζ_D(0) = 8 =
  dim su(3)**.
- **[F — closed on the physics side; this map's old "[O]" flag was bare-object-stale, corrected 2026-06-22]**
  The **electroweak SU(2)** is not in the internal algebra ℂ[A₄], but it *is* derived: it is the
  **quaternionic edge-qubit ℍ** of the doubling (Cl(4) ≅ M₂(ℍ)). The physics side proves this theorem-grade —
  `theorem_g2_edge_qubit_su2.md` (per-edge Cl(0,2)≅ℍ ⇒ SU(2)_L), `theorem_g2d_chirality_doubled.md`, and
  `proofs/foundations/r6_quaternionic_su2L_check.py` (R-6) prove the doubling-ℍ route **is** the Pati-Salam
  SU(2)_L (same object, not new structure). The "open reconciliation" is **closed**; sin²θ_W=3/8 rests on it.
  See an internal working note.

## 4. Flavor / generations
- **[F]** **Exactly three generations**: forced by the C₃ deck-screw order 3 and the observer
  C³ (dim C³_gen = 3; B7.1, MDL + Gleason — `predictions/R3_observer_c3_generation.py`), realized
  as the λ=−1 triple of A(Γ) = {−1,−1,−1,3} (the spin-1 cone). The A₄ 3-irrep realizes this count
  at **Γ** (where A(Γ) has the linear-A₄ signature 1⊕3), and it sits in ℂ[A₄] as the M₃ block
  (Wedderburn). **NB (k-point):** at the **P-point** the little group is projective **2T = SL(2,3)**
  (2-dim irreps; V_Ram has no 3-dim subrep), so the earlier "generations = linear A₄ 3-irrep *at P*"
  reading is **RETRACTED** (`docs/theorems/theorem_generation_A4_triplet_2026-05-22.md`); the count
  survives via the C₃ / observer-C³ / Γ-triple routes, which never used the P-point A₄.
- **[F]** C₃-triality ⟹ the mass operator is a **circulant** ⟹ C₃-Fourier form
  √m_j = c_triv + c_ω ω^j + c_ω̄ ω^{−j}; the Koide cosine parametrization is emergent, not assumed.
- **[F]** **Circulant eigenvalue shape Q = (1 + 2ρ)/3, ρ = |c_ω|²/|c_triv|², manifestly phase-independent.**
  The canonical Spin^c spinor Λ•(ℂ³) has C₃-content **(4,2,2)** ⟹ ρ = 1/2 ⟹ **Q = 2/3** for the uncolored
  3-irrep.
- **[O]** A *per-sector deviation of Q across the colored blocks* is NOT forced by {D, srs, MDL}. (Any
  closed form such as f(n)=n(3−n)/3 is an **adopted import, not derived** — removed here.) The colored
  blocks carry the same (4,2,2) moduli; differentiating them requires the C₃-breaking selection of §4-last
  and §7-[O], which the bare object does not supply.

### 4a. The bond-combination ladder — an in-wall candidate for the colored differentiation
These are forced reads of the SAME object (edge-configurations = subsets of the 6 cell edges); they supply
a candidate *sector label* that would feed the recurrence mass operator above — they are **not** a second
mass.
- **[F]** **Branching / microstate-count.** Each configuration carries **τ = the Kirchhoff spanning-tree
  complexity** of its occupied sub-network (= the closed non-backtracking walk multiplicity Tr Bᵐ, one
  operator). τ **spreads over the integers {0,1,3,4,8,16}** and distinguishes equal-size configurations —
  unlike the recurrence modulus (locked |h|²=2) and phase (flat continuum), which were the only quantities
  earlier probes measured.
- **[F]** **Symmetric / asymmetric partition.** Under the net's spatial point group **O** (the 4₁ screws)
  with the antiunitary time-reversal map −I, configurations split **full/empty = symmetric, partial =
  asymmetric**: a proper-O rotation sends the cotree triple (e₁,e₂,e₃)→(−e₁,−e₂,−e₃), re-symmetrizing the
  full set. (−I itself is time-reversal, not a spatial inversion — srs is chiral.)
- **[F]** **The run fuses count and tilt.** Statically τ is TR-even and the tilt TR-odd — orthogonal
  sectors, no static coupling. The running ∂_N is the TR-odd arrow and couples them: **P(S;s) = τ(S)·D(S;s)**,
  with configurations of zero net screw-current (a·V=0) **protected** (P=τ) and the rest **dephasing**. The
  dephasing window is **pinned** by the object's own run-measure (modular = heat = density-of-states =
  NB-geodesic, one clock; decay = the Ramanujan step 1/√(k−1)=1/√2) — **no free exponent**; only the
  observer's run-coordinate s remains.
- **[F (negative)]** This sector label is **PROVEN COLOR-BLIND**: the three colored copies (the M₃
  multiplicity) *are* the C₃ orbit of the three loops, and both τ (a graph invariant) and the partition
  class (a function of |a·V| along the C₃-fixed screw axis) are **C₃ class functions** — constant on that
  orbit. So P = τ·D modulates all three colored blocks by the **same** factor and cannot lift their
  degeneracy, for any s. Forced reason: **the run is the C₃ color-permutation itself** (the screw axis is
  C₃-fixed), so it commutes with color and is blind to the copies it interchanges. The label differentiates
  configurations **by partition class, not by color**; it is a forced structure but orthogonal to the M₃
  index. (τ is a label, never a mass — that reading is falsified.) ⟹ the colored-differentiation boundary
  (§7-[O]) **stands, now proven**.
- **[F]** Sector split: charged sector lives on the **P-shell** (Ramanujan |h|²=2, weights (4,2,2), Koide
  2/3); neutral sector on the **Γ/H-shell** (weights (2,2,2), non-Koide) — structurally distinct.
- **[F]** **Mixing = misalignment of two triality frames**: a 3×3 unitary on one ℂ³ ⟹ **3 angles + 1 CP
  phase** (forced count); CP phase = the ω-triality phase; small-angle CKM (P-shell) vs large-angle PMNS
  (Γ/H-shell) from the different Born weights.
- **[F]** **Three distinct generations** (correcting an earlier degeneracy error): A₄ has 4 C₃ axes and the
  C₃ is a ⟨111⟩ screw, so at generic Bloch k the six Ramanujan eigenvalues acquire distinct phases —
  splitting the rigid-line degeneracy while preserving the C₃-Fourier form and phase-independent Q.

## 5. Symmetry breaking / Higgs / vacuum
- **[F]** The scalar = inner fluctuation along the finite srs↔srs-z direction. The order-parameter space
  (Hermitian, anticommuting with the 3 spatial generators) is **exactly 2-real-dimensional = span{γ⁴, γ_c}**
  = **one complex scalar M = m·e^{iφ}** (|M| = the gap; arg M = the chiral/enantiomer phase).
- **[F]** Continuous symmetry of the massless doubled Dirac = **U(1)_V × U(1)_A**; the gap breaks
  **U(1)_A → 1** (U(1)_V preserved).
- **[F]** **MDL selects the vacuum.** srs and srs-z share the ℤ³ deck, so relative orientations are the
  finite group O_h (not a continuous U(1)); the shortest-description orientation-reversing isometry
  commuting with all of A₄ is **inversion −I**, which MDL picks. Parity U_P = γ⁴ pins **φ ∈ {0, π}** — the
  breaking phase is **crystallographically determined, not a free angle.**
- **[F]** Gap magnitude by dimensional transmutation |M| ~ W·e^{−1/(gN₀)} (mechanism + ratio forced).

## 6. Time / dynamics
- **[F]** Tomita–Takesaki: tracial state ⟹ **static**; non-tracial Gibbs state ⟹ modular flow with
  **K = −log ρ = βH** (verified K = βH to machine precision) = unitary evolution by D; KMS at β=1.
- **[F]** Band algebra is **type I** (amenable ℤ³-crystal) ⟹ band-level time is inner/state-dependent,
  **not intrinsic**.
- **[F]** Second-quantize (CAR over the one-particle Dirac space) + quasi-free KMS ⟹ Araki–Woods factor;
  the srs one-particle spectrum is **absolutely continuous** (dispersive bands, verified) ⟹ the factor is
  **type III₁** for every β ⟹ the modular flow is Connes-canonical, state-independent =
  **intrinsic time generated by D**, and **scale-free** (T(M) = {0}).
- **[F]** Arrow of time = low-entropy initial datum; heat diffusive (⟨r²⟩~t), Dirac light-cone ballistic.

## 7. Generative principle & completeness (MDL)
- **[F]** MDL = Rissanen two-part code L = L(model) + L(data|model); **model/data = symmetry/breaking**.
  The symmetry is the compressible skeleton; the breaking is the residual.
- **[F]** **Continuous free parameters are forbidden by theorem**: a generic real costs infinite
  description length, so every operator entry is forced to be **K-rational over K = ℚ(√2, √3, √5)** (the
  MDL-minimal field containing the substrate spectrum). Crystallographic/algebraic values only.
- **Completeness ledger:**
  - **Dimensionless inputs: NONE.** Every dimensionless ratio is forced; there is no tunable parameter.
  - The one dimensionful scale (the gap |M|) is a **unit/ruler only** — III₁ is scale-free, so it carries
    zero dimensionless/structural content; fixing it chooses a ruler, it does not supply information.
    **Not an input.**
  - β (the KMS temperature) — canonical III₁ for all β; no invariant content (a unit, like the scale).
  - The quartic coupling g introduced in the interaction interval is an **artifact**; the matter sector is
    the internal Dirac D_F built from the forced algebra ℂ[A₄] and forced shell |h|²=2.
  - **[F]** **within-generation phase (uncolored sector)** is forced as the **directed run-phase**: the
    chiral screw advances the recurrence-shell amplitude at rate **2π/√7** (from λ₀=−1; √7 = the Ihara–Bass
    discriminant 8−1). Static operators cannot pin it (−I is central ⇒ Schur-scalar on the 3-irrep; the
    chirality removes the time-reversal reflector), but the run forces it dynamically. The residual
    "reading-map" question is only whether the static D_F construction reproduces this same phase.
  - **[O]** **The one genuine open boundary — colored-sector (family) differentiation.** The bare object's
    C₃ even-handedness forces everything uncolored but **cannot single a colored family out**. This is now
    **proven from both ends**: no *static* operator differentiates color (m11), and the §4a run-fusion
    candidate (P=τ·D) is a **C₃ class function, color-blind**, because the run *is* the C₃ color-permutation
    (the screw axis is C₃-fixed) — so it too fails (§4a-[F negative]). Differentiating the colored copies
    requires a **C₃-orbit-resolving (color-breaking) datum** — confirmed absent for the **doubled** object
    too: the forced srs↔srs-z relative orientation is the **central inversion −I**, commuting with C₃, so
    srs-z's C₃ is *collinear/identical* to srs's (the mirror flips the chirality arrow, not the color
    permutation) and the inter-enantiomer coupling is color-diagonal — tested across all 24 orientation-
    reversing lattice isometries, all color-blind (s=0 Fourier sector). The precise missing ingredient: a
    **non-central relative TWIST** between the enantiomers (a C₃-breaking inter-enantiomer orientation); the
    object forces that twist to **zero**. The boundary is genuine.

---

## Bottom line

From **{D, srs, MDL}** alone, the following emerge as **forced structure** (no fitting, no knobs):

| Standard-Model + gravity feature | status |
|---|---|
| 3 space dimensions + intrinsic time | **forced** (d = b₁ = 3; III₁ modular time from D) |
| gravity: Λ + Einstein–Hilbert + higher curvature | **forced** (spectral action; coeffs = walk counts) |
| chiral fermions, KO-dimension 4 | **forced** (Cl(4), quaternionic J) |
| color **SU(3)** | **forced** (M₃ block of ℂ[A₄]) |
| **three generations** | **forced** (C₃ deck order 3 + observer C³ + Γ λ=−1 triple; A₄ 3-irrep realizes it at Γ. NB: P-point A₄ retracted → 2T) |
| **Koide 2/3** (charged, uncolored) | **forced & phase-independent** (Spin^c moduli (4,2,2)) |
| one complex Higgs + EWSB pattern U(1)_A→1 | **forced** |
| the vacuum (inversion; φ∈{0,π}) | **forced by MDL** |
| mixing: 3 angles + 1 CP phase; CKM vs PMNS contrast | **forced (structure)** |
| no continuous free parameters | **forced (MDL theorem)** |

**Dimensionless inputs: none** (the one dimensionful scale is a unit/ruler, not a parameter; III₁ is
scale-free). **Open frontiers — corrected 2026-06-22 (combined math+physics; see
an internal working note):** frontier (a) [electroweak SU(2)] is **CLOSED** (the doubling-ℍ
= Pati-Salam SU(2)_L, theorem-grade; §3). What remains is **one** frontier: a **C₃-breaking, channel-
distinguishing datum** the bare/doubled object + the run force to **zero**. The *same* datum supplies BOTH
the colored-sector differentiation AND the CKM/PMNS off-diagonal mixing — they are one boundary, not two.
The off-diagonal of the persistence-overlap is **forced trivial (V(s)=I)** because the run advances along the
C₃-fixed axis (both channels co-diagonalize under C₃; `bridge/offdiag_interchannel_mixing.py`), so the
mixing is *not* in the bare run. It lives in the **observer's internal labeling (A5 / the Cl(6) Hamming-weight
species map)** — the physics' matching CKM elements are combinatorial label-readings, not a diagonalization.
The standing boundary is therefore: derive the C₃-breaking datum from the observer's reading of the full
state (internal Cl(6) labels + the flow), the one place it is not forced to zero. **RESOLVED 2026-06-22
(decisive no-tuning test): the framework is ONE-INPUT.** The run forces V=I (no continuous mixing; run on the
C₃-fixed axis); the internal reading gives only the rigid ℤ₂ conjugation; the iso re-describes but does not
compute the CKM. The continuous mixing angles come from a **separate combinatorial counting structure**
(srs density / Hashimoto BFS / twisted-walker) **selected by A5(b)** — the one irreducible input, theorem-grade
internally but not derived from A1+MDL. **Deriving A5(b) from A1+MDL is the one thing between this framework
and zero inputs.** See an internal working note.

*Derived by seven independent agents, each verified against `srs.py`; consolidated derivations in
`dirac_srs_mdl/`, `matter_bridge/explore_m05–m08`, `time_bridge/explore_t06`, `interaction/explore_i03`.*
