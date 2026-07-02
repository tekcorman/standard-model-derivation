# The full mathematical structure of {Dirac operator, srs, MDL}

The object is **Sunada's K₄ crystal** (= srs = (10,3)-a = Laves = gyroid labyrinth). Below is the
structure that flows, organized by layer, with the script that computes each. Pure math; no physics.
Established-in-literature vs novel-here is marked per `LITERATURE.md`.

## 0. Foundation — MDL forces the substrate            [explore_05]
- K₄ is the **unique simple regular graph whose maximal abelian cover is 3-dimensional**
  (b₁=3 ⇒ E=V+2; k-regular ⇒ E=kV/2 ⇒ V(k−2)=4 ⇒ only (k=3,V=4)).
- = Sunada's strong-isotropy uniqueness (diamond + srs are the only two 3D strongly-isotropic nets);
  the Kotani–Sunada "standard realization" is defined by **harmonic-map energy minimization** (the MDL).

## 1. The object                                        [explore_01, 10]
- srs = maximal abelian **ℤ³ cover of K₄**. Aut(K₄)=S₄ (24); rotations **A₄** (12).
- **Chiral**: no orientation-reversing symmetry (verified over 24 perms × 48 signed lattice maps).
  srs ≠ srs*. (A(−k)=conj A(k) is time-reversal, not a spatial symmetry.)
- **Girth 10**; 15 shortest cycles through a vertex = **5 free C₃-triples** (radius-stable).

## 2. Spectral structure (the Dirac operator)           [explore_01, 02, 07]
- Hodge–Dirac D=[[0,∂],[∂*,0]], **D²=graph Laplacian**.
- **Spectral dimension d = 3** (scalar-Laplacian DOS, N(<E)~E^{d/2}).
- **ζ_D(0) = 8 exactly** = number of nonzero modes per cell (Mellin-robust, integer).
- **Zero modes = cohomology**: Γ ⇒ b₀+b₁ = 1+3 = 4; P,H ⇒ b₁ = 2; generic k ⇒ 2.
- Heat trace is bounded (no continuum Weyl power); analytic moments {a₀,a₁,a₂}={10,24,48}.

## 3. Non-backtracking operator, zeta, number theory    [explore_01, 04, 11]
- Ihara–Bass **h²−λh+(k−1)=0**; the non-trivial spectrum sits on the **Ramanujan shell |h|²=2**.
- Ihara zeta (finite K₄): ζ⁻¹=(1−u²)²(1+u+2u²)³(1−3u+2u²); **cover zeta computed across the BZ**.
- **Graph Riemann hypothesis** (exact, every k): non-trivial zeros on |u|=1/√(k−1)=1/√2
  (⇔ |λ|<2√2, the Alon–Boppana/Ramanujan bound; product-of-roots = 1/q).
- **Functional equation** (Stark–Terras), exponents all = r−1 = b₁−1 = 2:
  **Λ(u) = (1−u²)²(1−qu²)²u² ζ(u) = Λ(1/(qu))**; self-dual ('central') radius |u|=1/√q=1/√2
  coincides with the graph-RH critical circle. Holds at **every Bloch fiber k** (same per-cell
  combinatorics); time reversal A(−k)=conj A(k) relates k↔−k separately.
- **Bloch decomposition = Artin–Ihara factorization** over the deck group ℤ³: the Bloch parameter
  IS the character χ_k(n)=e^{2πik·n}, and **ζ(u,k)⁻¹ = L_{K₄}(u,χ_k)⁻¹ = det(I−uB(k))** (twisted
  Ihara–Bass, edge form = vertex/Bass form, verified to 1e−16). For a finite (ℤ_N)³ cover Y,
  **ζ_Y(u)⁻¹ = ∏_χ L_{K₄}(u,χ)⁻¹** verified by building Y explicitly (N=2: 32 verts; N=3: 108 verts).
- **Cover zeta = a Mahler measure** [explore_15]: per cell, log ζ_cover(u)⁻¹ = 2 log(1−u²) + M(u),
  M(u)=∫_{T³} log|det(I−uA(k)+2u²I)| dk. At u=1 (where I−A+2I = 3I−A = the Bloch Laplacian) this is the
  **srs spanning-tree entropy h = ∫ log det(3I−A) dk = 3.32861/cell = 0.83215/vertex** (converged ~6
  digits; ℤ²-calibrated). A genuine *new* lattice constant, but **no elementary closed form** (typical
  of 3D lattice constants — they need the integral / Watson-type representation).

## 4. Representation structure (A₄ on the spectrum)      [explore_02, 06, 08, 10]
- vertices = 1⊕3; edges = 1⊕1′⊕1″⊕3; **darts = 1⊕1′⊕1″⊕3·3** (three copies of the 3-irrep).
- B(Γ): the **Ramanujan shell |h|²=2 = exactly two copies of the 3-irrep**; Perron h=2 = 1; h=±1 = 1′⊕1″⊕3.
- **C₃ triality** (fixed line k=t(1,−1,1)): the Ramanujan shell distributes **2–2–2 across {1,ω,ω̄}**,
  and is **rigid** along the whole line (|h|²=2 never moves; only tree roots move).
- **Commutant of the A₄-action on the dart space = ℂ ⊕ ℂ ⊕ ℂ ⊕ M₃(ℂ)** (Wedderburn; center dim 4).
  The **M₃(ℂ)** is the endomorphism algebra of the three-fold multiplicity space of the 3-irrep.
- Connectivity: the Ramanujan 6-shell is rigid Γ→P; the tree modes merge onto it at P (8-fold |h|²=2).

## 5. Topological structure (band topology)              [explore_09]
- The Bloch adjacency is a **nodal (Weyl) band structure** with quantized, r/N-independent charges:
  - **charge-2 (double-Weyl) Berry monopoles at Γ and H** (carried by the e=∓1 triplets; the isolated
    e=±3 singlet is trivial);
  - **charge-1 Weyl nodes at the P-type points** on the kz=1/4 plane (kₓkᵧ-slice Chern jumps +1↔0 there);
  - P=(¼,¼,¼) itself topologically trivial; total Chern over all bands = 0.

## 6. Algebraic structure                                [explore_10]
- **Spectral-triple Clifford grading** {D,G}=0 (G = +I on vertices, −I on edges) — flows un-imposed
  from the bipartite vertex/edge complex. A second ℤ₂: dart-reversal J, J²=I.
- The genuine algebra organizing the three 3-irreps is **M₃(ℂ)** (above) — NOT a full Clifford algebra
  on the dart space (honest scope).

## 7. Dynamics                                          [explore_14]
- The **non-backtracking walk is the (geodesic) flow**; topological entropy h = log(k−1) = log 2
  (= log spectral radius of B). Closed-orbit counts N_m = Tr Bᵐ (K₄ quotient: 0,0,24,24,0,96,168,…).
- **The Ihara zeta IS the Ruelle dynamical zeta of this flow**: det(I−uB) = exp(−Σ_m N_m uᵐ/m)
  (verified to machine precision), Euler product over PRIME CYCLES = primitive periodic orbits. The
  static spectrum (the Ramanujan shell |h|²=k−1) is the flow's **resonance spectrum**.
- **Continuous flow**: the spectral triple's Dirac flow e^{itD} and heat flow e^{−tD²} (D² = Laplacian).
- **The arrow, concretely** [explore_16]: from a localized (low-entropy) initial state on a finite
  cover patch, the heat-flow Shannon entropy rises **monotonically 0 → log(nv)** (the arrow of time);
  spreading is **diffusive ⟨r²⟩ ~ t (fit 1.02)**. The wave/Dirac flow is unitary and **ballistic,
  ⟨r²⟩ ~ t^(≈2)** (the geodesic light-cone; small-patch onset pushes the fit to ~2.8). The **law is
  forced** (D is the generator); a *history* needs only an **initial condition**, exactly like a PDE —
  the arrow is the low-entropy initial data, NOT an observer. Equilibrium (max-entropy/uniform) is the
  static fixed point.
- **Time itself vs. initial data — honest distinction**: deriving *time itself* from the state
  (Connes–Rovelli modular/thermal time) is a separate, stronger question, arising only under full
  background-independence (no external t). The natural ℤ³ trace is tracial ⇒ trivial modular flow, so
  that would require a non-tracial state. For the dynamics *as a PDE*, none is required.

## 8. Standard-realization geometry                     [explore_12]
- Harmonic (Kotani–Sunada) embedding: cell vertices at {0, ±¼} fractional; the **Albanese lattice is bcc**
  (period Gram ratio 3); **all bonds equal length, all bond angles exactly 120° (cos = −½)** — one vertex
  orbit, one edge orbit (strong isotropy, geometric form).
- The C₃ generator realizes as a **⟨111⟩ 3-fold screw** (pitch 0.408, axial period √(3/2)) — a genuine
  helix; chiral. srs is the gyroid TPMS labyrinth graph; net chirality = gyroid chirality.

## 9. K-homology / index of the spectral triple         [explore_13]
- **Analytic index = Euler characteristic χ = V−E = 4−6 = −2**, k-INDEPENDENT (McKean–Singer; verified
  three ways: SVD rank, kernel supertrace, heat supertrace). The index −2 is the K-homology pairing [D]·[1].
- **Index rigidity / spectral flow**: b₀−b₁ = −2 everywhere; the unsigned b₀+b₁ jumps 2→4 only at Γ
  (extra harmonic 0- and 1-forms enter as an index-preserving pair).
- The Bloch-bundle Chern data reproduces §5 (charge-2 at Γ/H, charge-1 at P; total 0) — the bundle-level
  shadow of index rigidity (Poincaré–Hopf / Nielsen–Ninomiya).

## Novelty (per a 2026 literature search — see LITERATURE.md)
- Established: the net's identity/uniqueness, the finite-K₄ Ihara zeta + Ramanujan, the A₄/T₂ degeneracy
  at Γ/H, chirality, girth 10, and (physics literature) the charge-2 / charge-1 nodal structure.
- Novel here: the **cover's Ihara zeta + graph-RH across the BZ**, the **full-BZ A₄ decomposition of the
  edge space** (giving the zeta exponents {2,3,1} their rep-theoretic meaning), the **C₃ triality**, and
  the **M₃(ℂ) commutant** statement.

## Verification  [verify_pass.py — independent re-derivation, fresh code & different methods]
All load-bearing claims **CONFIRM** under a second independent implementation:
srs = max-abelian-cover-of-K₄ (b₁=3, voltage ℤ³-basis, **girth 10 via fresh BFS**); **index = V−E = −2**
at every k; **ζ_D(0)=8 UPGRADED** — trivially the #nonzero modes/cell (10−2), no regularization needed;
**Weyl charges 2 (Γ), 1 (P-type)** via an *independent* sphere Fukui–Hatsugai (vs explore_09's planar
slices) → −2.000, +1.000; **tree entropy 3.3286** (method calibrated on ℤ²→(4/π)Catalan); **A₄-commutant
dim 12**; chirality (no inversion permutation). NOT independently re-verified: the harmonic-geometry
numbers (rest on an imposed isotropy assumption), the functional-equation exponents (agent self-checked
to 1e−16), the literature-novelty claims (single web search). "Complete" = the *solution manifold* is
complete by Floquet–Bloch; the *invariant map* is thorough, not exhaustive.

## Bucket A — clean-room completions (explore_18-20)
- **K-theory / SPT** [explore_20] — SOLID: the 4-band Bloch bundle over T³ is **stably trivial**
  (K⁰(T³), K¹(T³) classes both 0). Per-band weak first-Chern triples: bands 0,2 = (0,0,0); bands 1,3 =
  (±1,±1,∓1); total = 0 (Nielsen–Ninomiya). Weyl charges 2 (Γ/H), 1 (P-type) re-derived, match
  verify_pass. Second Chern = 0 (dim), Hopf linking = 0. A charge-balanced Weyl structure on a trivial
  total bundle.
- **Spectral statistics & mixing** [explore_19, 21] — RESOLVED: geodesic-flow mixing = the Ramanujan gap
  (non-trivial NB spectrum on |h|=√2, Perron 2 → gap √2/2, optimal expansion). Level statistics, proper
  unfolding: **integrable / Poisson-like** — no level repulsion (P(s<0.1)=0.107 ≈ Poisson 0.10, vs
  Wigner's ~0), variance 1.59 (near Poisson's 1; the earlier 4.43 was a bad unfolding). NOT chaotic, as
  expected for a band crystal. Van Hove energies (band critical points): ±3, ±(1+√2), ±√5, ±√3, ±1, ±(√2−1).
- **Full space group** [explore_18, 21] — RESOLVED: the proper point group is **O (order 24)**, confirmed
  via the SPECTRUM symmetry spec(A(Rk))=spec(A(k)) (which captures the non-symmorphic **4₁ screws** the
  constant-U search missed): proper element orders {1:1, 2:9, 3:8, 4:6} — the six order-4 elements are the
  4-folds A₄ lacks, so it is **O, not A₄** (resolving explore_18's undercount). **Chirality intact**: the
  spec-test also flags 24 improper (det −1) ops, but those are **time-reversal** (A(−k)=conj A(k)=A(k)*,
  antiunitary), NOT spatial mirrors — no spatial improper symmetry, so srs stays chiral (consistent with
  explore_10/verify_pass). [explore_21's raw "achiral" print is the TR artifact, corrected here.]

## Remaining threads
The bare object is now mapped across all ten layers (foundation, object, spectral, zeta/number-theory,
representation, topology, algebra, dynamics, geometry, K-homology). Open refinements only:
- the explicit harmonic-energy derivation of the Albanese metric (here fixed via the isotropy condition)
  — confirmatory only (a known Kotani–Sunada theorem);
- [DONE, explore_15] the cover's BZ-integrated zeta = the Mahler measure M(u); the tree entropy
  h≈3.32861/cell is a new invariant with no elementary closed form — a PSLQ closed-form hunt is possible
  but low expected payoff for a 3D lattice constant;
- the precise non-tracial observer state that would generate a thermal/modular time (the one dynamical
  ingredient shown NOT to be intrinsic to {D, srs, MDL}) — this is the door OUT of the clean room
  (it adds the observer), i.e. the bridge back to the broader project, not an internal refinement.
