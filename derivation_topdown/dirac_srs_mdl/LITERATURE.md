# Where this clean room sits in the literature

The object — the maximal abelian (ℤ³) cover of K₄ — is canonical and much-studied:
**Sunada's K₄ crystal = srs net = (10,3)-a (Wells) = Laves graph (Coxeter) = gyroid labyrinth = SrSi₂ net.**

## Confirmed — our computations reproduce established theory

- **MDL → K₄ = Sunada's strong-isotropy theorem.** In 3D there are *exactly two* strongly isotropic
  crystal nets: diamond (4-regular) and srs/K₄ (3-regular). Our `V(k−2)=4 ⇒ unique (k=3,V=4)`
  (explore_05) is the combinatorial core. The Kotani–Sunada **"standard realization" is defined by
  energy minimization (harmonic maps)** — the MDL/minimization framing is the *established*
  characterization of this net, not an imposition.
  [Sunada, *Notices AMS* 55 (2008) 208; Kotani–Sunada, *Trans. AMS* 353 (2001) 1; Sunada,
  *Topological Crystallography*, Springer 2013.]
- **Ihara zeta + Ramanujan.** Known closed form
  `ζ_{K₄}(u)⁻¹ = (1−u²)² (1+u+2u²)³ (1−3u+2u²)`.
  Matches our B(Γ) spectrum exactly: `(1+u+2u²)³` = the Ramanujan pair at multiplicity 3 (= the two
  3-irreps); `(1−u)(1−2u)` = the Perron h∈{1,2}; `(1−u²)²` = h=±1 (the 1′⊕1″ at h=−1). K₄ is a known
  Ramanujan graph. [Terras, *Zeta Functions of Graphs*, Cambridge 2010.]
- **A₄ / T₂ at Γ, H.** The 3-fold T₂ (= A₄ 3-irrep) degeneracy at Γ/H is documented in the
  tight-binding band structure — exactly our 3-irrep. [Hatsugai/Mizoguchi et al., *Phys. Rev. B* 94,
  195426 (2016), arXiv:1609.09762.]
- **Chirality, girth 10.** srs is chiral (enantiomers srs, srs*); girth 10.
  [Coxeter, "On Laves' graph of girth ten," *Canad. J. Math.* 7 (1955) 18.]

## Novel here — not found in the literature (per a 2026 search)

1. The **Ihara zeta of the COVER, across the Brillouin zone** (explore_04, 11): the graph-RH and the
   Stark–Terras functional equation Λ(u)=Λ(1/(qu)) holding at *every* k of the periodic net, and the
   explicit identification **Bloch decomposition = Artin–Ihara L-factorization over ℤ³** (the Bloch
   phase IS the deck character χ_k; ζ(u,k)⁻¹ = twisted L_{K₄}(u,χ_k)⁻¹, factorization checked on
   built (ℤ_N)³ covers). The *theory* is standard Stark–Terras (Adv. Math. 121 (1996) 124; Terras 2010,
   Thm 10.4 & Ch. 18–21); the explicit srs/K₄-cover instantiation across the BZ is the contribution.
2. The **full-BZ A₄ representation decomposition of the directed-edge space** (explore_02):
   `1 ⊕ 1′ ⊕ 1″ ⊕ 3·3`. This gives the published zeta exponents {2,3,1} their representation-theoretic
   meaning — a reading the literature has only implicitly (special-point T₂ only).
3. The **C₃ triality** of the Ramanujan shell (explore_03) — not published. (Currently Γ-only.)

## Caveat

Space-group setting: **I4₁32** (No. 214) for the maximal standard realization; **P4₁32 / P4₃32**
(Nos. 213/212) for the single-net SrSi₂ / RCSR enantiomorph setting. Confirm the exact symbol against
RCSR (symbol "srs") for any anchored statement.
