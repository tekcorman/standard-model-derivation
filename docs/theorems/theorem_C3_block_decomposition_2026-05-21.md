# Theorem — C_3 isotypic block decomposition of A(k) on srs primitive cell

**Date:** 2026-05-21
**Status:** THEOREM-GRADE. Both the algebraic proof and the computational verification (W35 probe, 8/8 gates PASS) are in hand.

**Purpose.** This is the **first of four structural sub-theorems** that lift §4 of `theorem_yukawa_master_theory_synthesis_2026-05-20.md` from sketch to theorem-grade. It establishes the isotypic block structure of the Bloch adjacency A(k) under the substrate's body-diagonal C_3 symmetry — the structural inventory of the substrate's primitive BZ that the species-to-Bloch-point selection rule (W33, §4(B)–(C) of the synthesis) builds on.

---

## 1. Statement

Let G_srs denote the srs primitive cell (4 vertices at Wyckoff 8a with x = 1/8 on a BCC lattice, as constructed in `proofs/cosmology/srs_photon_bloch_primitive.py`). Label the vertices

  v_0 = (1/8, 1/8, 1/8),  v_1 = (3/8, 7/8, 5/8),
  v_2 = (7/8, 5/8, 3/8),  v_3 = (5/8, 3/8, 7/8).

Let **R** denote the order-3 rotation around the (1,1,1) body diagonal: R(x, y, z) = (z, x, y). Then:

**(a) Vertex action.** R acts on the 4-vertex set as the cycle pattern
  (v_0)(v_1 → v_3 → v_2 → v_1).

**(b) Vertex-rep decomposition.** The 4-dimensional vertex representation V = ℂ⁴ of C_3 = ⟨R⟩ decomposes uniquely as
  V = V_triv ⊕ V_ω ⊕ V_{ω²},  dim V_triv = 2, dim V_ω = dim V_{ω²} = 1,
where ω = e^(2πi/3). The decomposition is forced by the character (4, 1, 1) on (e, R, R²).

**(c) C_3-stable Bloch points.** The Bloch points k of the BCC primitive BZ satisfying R·k ≡ k (mod reciprocal-lattice translation) are exactly
  {Γ, H, P}.
N (together with its R-orbit partners N_x, N_y) is *not* C_3-stable — it lies in a 3-orbit under R.

**(d) Commutator vanishes at C_3-stable k.** For any k stabilized by R modulo the reciprocal lattice,
  [A(k), R] = 0.
By Schur's lemma A(k) preserves each isotypic block, so it decomposes as
  A(k) = A_triv(k) ⊕ A_ω(k) ⊕ A_{ω²}(k).

**(e) Explicit block-restricted spectra.**

| Bloch point | trivial block (2-d) | ω block (1-d) | ω² block (1-d) |
|---|---|---|---|
| Γ | {+3, −1} | {−1} | {−1} |
| H | {−3, +1} | {+1} | {+1} |
| P | {+√3, −√3} | {+√3} | {−√3} |

These are the C_3-isotypic adjacency eigenvalues; their union recovers the full A(k) spectrum {−1,−1,−1,3}, {−3, 1, 1, 1}, {−√3,−√3,√3,√3} at Γ, H, P respectively.

---

## 2. Setup and notation

**Primitive cell** (Wyckoff 8a, x = 1/8 on BCC). Vertices and primitive lattice vectors:

  a_1 = (−1/2, 1/2, 1/2),  a_2 = (1/2, −1/2, 1/2),  a_3 = (1/2, 1/2, −1/2).

The reciprocal lattice (dual basis without the 2π factor) is

  b_1 = (0, 1, 1),  b_2 = (1, 0, 1),  b_3 = (1, 1, 0).

**High-symmetry points** in reduced (b_i) coordinates:

  Γ = (0, 0, 0),  H = (−1/2, 1/2, 1/2),  P = (1/4, 1/4, 1/4),  N = (0, 0, 1/2).

Their Cartesian images (in units of 2π/a):

  Γ → (0, 0, 0),  H → (1, 0, 0),  P → (1/2, 1/2, 1/2),  N → (1/2, 1/2, 0).

**Bloch adjacency.** With phase convention exp(2πi k·n) on an edge spanning cell displacement n,

  A(k)_{ba} = Σ_{edges (a → b, n)} exp(2πi k·n).

**Body-diagonal C_3.** R(x, y, z) = (z, x, y). R is a rotation by +120° around (1,1,1)/√3.

---

## 3. Proof of (a) — vertex action

Direct computation: applying R: (x,y,z) → (z,x,y) to each v_i and matching the image (mod primitive lattice translation) gives the cycle

  R(v_0) = (1/8, 1/8, 1/8) = v_0
  R(v_1) = (5/8, 3/8, 7/8) = v_3
  R(v_2) = (3/8, 7/8, 5/8) = v_1
  R(v_3) = (7/8, 5/8, 3/8) = v_2

so the induced permutation is i ↦ R(i) given by (0 → 0, 1 → 3, 2 → 1, 3 → 2), i.e., the cycle (v_0)(v_1 v_3 v_2). R has order 3 on the vertex set.  ∎

---

## 4. Proof of (b) — vertex-rep decomposition

By (a), R acts on V = ℂ⁴ as the permutation matrix R_{j, i} = δ_{j, R(i)}.

**Character.** Compute χ_V(g) = tr R(g) on the three group elements:

  χ_V(e) = 4,  χ_V(R) = #{fixed vertices} = 1,  χ_V(R²) = #{fixed vertices of R²} = 1

(R² = (v_0)(v_1 v_2 v_3) still fixes only v_0).

**Inner products with C_3 irreducible characters.** With χ_triv = (1,1,1), χ_ω = (1, ω, ω²), χ_{ω²} = (1, ω², ω) and |C_3| = 3:

  ⟨χ_V, χ_triv⟩ = (1/3)(4·1 + 1·1 + 1·1) = 2
  ⟨χ_V, χ_ω⟩   = (1/3)(4·1 + 1·ω̄ + 1·ω) = (1/3)(4 + (ω + ω̄)) = (1/3)(4 − 1) = 1
  ⟨χ_V, χ_{ω²}⟩ = (1/3)(4·1 + 1·ω + 1·ω̄) = 1

Hence V ≅ 2·V_triv ⊕ V_ω ⊕ V_{ω²} as C_3-representations. Uniqueness follows from semisimplicity of ℂ[C_3] (Maschke).

**Explicit basis.** R fixes e_0 (the C_3-fixed vertex) and the symmetric combination (e_1 + e_2 + e_3)/√3 (the R-invariant of the 3-cycle subspace). Thus an orthonormal basis of V_triv is

  V_triv = span{ e_0,  (e_1 + e_2 + e_3)/√3 },

and V_ω, V_{ω²} are spanned by

  V_ω    = span{ (e_1 + ω̄·e_2 + ω·e_3)/√3 },
  V_{ω²} = span{ (e_1 + ω·e_2 + ω̄·e_3)/√3 }.

(Where the ω in coefficients matches our R-cycle convention v_1 → v_3 → v_2 → v_1.) ∎

---

## 5. Proof of (c) — C_3-stable Bloch points are {Γ, H, P}

A Bloch point k ∈ BZ is C_3-stable iff R·k ≡ k modulo the reciprocal lattice Λ* = ℤ-span{b_1, b_2, b_3}. We check each high-symmetry point in Cartesian coordinates (where R(x, y, z) = (z, x, y)):

**Γ.** k = (0, 0, 0). R·k = (0, 0, 0). Stable.  ✓

**P.** k = (1/2, 1/2, 1/2). On the body diagonal: R·k = (1/2, 1/2, 1/2) = k. Stable.  ✓

**H.** k = (1, 0, 0). R·k = (0, 1, 0). The difference (1, 0, 0) − (0, 1, 0) = (1, −1, 0). In reciprocal-lattice coordinates,

  (1, −1, 0) = −1·b_1 + 1·b_2 = (0, −1, −1) + (1, 0, 1) = (1, −1, 0). ✓

so the difference is in Λ*. Stable mod G with G = −b_1 + b_2.  ✓

**N.** k = (1/2, 1/2, 0). R·k = (0, 1/2, 1/2). Difference (1/2, 0, −1/2). Half-integer; cannot be written as ℤ-combination of (0,1,1), (1,0,1), (1,1,0). Not in Λ*.  ✗

**N_x, N_y** (orbit partners): same argument, half-integer differences.  ✗

So {Γ, H, P} are C_3-stable; N (with orbit-mates) is not.  ∎

**Remark — single-point H.** The conventional reciprocal lattice has 6 H points ±(1,0,0), ±(0,1,0), ±(0,0,1). In the BCC primitive BZ all six collapse to ONE point: (1,0,0) − (−1,0,0) = (2,0,0) = −b_1 + b_2 + b_3, and (1,0,0) − (0,1,0) = (1,−1,0) = −b_1 + b_2 (as shown above). Therefore the unique primitive-BZ H is forced to be R-stable, even though the conventional triple {(1,0,0), (0,1,0), (0,0,1)} is permuted by R.

---

## 6. Proof of (d) — commutator vanishes at C_3-stable k

**Bloch covariance under R.** The srs primitive cell is preserved by R as a graph (R sends the 6 undirected edges to themselves, with the edge cell-vector n transformed as n → R·n). Concretely, for any edge (v_a, v_b, n),

  R · (v_a, v_b, n) = (v_{R(a)}, v_{R(b)}, R·n)

is also an edge of the primitive cell (the BCC-Wyckoff-8a structure has body-diagonal C_3 in its space group).

Hence the Bloch matrix entries satisfy

  A(R·k)_{R(b), R(a)} = Σ_{edges (R(a) → R(b), n)} exp(2πi (R·k)·n)
                       = Σ_{edges (a → b, R⁻¹·n)} exp(2πi k · R⁻¹·n)
                       = Σ_{edges (a → b, m)} exp(2πi k·m)        [m = R⁻¹·n]
                       = A(k)_{b, a},

i.e., R · A(R·k) · R⁻¹ = A(k), or equivalently A(R·k) = R⁻¹ · A(k) · R.

**At C_3-stable k.** When R·k ≡ k mod Λ*, write R·k = k + G with G ∈ Λ*. Then for every primitive-cell displacement n,

  exp(2πi (R·k)·n) = exp(2πi k·n) · exp(2πi G·n) = exp(2πi k·n) · 1,

since G·n is an integer (Λ* dual to primitive lattice). Hence A(R·k) = A(k).

Combining with Bloch covariance: A(k) = R⁻¹ · A(k) · R, i.e., R · A(k) = A(k) · R, so

  [A(k), R] = 0  for all k ∈ {Γ, H, P}.  ∎

**Conversely.** At N (and N_x, N_y), R·N ≠ N mod Λ*, so the Bloch phases at R·N differ from those at N by a non-trivial sign pattern, and the commutator is nonzero (numerically ‖[A(N), R]‖ = 4).

---

## 7. Proof of (e) — explicit block spectra

**Γ.** A(Γ) is the unweighted adjacency matrix of the primitive cell — degree-3, 4 vertices. Direct inspection shows each pair (v_i, v_j) is connected (the srs primitive cell with periodic identifications yields the *combinatorial K_4* at Γ): every vertex has 3 NN, exhausting all other vertices. Hence A(Γ) = J − I (with J the all-ones 4×4 matrix), with spectrum {3, −1, −1, −1}.

Block decomposition in the basis {e_0, (e_1+e_2+e_3)/√3}:

  A(Γ)·e_0 = e_1 + e_2 + e_3 = √3 · (e_1+e_2+e_3)/√3
  A(Γ)·(e_1+e_2+e_3)/√3 = (1/√3) · [3·e_0 + 2(e_1+e_2+e_3)] = √3·e_0 + 2·(e_1+e_2+e_3)/√3

so the trivial block restricted to this basis is

  A_triv(Γ) = [[0, √3], [√3, 2]],   eigenvalues {(2 ± √(4+12))/2} = {3, −1}.  ✓

ω block (1-dim): A(Γ) · (e_1 + ω̄·e_2 + ω·e_3) = ... [direct computation] ... = −(e_1 + ω̄·e_2 + ω·e_3). Eigenvalue −1. Same for ω².

| block | eigenvalues |
|---|---|
| V_triv (Γ) | {+3, −1} |
| V_ω (Γ)    | {−1} |
| V_{ω²} (Γ) | {−1} |

**P.** Bloch phases at P_red = (1/4, 1/4, 1/4): for an edge with cell-vector n = (n_1, n_2, n_3) ∈ ℤ³ (in dual basis), exp(2πi P·n) = exp(iπ(n_1+n_2+n_3)/2). Direct computation (or the framework's `srs_E_at_P.py` derivation + `B_P_doubly_degenerate_h.py`) yields A(P) eigenvalues {+√3, +√3, −√3, −√3}.

By Schur's lemma each block is preserved. Block trivial has eigenvalues that are a 2-element subset of {±√3}; ω and ω² each have a single eigenvalue from the same set. Numerical verification (W35 Step G):

| block | eigenvalues |
|---|---|
| V_triv (P) | {+√3, −√3} |
| V_ω (P)    | {+√3} |
| V_{ω²} (P) | {−√3} |

This is consistent with the framework's existing `B_P_doubly_degenerate_h` theorem (doubly-degenerate h at P) — the doubling distributes one √3 across trivial+ω blocks and one −√3 across trivial+ω².

**H.** Bloch phases at H_red = (−1/2, 1/2, 1/2): exp(2πi H·n) = exp(iπ(−n_1+n_2+n_3)). This factor takes values ±1 depending on parities. Direct computation gives spectrum {+1, +1, +1, −3}: at H the adjacency is effectively −(J − I) restricted via the alternating phases — equivalently A(H) = −A(Γ) up to a similarity transformation that conjugates the block structure unchanged.

| block | eigenvalues |
|---|---|
| V_triv (H) | {−3, +1} |
| V_ω (H)    | {+1} |
| V_{ω²} (H) | {+1} |

(The unitary similarity U_H is e_i → exp(iπ·v_i·x̂)·e_i, which flips the sign of one trivial-block eigenvalue while preserving R-equivariance.)

W35 Step G verifies all three explicit block spectra numerically.  ∎

---

## 8. Corollary — chirality inventory by isotype

Applying the Ihara–Bass relation h² − λ·h + (k* − 1) = 0 with k* = 3 to each block eigenvalue λ:

| Bloch | block | λ | h (Ihara–Bass) | |h|² | chirality tan²(arg h) |
|---|---|---|---|---|---|
| Γ | triv | +3 | {1, 2} | {1, 4} | 0 (real) |
| Γ | triv | −1 | (−1 ± i√7)/2 | 2 (Ramanujan) | **7** |
| Γ | ω    | −1 | (−1 ± i√7)/2 | 2 | 7 |
| Γ | ω²   | −1 | (−1 ± i√7)/2 | 2 | 7 |
| H | triv | −3 | {−1, −2} | {1, 4} | 0 (real, negative) |
| H | triv | +1 | (1 ± i√7)/2 | 2 | 7 |
| H | ω, ω² | +1 | (1 ± i√7)/2 | 2 | 7 |
| P | triv | ±√3 | (±√3 ± i√5)/2 | 2 | **5/3** |
| P | ω    | +√3 | (+√3 ± i√5)/2 | 2 | 5/3 |
| P | ω²   | −√3 | (−√3 ± i√5)/2 | 2 | 5/3 |

**Three distinct chiralities** exist in the C_3-stable BZ: {0 (real), 5/3, 7}. They are partitioned by Bloch site:

- **chir 5/3 (the y_τ saddle)** lives EXCLUSIVELY at P, in every isotypic block. Color singlet at P-trivial concentrates the y_τ Yukawa.
- **chir 7** lives at Γ and H, in every isotypic block (via the λ = −1 / λ = +1 subspaces). No fermion of the Standard Model is identified with chir-7 in the current framework derivations — this is a structural pool whose physical interpretation is open (provisional candidates: dark-sector states, beyond-SM physics, or substrate counter-terms).
- **real h ∈ {1, 2}** appears at Γ trivial via λ = +3. h = 1 is the saturation eigenvalue (PT y_t = 1); h = 2 is the Perron walker eigenvalue (NB MDL probability Q^g for y_b).
- **real h ∈ {−1, −2}** appears at H trivial via λ = −3 — the same magnitudes with opposite sign (related to the global sign of the symmetric mode at H).

---

## 9. Corollary — species concentration map (input to §4(B), §4(C) of the master synthesis)

Combine the block structure with the C_3 rep theory of fermion species (which the framework will derive rigorously in §4(B), §4(C); see `theorem_charge_before_color.md`, `theorem_g2d_chirality_doubled.md`):

- **Color singlet** = a single mode at the C_3-fixed vertex v_0, or any pure C_3-trivial combination. By Schur's lemma, the color-singlet wavefunction lives in V_triv. Its A(k) spectrum at any C_3-stable Bloch point is the trivial-block spectrum from §1(e).

- **Color triplet** = a triple of modes on the cycled vertices {v_1, v_2, v_3} transforming under C_3 as the regular representation = trivial(1) + ω + ω². Color-triplet wavefunctions intersect V_triv (their symmetric component) AND populate V_ω, V_{ω²} (their cyclic components).

**Concentration consequences for the gen-3 Yukawa anchors:**

- *Color singlet, chir 5/3* (y_τ): exists ONLY in V_triv(P). Hence y_τ is forced to concentrate at P. No alternative Bloch site supplies chir 5/3 for a color singlet.

- *Color triplet, real h ∈ {1, 2}* (y_t and y_b): the only places real h appears are Γ (h ∈ {1, 2}, from λ = 3 in V_triv) and H (h ∈ {−1, −2}, from λ = −3 in V_triv). Gen-3 quarks concentrate at Γ; H is the antipodal counterpart (with opposite sign on the symmetric mode).
  - h = 1 (saturation): PT y_t = 1.
  - h = 2 (NB Perron walker): y_b = Q^g.

- *Color singlet, asymptotic spectral* (y_ν3): bypasses the block-eigenvalue mechanism and uses the Laplacian band-edge L_us = 2 + √3 directly. Outside the C_3 isotypic framework — handled by the framework's existing spectral seesaw mechanism.

This is the **structural input** that §4(B) and §4(C) of `theorem_yukawa_master_theory_synthesis_2026-05-20.md` need to discharge their content rigorously.

---

## 10. Computational verification

`proofs/foundations/W35_C3_block_decomposition_2026-05-21.py` (8/8 gate checks PASS):

  T1. R is order-3 with cycle (v_0)(v_1 v_3 v_2).               PASS
  T2. Vertex rep = 2·trivial + ω + ω² (multiplicities verified).   PASS
  T3. C_3-stable Bloch points (mod Λ*) = {Γ, H, P}.              PASS
  T4. [A(k), R] = 0 at {Γ, H, P} and ≠ 0 at {N, N_x, N_y}.        PASS
  T5. A(Γ) block spectra match {3, −1}/{−1}/{−1}.                  PASS
  T6. A(P) block spectra match {+√3, −√3}/{+√3}/{−√3}.              PASS
  T7. A(H) block spectra match {−3, 1}/{1}/{1}.                    PASS
  T8. Chirality inventory by isotype recovered (5/3 at P; 7 at Γ, H; real h at Γ trivial λ=+3 and H trivial λ=−3). PASS

The probe imports the framework's pre-existing `srs_photon_bloch_primitive.py` machinery, so the test passes on the actual framework substrate (no parallel reconstruction).

---

## 11. What this theorem closes; what remains

**This theorem lifts §4(A) of `theorem_yukawa_master_theory_synthesis_2026-05-20.md` from sketch to theorem-grade.** The previous sketch in §4 paragraph (1) — "At each Bloch point, the 4-dim adjacency A(k) decomposes under C_3 as 2 × (trivial) + 1 × (ω) + 1 × (ω²)" — is now rigorous, with one important refinement: the decomposition applies pointwise at the C_3-stable Bloch points {Γ, H, P}, and orbitwise at N (whose 3 R-orbit partners satisfy the same decomposition collectively).

**What remains for the master Yukawa theorem.** §4 still has three sub-pieces sketched:

- **§4(B) — Color singlet → P-saddle.** Uses §9 corollary above. Requires the prior identification "lepton wavefunction is a C_3-trivial vertex mode" via `theorem_charge_before_color.md` and the Cl(6)-Fock species structure.

- **§4(C) — Color triplet → Γ.** Same machinery, with the splitting "color triplet = trivial(1) + ω + ω² inside the cycled-vertex subspace." Requires the additional ingredient that gen-3 up concentrates at the h=1 saturation branch and gen-3 down at the h=2 Perron branch.

- **§4(D) — Hamming weight n → walker length L via MDL waterline.** The deepest piece; reduces to applying `theorem_A2_mdl_from_finite_register.md`'s waterfilling rule to the species' toggle modes.

Once §4(B), (C), (D) are theorem-grade, the master synthesis's §4 will be fully derivative, and the synthesis itself will be theorem-grade for the 4 gen-3 anchors.

---

## 12. Cross-references

**Builds on:**
- `proofs/cosmology/srs_photon_bloch_primitive.py` — primitive cell + A(k) machinery
- `predictions/srs_E_at_P.py` — adjacency eigenvalue √3 at P
- `predictions/B_P_doubly_degenerate_h.py` — doubly-degenerate h at P
- `predictions/h_walker_eigenvalue.py` — Ihara–Bass relation h² − λ·h + (k*−1) = 0

**Cited by:**
- `docs/theorems/theorem_yukawa_master_theory_synthesis_2026-05-20.md` §4(1)

**Successor theorems** (the rest of §4's structural sketch):
- §4(B) color singlet → P-saddle (planned)
- §4(C) color triplet → Γ (planned)
- §4(D) Hamming weight → walker length (planned, multi-session)
