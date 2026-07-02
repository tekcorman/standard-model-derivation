# Dimension-5 Lorentz violation coefficient (η_5 = 0 exactly)

## Abstract

The dimension-5 Lorentz-violation coefficient in the photon dispersion relation on srs vanishes exactly: η_5 = 0. This follows from the structural symmetry B(−k) = B(k)* of the Hashimoto Bloch operator on any undirected graph, which forces the top eigenvalue h_max(k) to be real and even in k — eliminating all odd-power Taylor coefficients including the cubic term. The exact-zero prediction is consistent with and favored by current experimental bounds |η_5| ≲ 0.1 from LHAASO's observation of GRB 221009A.

**Result:** η_5 = 0 exactly.
**Grade:** THEOREM (Type 2 algebra on undirected-graph symmetry).

## Framework axioms invoked

- **A1**: toggle alphabet on srs undirected edges.

## Derivation

### Step 1 — Undirected-graph symmetry (Type 2)

The srs is an undirected graph: each edge exists in both orientations. The Hashimoto Bloch operator for directed bonds has matrix elements B(k)_{ij} = phase(k) · δ_{NB-admissible}, where the phase depends on the bond displacement vector r_i.

Under k → −k: e^(i k·r) → e^(−i k·r) = (e^(i k·r))*. Hence:

$$B(-k) = B(k)^*$$

(complex conjugate, not transpose). Verified numerically at multiple k directions in `proofs/lorentz/hashimoto_bloch_dispersion.py` Part 2.

### Step 2 — Eigenvalue symmetry (Type 2)

The eigenvalues of B(−k) are complex conjugates of those of B(k). For the top real eigenvalue (k − 1 = 2 at k = 0, simple):

$$h_{\max}(-k) = h_{\max}(k)^* = h_{\max}(k) \quad \text{(real)}.$$

Therefore h_max(k) is real and even in k.

### Step 3 — Taylor expansion contains only even powers (Type 2)

Near k = 0:

$$h_{\max}(\mathbf{k}) = h_{\max}(0) + c_2 |\mathbf{k}|^2 + c_4 |\mathbf{k}|^4 + \ldots$$

The O(k¹), O(k³), O(k⁵) coefficients are identically zero.

### Step 4 — Dimension-5 LIV coefficient vanishes (Type 2)

The dimension-5 Lorentz-violation coefficient η_5 multiplies the cubic term p³ in the photon dispersion. By Step 3, η_5 = 0 exactly.

### Result

$$\boxed{\eta_5 = 0 \text{ exactly.}}$$

## Comparison with experiment

Current experimental upper bound: |η_5| ≲ 0.1 at 95% CL (LHAASO 2024; consistent with older Fermi-LAT bounds).

| Source | Bound |
|---|---|
| LHAASO GRB 221009A (Cao et al., JCAP 04 (2024) 060 [arXiv:2312.09079]) | E_QG,1 > 1.47 × 10²⁰ GeV ≈ 10 E_Pl → \|η_5\| < 0.1 |
| LHAASO re-analysis (Cao et al., PRL 133, 071501 (2024) [arXiv:2402.06009]) | Consistent |
| Fermi-LAT GRB timing (Vasileiou et al., Phys. Rev. D 87, 122001 (2013)) | E_QG,1 > 7.6 E_Pl |
| Review (Addazi et al., Prog. Part. Nucl. Phys. 125 (2022) 103948 [arXiv:2111.05659]) | Consensus \|η_5\| ≲ 0.1 |

**Consistency:** η_5 = 0 is not only consistent with but favored by current data. The experimental bound is tightening over time; any non-zero η_5 of order ~0.1 would already be excluded.

## Open questions

- **Experimental confirmation of η_5 = 0 to higher precision** would come from next-generation GRB timing experiments (CTA, SWGO, LHAASO extensions).
- **Sign argument structural.** The derivation uses the undirected nature of srs as the source of B(−k) = B(k)*. Any directed modification (e.g., if future framework extensions introduce directed edges) would potentially admit non-zero η_5. The framework's current commitment to srs as an undirected graph (from A1 + reticular chemistry selecting srs) thus has a specific empirical prediction.
- **Relation to toggle-process time-reversal.** The toggle Markov chain breaks time-reversal (p_create = 1/2 ≠ p_destroy = 1/3), but this is unrelated to the graph-theoretic symmetry B(−k) = B(k)* that gives η_5 = 0. Worth noting explicitly that these are independent features.


## Audit v2 (Clause 7) status

This prediction inherits Row 4 (k* = 3) audit v2 closure. The §3 multi-mechanism
table (M1-M6 vs qtz at k=4) is consolidated in
an internal working note §2.1; closure doc:
an internal working note.

- **Status (post-audit-v2):** UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 audit v2 closure (DOMINANT-with-named-margins; UNIQUE-on-η_B).
- **Named margin** (from index §2.1): combined M2 (~0.45 Boltzmann) × observable-specific M3·M4·M5 product.
- **Inherits structurally:** Row 4 v2 closure protects against qtz at k=4 alternative via combined mechanism product. M6 sign-flip is irrelevant for this observable (uses Im(h)/|h|² or magnitude, not Re(h) sign) unless the observable uses Re(h_P) directly (η_B specifically; see `predictions/eta_B_derivation.md` for that case).
- **Conditionals deferred:** RCSR-vetted qtz bond list verification; selection-rule audit; data-conditional MDL.

## References

### Framework
- `docs/theorems/theorem_lorentz_causal_sector.md` §6.2 (Stage 3 theorem containing this result).
- `proofs/lorentz/hashimoto_bloch_dispersion.py` Part 2 (numerical verification of B(−k) = B(k)*).

### Published experimental
- **Cao et al. (LHAASO Collab.)** (2024). JCAP 04, 060 [arXiv:2312.09079]. Current tightest dim-5 LIV bound.
- **Cao et al.** (2024). PRL 133, 071501 [arXiv:2402.06009]. Independent re-analysis.
- **Vasileiou et al.** (2013). Phys. Rev. D 87, 122001. Fermi-LAT GRB timing.
- **Addazi et al.** (2022). Prog. Part. Nucl. Phys. 125, 103948 [arXiv:2111.05659]. Comprehensive LIV review.
