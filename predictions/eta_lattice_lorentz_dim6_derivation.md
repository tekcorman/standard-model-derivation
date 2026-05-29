# Dimension-6 Lorentz violation coefficient (η_lattice = 1/12)

## Abstract

The dimension-6 Lorentz violation coefficient for photon dispersion on srs equals η_lattice = 1/12 exactly (CAS-verified to 24+ decimal digits at 500-bit precision). Sign is positive (subluminal: propagation speed decreases at high energy). This follows from high-precision extraction of the Hashimoto Bloch dispersion coefficients D_NB = 1/8 and D4_aniso = 1/768 with η = D4_aniso/D_NB². Current experimental bounds are ~16 orders of magnitude weaker than this prediction; η_lattice = 1/12 is neither confirmed nor excluded and sets a specific future test target at the ~147 PeV scale.

**Result:** η_lattice = 1/12 exactly.
**Grade:** THEOREM-GRADE SYMBOLIC. Closed by the Ihara cross-walker corollary of `predictions/srs_bloch_lv_dim6_derivation.md`: the scalar-Bloch quartic coefficients $D_4^{{\rm iso},H} = -1/1024$ and $D_4^{{\rm aniso},H} = +1/1536$ are derived symbolically (Feshbach-Löwdin partition; `proofs/foundations/lorentz_sig_h_lv_4th_order_symbolic.py`), and the Ihara factorization $u^2 - \lambda u + 2 = 0$ for 3-regular graphs (Ihara 1966 / Stark-Terras 1996) forces $D_{\rm NB} = h'(3) D_H = 1/8$ and $D_4^{{\rm aniso},{\rm NB}} = h'(3) D_4^{{\rm aniso},H} = 1/768$, hence $\eta_{\rm lattice} = 1/12$ as a closed-form rational.

## Framework axioms invoked

- **A1**: toggle alphabet on srs edges.
- **A2 refined**: MDL observer (giving Stage 2a thresholds used for p_create, p_destroy).

## Derivation

### Step 1 — Hashimoto Bloch dispersion setup (Type 2)

The Hashimoto (non-backtracking) Bloch matrix B(k) on the srs primitive cell is a 12×12 matrix parameterized by the reciprocal-space Cartesian k. Its top eigenvalue h_max(k) has Taylor expansion near k = 0:

$$h_{\max}(\mathbf{k}) = 2 - D_{\text{NB}} |\mathbf{k}|^2 - \left[D_{4,\text{iso}} + D_{4,\text{aniso}} \cdot f_4(\hat{\mathbf{k}})\right] |\mathbf{k}|^4 + O(k^6)$$

where f₄(k̂) = k̂_x⁴ + k̂_y⁴ + k̂_z⁴ is the cubic anisotropy invariant.

### Step 2 — Closed form via Ihara cross-walker theorem (Type 2 + 3)

The scalar adjacency Bloch coefficients on srs are theorem-grade symbolic
(`predictions/srs_bloch_lv_dim6.py`, derived via Feshbach-Löwdin partition
in `proofs/foundations/lorentz_sig_h_lv_4th_order_symbolic.py`):

$$
D_H \;=\; \tfrac{1}{16},\qquad D_4^{{\rm iso},H} \;=\; -\tfrac{1}{1024},\qquad D_4^{{\rm aniso},H} \;=\; +\tfrac{1}{1536}.
$$

For 3-regular graphs the **Ihara factorization** (Ihara 1966, Stark–Terras 1996)
relates scalar adjacency eigenvalue $\lambda$ and Hashimoto eigenvalue $u$ via
$u^2 - \lambda u + (k-1) = 0$. For $k=3$ this is $u^2 - \lambda u + 2 = 0$ with
upper root $h(\lambda) = \tfrac12(\lambda + \sqrt{\lambda^2 - 8})$. Direct
differentiation gives $h(3) = 2$, $h'(3) = 2$, $h''(3) = -4$. Substituting the
Taylor expansion $\lambda_0(\mathbf k) - 3 = -D_H k^2 - \alpha^H k^4 + O(k^6)$
into $h(\lambda_0(\mathbf k))$ and matching against
$h_{\max}(\mathbf k) - 2 = -D_{\rm NB} k^2 - \alpha^{\rm NB} k^4 + O(k^6)$ gives:

$$
D_{\rm NB} \;=\; h'(3)\,D_H \;=\; 2 \cdot \tfrac{1}{16} \;=\; \tfrac{1}{8},
$$
$$
D_4^{{\rm aniso},{\rm NB}} \;=\; h'(3)\,D_4^{{\rm aniso},H} \;=\; 2 \cdot \tfrac{1}{1536} \;=\; \tfrac{1}{768},
$$
$$
D_4^{{\rm iso},{\rm NB}} \;=\; h'(3)\,D_4^{{\rm iso},H} - \tfrac{1}{2}\,h''(3)\,D_H^{\,2} \;=\; 2\cdot(-\tfrac{1}{1024}) - \tfrac{1}{2}(-4)(\tfrac{1}{16})^2 \;=\; -\tfrac{1}{512} + \tfrac{1}{128} \;=\; +\tfrac{3}{512}.
$$

This closure is symbolic end-to-end (Ihara as a cited theorem; cross-walker
algebra mechanical). Verified in `proofs/foundations/lorentz_sig_ihara_lv_relation.py`.

The earlier high-precision numerical extraction
(`proofs/lorentz/hashimoto_dispersion_symbolic.py`, 4-point Vandermonde at
500-bit `mpmath`) is now an independent cross-check rather than the primary
source. Result (cross-check `tail` output):

| Coefficient | Extracted value | Symbolic exact | Agreement |
|---|---|---|---|
| D_NB | 0.125000000...0 | 1/8 | 39 digits |
| D4_aniso | 0.00130208333...3 | 1/768 | 25 digits |
| D4_iso | (extracted in `lorentz_sig_hashimoto_d4_iso.py`) | +3/512 | 25+ digits |

### Step 3 — η_lattice = D4_aniso / D_NB² (Type 2)

$$\eta_{\text{lattice}} = \frac{D_{4,\text{aniso}}}{D_{\text{NB}}^2} = \frac{1/768}{(1/8)^2} = \frac{1/768}{1/64} = \frac{64}{768} = \frac{1}{12}.$$

Verified in the symbolic script to 24 decimal digits consistent with 1/12.

### Step 4 — Sign (Type 2)

D4_aniso > 0 (positive; dispersion decreases faster with k than the isotropic approximation). Hence η_lattice > 0, interpreted as subluminal: propagation speed decreases at high energy.

### Result

$$\boxed{\eta_{\text{lattice}} = \frac{1}{12}, \quad \text{subluminal}.}$$

## Comparison with experiment

| Probe | Current bound | η_lattice = 1/12 status |
|---|---|---|
| LHAASO GRB 221009A (propagation test) | E_QG,2 > 7.3 × 10¹¹ GeV → \|η_6\|_prop < ~10¹⁴ | Far below sensitivity |
| Threshold-shift (UHE-photon transparency) | E_QG,2 > ~10¹⁰-10¹¹ GeV → \|η_6\|_thresh < ~10¹⁶-10¹⁸ | Far below sensitivity |
| Anomalous GRB 221009A 18-TeV photon (tentative LIV evidence) | Finke & Razzaque, ApJL 942, L21 (2023) [arXiv:2210.11261] | Sign consistent (subluminal) but value not compatible with 1/12 if taken literally |

**Consistency:** η_lattice = 1/12 is ~16 orders of magnitude below current sensitivity. The prediction is a specific future test target, not currently falsified. LHAASO upgrades and SWGO (2030s) will probe this range.

## Honest scope

**The claim "η_lattice = 1/12 EXACTLY" is theorem-grade symbolic** under the
parameter_linter hard quality gate. The proof chain is:

1. The scalar adjacency Bloch coefficients $D_H$, $D_4^{{\rm iso},H}$, $D_4^{{\rm aniso},H}$
   are derived symbolically by Feshbach-Löwdin partition of $H(\mathbf k)$ relative
   to the Perron eigenstate (`proofs/foundations/lorentz_sig_h_lv_4th_order_symbolic.py`).
2. The Ihara factorization $u^2 - \lambda u + 2 = 0$ for 3-regular graphs (Ihara 1966
   / Stark-Terras 1996) is a cited theorem.
3. The cross-walker algebra (substitute scalar Taylor expansion into $h(\lambda)$
   and match against Hashimoto Taylor expansion) is mechanical algebra.

All three are gate-passing. The earlier 24-digit `mpmath` numerical extraction
serves as an independent cross-check, not the primary source of the rational values.

## Open questions

- **Higher-order dispersion coefficients** (D6, D8). Extracted numerically in
  the script but not interpreted physically.
- **Physical translation from Hashimoto dispersion to photon propagator.** Stage 3
  §6.4 states the translation schematically; the full propagator argument is a
  separate workstream (cf. `predictions/c1_photon_bundle.py`).


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
- `predictions/srs_bloch_lv_dim6.py` + `_derivation.md` — scalar-Bloch sister with
  the Feshbach-Löwdin proof of $D_H = 1/16$, $D_4^{{\rm iso},H} = -1/1024$, $D_4^{{\rm aniso},H} = +1/1536$.
- `proofs/foundations/lorentz_sig_h_lv_4th_order_symbolic.py` — primary symbolic
  source for the scalar coefficients (Feshbach-Löwdin partition + sympy exact arithmetic).
- `proofs/foundations/lorentz_sig_ihara_lv_relation.py` — symbolic Ihara cross-walker theorem.
- `proofs/foundations/lorentz_sig_hashimoto_d4_iso.py` — independent numerical
  verification of $D_4^{{\rm iso},{\rm NB}} = +3/512$.
- `docs/theorems/theorem_lorentz_causal_sector.md` §6 (Stage 3 theorem containing this result).
- `proofs/lorentz/hashimoto_dispersion_symbolic.py` (high-precision numerical cross-check).
- `proofs/lorentz/hashimoto_bloch_dispersion.py` (initial numerical dispersion fit).
- `proofs/common.py` (srs atom positions and NN_DIST).

### Published experimental
- **Cao et al. (LHAASO Collab.)** (2024). JCAP 04, 060 [arXiv:2312.09079]. Current tightest dim-6 propagation bound.
- **Li & Ma** (2023). EPJC 83, 192 [arXiv:2210.06338]. Threshold-shift analysis.
- **Martinez-Huerta et al.** (2020). Symmetry 12, 1232. Threshold-shift review.
- **Finke & Razzaque** (2023). ApJL 942, L21 [arXiv:2210.11261]. Tentative LIV evidence from GRB 221009A.
- **Addazi et al.** (2022). Prog. Part. Nucl. Phys. 125, 103948 [arXiv:2111.05659]. LIV review.
