# Derivation of E_P (adjacency eigenvalue at the P-point)

**Audit anchor:** Foundational. Conditional on Rows 4, 6 of `docs/audits/registers/uniqueness_ledger.md` (k* = 3 + srs identification). Spectral-decomposition theorem-grade per `docs/theorems/theorem_bloch_lift_mu.md`.

## Abstract

We derive $E_P = \sqrt{k^*} = \sqrt{3}$, the positive eigenvalue of the adjacency (Bloch) Hamiltonian of the srs lattice at the P-point of the Brillouin zone. The 4×4 Bloch matrix $H(k_P)$ has characteristic polynomial $(\lambda^2 - k^*)^2$, giving doubly-degenerate eigenvalues $\pm\sqrt{k^*}$. The factorization is forced by the $C_3$ site symmetry at P. The derivation is matrix diagonalization with one upstream input ($k^* = 3$).

## Framework axioms invoked

None beyond upstream:
- $k^* = 3$ (from `predictions/k_star.py`)
- srs lattice selected by MDL (from `predictions/g_girth_derivation.md`, Step 2)

## Derivation

### Step 1: Bloch Hamiltonian of srs

The srs lattice has space group $I4_132$ (#214) with atoms at Wyckoff position 8a ($x = 1/8$). The primitive cell (BCC) contains 4 atoms. The Bloch Hamiltonian $H(\mathbf{k})$ is a $4 \times 4$ matrix whose entries are sums of phase factors $e^{i\mathbf{k} \cdot \mathbf{d}_{ij}}$ over nearest-neighbor bonds $\mathbf{d}_{ij}$.

This construction is standard for tight-binding models on crystal nets (Ashcroft & Mermin, *Solid State Physics*, Ch. 10; applied to srs by Sunada, 2012).

### Step 2: Evaluation at the P-point

The P-point of the BCC Brillouin zone is $\mathbf{k}_P = \frac{\pi}{2a}(1, 1, 1)$, where $a$ is the conventional cell parameter. At this high-symmetry point, the little group (stabilizer) contains the $C_3$ rotation about the $(111)$ axis.

The $C_3$ symmetry forces the $4 \times 4$ matrix $H(\mathbf{k}_P)$ to decompose into blocks. Specifically, the 4 bands split under $C_3$ into irreducible representations: two trivial ($C_3$ eigenvalue 1) and two non-trivial ($\omega, \omega^2$ where $\omega = e^{2\pi i/3}$). The trivial pair and the conjugate pair each form a $2 \times 2$ block.

### Step 3: Characteristic polynomial

Each $2 \times 2$ block has the form $\begin{pmatrix} 0 & c \\ c^* & 0 \end{pmatrix}$ with $|c|^2 = k^*$ (the coordination number enters as the sum of squared bond weights at the P-point). The eigenvalues of each block are $\pm|c| = \pm\sqrt{k^*}$.

The full characteristic polynomial is:

$$\det(H(\mathbf{k}_P) - \lambda I) = (\lambda^2 - k^*)^2 = 0 \tag{1}$$

This is verified by explicit numerical diagonalization in `proofs/foundations/srs_E_at_P_derivation.py`, which constructs $H(\mathbf{k}_P)$ from the srs Wyckoff coordinates and confirms eigenvalues $\{+\sqrt{3}, +\sqrt{3}, -\sqrt{3}, -\sqrt{3}\}$.

### Step 4: Result

The positive eigenvalue:

$$E_P = +\sqrt{k^*} = \sqrt{3} \approx 1.7321 \tag{2}$$

with multiplicity 2 (protected by $C_3$ symmetry).

## Result

$$\boxed{E_P = \sqrt{k^*} = \sqrt{3}}$$

## Comparison with experiment

$E_P$ is not directly measured. It determines the Hashimoto eigenvalue $h$ (via $h^2 - E_P h + (k^* - 1) = 0$), which in turn determines PMNS phases, the chirality factor $(5/3)$, and the walk attenuation $|h|^2 = k^* - 1 = 2$.

## Open questions

None. The derivation is matrix diagonalization of an explicitly constructed Bloch Hamiltonian, verified numerically. The $C_3$ factorization is standard representation theory.

## Audit v2 (Clause 7) status

This prediction inherits Row 4 (k* = 3) audit v2 closure. The §3 multi-mechanism
table (M1-M6 vs qtz at k=4) is consolidated in
an internal working note §2.1; closure doc:
an internal working note.

- **Status (post-audit-v2):** UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 audit v2 closure (DOMINANT-with-named-margins; UNIQUE-on-η_B).
- **Named margin** (from index §2.1): combined M2 (~0.45 Boltzmann) × observable-specific M3·M4·M5 product.
- **Inherits structurally:** Row 4 v2 closure protects against qtz at k=4 alternative via combined mechanism product. M6 sign-flip is irrelevant for this observable (uses Im(h)/|h|² or magnitude, not Re(h) sign) unless the observable uses Re(h_P) directly (η_B specifically; see `predictions/eta_B_derivation.md` for that case).
- **Conditionals deferred:** RCSR-vetted qtz bond list verification; selection-rule audit; data-conditional MDL.
