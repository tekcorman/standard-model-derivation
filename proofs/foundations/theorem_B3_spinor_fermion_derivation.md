# Theorem B3 — Cl(6, 0) spinor decomposes as one SM generation (electroweak)

**Audit anchor:** Rows 16, 17 of `docs/audits/registers/uniqueness_ledger.md` (Cl(6,ℂ) at each node UNIQUE; Pati-Salam Spin(4)×Spin(2) ⊂ Spin(6) UNIQUE within srs cubic symmetry). Pati-Salam labeling within is ADOPTED-B3 per `docs/audits/registers/adoption_register.md`.

## Abstract

Given Theorem B1.b (invariant Clifford formulation) and Theorem B2 (the
canonical non-degenerate form Q on the 6-edge space has signature (6, 0)),
the framework's Clifford algebra is Cl(6, 0). Its unique complex irreducible
spinor representation has dimension $2^{6/2} = 8$. We show that under the
natural Spin(4) × Spin(2) = SU(2)_L × SU(2)_R × U(1)_{B−L} subgroup of
Spin(6), this 8-dim Dirac spinor decomposes as the electroweak content of
exactly one Standard-Model generation with colour factored out (the
colour-trivialised Pati–Salam multiplet $\{\nu, e, u, d\} \times \{L, R\}$).
The identification is **unique up to a $(Z/2)^3$ group of named convention
choices**. A right-handed $\nu_R$ is forced (no Majorana-Weyl reduction
exists for signature (6, 0)).

## Framework axioms invoked

- **A1** (self-inverse toggle): enters indirectly via upstream theorems B1.b and BP.
- **A2** (MDL): enters via Theorem B1.b selecting the invariant formulation.
- **Upstream frozen results:** B1.b (tensor-algebra Clifford), B2 (sig (6,0)), BP (P-point C_3).

The theorem itself is a standard representation-theoretic decomposition of
Cl(6, 0); the axiom content is entirely upstream.

## Derivation

### Step 1 — Cl(6, 0) on $\mathbb{C}^8$ via the Brauer-Weyl construction

Complexified Clifford algebras satisfy Cl$(2k) \cong M_{2^k}(\mathbb{C})$
(Lawson & Michelsohn 1989 I Theorem 5.7). For $k = 3$: Cl$(6) \cong M_8(\mathbb{C})$.
We realise the generators via the Brauer–Weyl iterative Pauli construction
(Brauer & Weyl 1935):

$$
\Gamma_1 = \sigma_x \otimes I \otimes I, \quad
\Gamma_2 = \sigma_y \otimes I \otimes I, \quad
\Gamma_3 = \sigma_z \otimes \sigma_x \otimes I,
$$
$$
\Gamma_4 = \sigma_z \otimes \sigma_y \otimes I, \quad
\Gamma_5 = \sigma_z \otimes \sigma_z \otimes \sigma_x, \quad
\Gamma_6 = \sigma_z \otimes \sigma_z \otimes \sigma_y.
$$

Verified: $\{\Gamma_a, \Gamma_b\} = 2\delta_{ab} I_8$ (signature $(+,+,+,+,+,+)$);
each $\Gamma_a$ is Hermitian; the 64 products $\Gamma^\alpha$ for
$\alpha \in \{0,1\}^6$ are linearly independent in $\mathrm{End}(\mathbb{C}^8)$,
confirming the isomorphism.

### Step 2 — Spin(6) Cartan and weight lattice

The bivectors $\Gamma_{ab} = \tfrac{1}{2}[\Gamma_a, \Gamma_b]$ generate Spin(6).
Cartan generators:

$$
T_1 = \frac{\Gamma_{12}}{2i}, \quad T_2 = \frac{\Gamma_{34}}{2i}, \quad
Y = \frac{\Gamma_{56}}{2i}.
$$

Each is Hermitian and mutually commuting. Their simultaneous eigenvalues give
$2^3 = 8$ one-dimensional weight spaces with labels $(\varepsilon_1, \varepsilon_2,
\varepsilon_Y) \in \{+1, -1\}^3$ (doubled from the $\pm\tfrac{1}{2}$
eigenvalues). This exhausts $\mathbb{C}^8$ and matches the Spin(6) Dirac spinor
weight lattice (Fulton & Harris 1991 §20.1).

The Spin(4) = SU(2)_L × SU(2)_R subgroup uses:
$T_L = T_1 + T_2$, $T_R = T_1 - T_2$.

A weight is in the SU(2)_L sector iff $\varepsilon_1 = \varepsilon_2$, and in
the SU(2)_R sector iff $\varepsilon_1 = -\varepsilon_2$.

### Step 3 — Chirality and Weyl decomposition

$$
\Gamma_7 := -i\,\Gamma_1\Gamma_2\Gamma_3\Gamma_4\Gamma_5\Gamma_6.
$$

Verified: $\Gamma_7 = \Gamma_7^\dagger$, $\Gamma_7^2 = I_8$,
$\{\Gamma_7, \Gamma_a\} = 0$ for $a = 1, \ldots, 6$,
$[\Gamma_7, \Gamma_{ab}] = 0$ for all bivectors.

Hence $\Gamma_7$ splits $\mathbb{C}^8 = S^+ \oplus S^-$ into $4 + 4$ Spin(6)-invariant
Weyl components. On weight vectors:

$$
\Gamma_7 \ket{\varepsilon_1, \varepsilon_2, \varepsilon_Y}
= s_{\rm conv} \cdot \varepsilon_1 \varepsilon_2 \varepsilon_Y \,
  \ket{\varepsilon_1, \varepsilon_2, \varepsilon_Y},
$$

where $s_{\rm conv} \in \{+1, -1\}$ is the overall sign convention. On each
chirality sector (4 states), exactly 2 lie in the SU(2)_L doublet and 2 in
the SU(2)_R doublet, each doublet with a definite value of $\varepsilon_Y$.

### Step 4 — Pati-Salam identification

Under the Pati–Salam SU(4) = Spin(6) embedding restricted to
$\mathrm{SU}(4) \to \mathrm{SU}(3)_c \times U(1)_{B-L}$ and factoring out
colour (Pati & Salam 1974 Eqs. (3)–(5); Baez & Huerta 2010 §4), the SU(4)
fundamental $\mathbf{4}$ collapses to a two-state multiplet discriminated by
the sign of $U(1)_{B-L}$ (lepton vs. quark).

The resulting 8-state particle dictionary (verified explicitly):

| chirality | $T_1$ | $T_2$ | $Y$ | species |
|:---------:|:-----:|:-----:|:---:|:-------:|
| $+1$ | $+1$ | $+1$ | $+1$ | $\nu_L$ |
| $+1$ | $-1$ | $-1$ | $+1$ | $e_L$ |
| $+1$ | $+1$ | $-1$ | $-1$ | $u_L$ |
| $+1$ | $-1$ | $+1$ | $-1$ | $d_L$ |
| $-1$ | $+1$ | $-1$ | $-1$ | $\nu_R$ |
| $-1$ | $-1$ | $+1$ | $-1$ | $e_R$ |
| $-1$ | $+1$ | $+1$ | $+1$ | $u_R$ |
| $-1$ | $-1$ | $-1$ | $+1$ | $d_R$ |

(Sign conventions (a)–(c) may permute rows; the structure as 4 SU(2) doublets
is fixed.)

### Step 5 — Uniqueness up to $(Z/2)^3$

Three independent convention freedoms:
- **(a)** Sign of $\Gamma_7$ (swaps $L \leftrightarrow R$).
- **(b)** Sign of $Y$ (swaps lepton $\leftrightarrow$ quark).
- **(c)** Self-dual vs anti-self-dual SU(2) combination (swaps $T_L \leftrightarrow T_R$).

These generate $(Z/2)^3$ (order 8). No structural criterion inside Cl(6, 0) or
the Spin(4) × Spin(2) embedding can fix any of them.

### Step 6 — $\nu_R$ forced

By Lawson & Michelsohn 1989 I.4 Table 5.1, Majorana-Weyl spinors exist only
for $(p - q) \equiv 0 \pmod{8}$ with $p + q \equiv 0 \pmod{8}$. For
$(p, q) = (6, 0)$: $p - q = 6 \not\equiv 0 \pmod{8}$. No Majorana-Weyl
reduction. The full 8-dim Dirac spinor cannot be truncated to a chiral 4;
$\nu_R$ is present.

## Result

The 8-dim Cl(6, 0) Dirac spinor decomposes as **one complete Pati-Salam
generation with colour factored out**, containing the eight SM species
$\{\nu, e, u, d\} \times \{L, R\}$, each appearing once inside a natural
SU(2) doublet. The identification is unique up to $(Z/2)^3$. A Dirac $\nu_R$
is forced by the signature.

## Comparison with experiment

N/A — foundational/structural theorem. The result matches the observed
fermion content of one SM generation (with colour deferred to B4), but this
match is a necessary consistency check rather than a numerical prediction.
The three-generation structure is not derived here (open question B3.1).

## Open questions

- **(B3.1)** Generation multiplicity: the 8-dim spinor accounts for exactly
  one generation; the physical factor of 3 requires a multi-k-point or full-BZ
  construction (addressed in Theorem B5.3-core).
- **(B3.3)** $\nu_R$ status: Cl(6, 0) forces $\nu_R$ as a Dirac partner but
  is silent on its Majorana mass term.
- **(B4)** Colour SU(3): not in Cl(6, 0); deferred to Workstream B4.
- **(B3.4)** Lorentzian extension: the Euclidean (6, 0) algebra requires
  an external time direction for relativistic physics.

## References

- Baez, J.C. & Huerta, J. (2010). The Algebra of Grand Unified Theories.
  *Bull. Amer. Math. Soc.* 47, 483–552. §4.
- Brauer, R. & Weyl, H. (1935). Spinors in $n$ dimensions.
  *Amer. J. Math.* 57, 425–449.
- Fulton, W. & Harris, J. (1991). *Representation Theory.* Springer GTM 129. §20.1.
- Lawson, H.B. & Michelsohn, M.-L. (1989). *Spin Geometry.* Princeton.
  I Theorem 4.3, 5.7, Table 5.1.
- Pati, J.C. & Salam, A. (1974). Lepton number as the fourth colour.
  *Phys. Rev. D* 10, 275–289.
