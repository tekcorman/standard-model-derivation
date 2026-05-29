# Chapter 5 — Complex Hilbert space is forced

[Chapter 4](04-recurrence-and-the-mdl-waterline.md) delivered the substrate to $L^2(F_{\mathrm{inv}}(E))$ over a yet-unselected field. This chapter walks the eight-step chain from there to $\mathbb{C}$ — the framework's central structural result. The setup is (A) + (B) + (I) + standard mathematics; the conclusion is that the substrate's natural Hilbert space is **complex**, not real, not quaternionic.

## The chain, step by step

1. **Substrate.** (A) + (B) + (I) give $F_{\mathrm{inv}}(E)$ and its Cayley graph as derived theorems ([Chapter 2](02-three-commitments.md), [Chapter 3](03-the-cover-that-holds-chirality.md)).
2. **Function space.** The observer reads functions on the substrate. The natural space is $L^2(F_{\mathrm{inv}}(E))$ over a yet-unselected field $\mathbb{F}$ (Folland 1999 §11.4).
3. **Operators.** Adjacency and Hashimoto operators on $L^2(F_{\mathrm{inv}}(E))$ carry the substrate's combinatorial content as bounded self-adjoint operators (Reed–Simon I §VI; Hashimoto 1989).
4. **Translation invariance.** $F_{\mathrm{inv}}(E)$ acts on $L^2(F_{\mathrm{inv}}(E))$ by left and right regular representations. Operators built from the Cayley graph commute with this action. **Bloch decomposition** over the substrate's lattice quotient becomes available (Sunada 2013 §6).
5. **Continuum-time limit.** Rapid decay of toggle correlations on the substrate ($\xi_t \approx 0.558\,\ell_P$, sub-Planckian, CAS-verified) licenses the **discrete-to-continuous quantum-walk limit** (Strauch 2006; Childs 2009). The discrete dynamics on the Cayley graph converges in strong operator topology to a strongly-continuous one-parameter unitary group $U(t)$ on $L^2(F_{\mathrm{inv}}(E))$.
6. **Stone's theorem.** $U(t)$ admits a unique infinitesimal generator (Stone 1932; Reed–Simon I §VIII.4):
   - **On $\mathbb{C}$-$L^2$:** generator is a self-adjoint operator $H$ with $U(t) = \exp(-iHt)$ and $\sigma(H) \subset \mathbb{R}$.
   - **On $\mathbb{R}$-$L^2$:** generator is a skew-symmetric operator $B$ with $U(t) = \exp(Bt)$ and $\sigma(B) \subset i\mathbb{R}$.
7. **Register-is-real.** By (B), the observer is a finite register whose content is real-valued — each bit takes values in $\{0, 1\} \subset \mathbb{R}$. **Any spectral data the observer extracts from the substrate must fit in the register, hence must be real.**
8. **Field selection.** On $\mathbb{R}$-$L^2$ the generator's spectrum is purely imaginary, incompatible with register-storable real eigenvalues. On $\mathbb{C}$-$L^2$ it is real, compatible. The substrate's natural Hilbert space is **complex $L^2(F_{\mathrm{inv}}(E); \mathbb{C})$**.

Each step is either one of the three commitments (A)/(B)/(I), a derived foundational theorem, or a citation to standard published mathematics. There is no additional axiom.

## Quaternionic is also excluded

The $\mathbb{H}$-$L^2$ alternative is excluded separately: Adler's quaternionic version of Stone's theorem gives an anti-self-adjoint generator with spectrum in the pure-imaginary quaternions (a 3-real-dimensional set), also incompatible with register-storable real eigenvalues. So among $\mathbb{R}$, $\mathbb{C}$, $\mathbb{H}$, only $\mathbb{C}$ satisfies the register-is-real constraint.

## What this chain accomplishes

Before Step 8, the framework's substrate is **field-agnostic** — every operation through Layer 4 of the operator sweep works over both $\mathbb{R}$-$L^2$ and $\mathbb{C}$-$L^2$. At Step 8, $\mathbb{C}$ is selected. Layer 5 becomes available: **Pauli operators, Clifford algebras, Jordan–Wigner, density matrices, Schrödinger evolution, complex Lie groups**. Layer 6 (smooth manifolds, Riemannian geometry, GR) becomes partially available pending the smooth-manifold portion of the continuum-limit closure.

The selection rests on (A) + (B) + (I) alone. **A5-mass is not invoked at any step.** Complex Hilbert space is not an empirical input; it is forced by what a finite register can extract from a toggle substrate undergoing recurrent dynamics.

## Why this is sharper than the standard derivations

The standard derivations of complex quantum mechanics — Hardy 2001, Chiribella–D'Ariano–Perinotti 2011, Masanes–Mueller 2011 — take operational axioms (local tomography, purification, ideal compressions) and derive the complex Hilbert structure. They presuppose an **operational scope**: a system that admits states, operations, and measurements.

The chain above is *upstream* of those derivations. The operational scope is itself a consequence: the finite register reads functions on $F_{\mathrm{inv}}(E)$; the substrate's dynamics has a continuum limit because rapid decay licenses Strauch–Childs; the limit is unitary; the generator's spectrum has to be register-storable. The CDP-style derivations sit naturally at Layer 5+ as theorems about the structure already forced at Step 8.

## Next

[Chapter 6 — The two pillars](06-the-two-pillars.md): complex Hilbert space in hand, two algebraic structures on srs carry all the Standard Model content. **Flavor** (the Hashimoto eigenvalue $h$ at the high-symmetry Brillouin-zone point) carries mixing angles, CP phases, mass hierarchies. **Gauge** (the Clifford algebra $\mathrm{Cl}(6) = \mathrm{Cl}(4) \otimes \mathrm{Cl}(2)$) carries gauge group, couplings, fermion content.
