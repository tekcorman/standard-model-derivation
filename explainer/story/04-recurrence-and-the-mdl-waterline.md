# Chapter 4 — Recurrence and the MDL waterline

## The observer's record is bounded

The observer registers recurrence ([Chapter 1](01-what-can-exist.md)): out of the combinatorial bath, only patterns that come back are *something* rather than noise. (B) makes this concrete — the observer is a finite register, and the register accumulates recurrent patterns as it encounters them.

The register's capacity is bounded. Recurrent patterns whose description is shorter than the raw stream they came from can be held; those whose description matches or exceeds the raw stream cannot. By **Shannon source coding** (Shannon 1948 §I, Theorem 9), the optimal compression rate of any source is its entropy. By **Rissanen / Grünwald** (Rissanen 1978; Grünwald 2007 §§5.1–5.3), the description-length comparison is **Minimum Description Length (MDL)**: a model $M$ is retained if and only if

$$L_{\mathrm{total}}(M) = L_{\mathrm{model}} + L_{\mathrm{data}\mid M} < L_{\mathrm{raw}}.$$

**MDL is therefore a theorem of (A) + (B)** ([`theorem_A2_mdl_from_finite_register.md`](https://github.com/tekcorman/standard-model-derivation/blob/main/docs/theorems/theorem_A2_mdl_from_finite_register.md)). It is what *finite register reading toggle activity* means under standard information theory. It is not an additional commitment.

## The waterline reading

MDL is most often stated as a selection rule: among compressed models, pick the shortest. The framework reads it as a **waterline**: every model with positive savings ($L_{\mathrm{total}} < L_{\mathrm{raw}}$) is retained. Multiple compressions of the same source coexist whenever they each clear the threshold.

```
                  L_raw  ─────────────────  the waterline
                    │                                  ▲
   compression-savings  RETAINED  ←─── every model clearing the line
                    │                       (plural, simultaneously)
                    │
                    ▼
                  L_total  DISCARDED ←──── below-threshold models
```

<video controls autoplay muted loop playsinline width="100%" style="max-width: 800px; display: block; margin: 0 auto;">
  <source src="../../assets/animations/mdl_waterline.mp4" type="video/mp4">
</video>

*Many candidate compressions of the same source. The waterline sits at the uncompressed data cost $L_{\mathrm{raw}}$. Every compression that pays for itself (positive savings $L_{\mathrm{raw}} - L_{\mathrm{total}} > 0$) is retained simultaneously; the framework's "uniqueness" claims are dominance claims within this retained set. The dominant compression (srs at coordination number three, spatial dimension three) is well above the line; the mirror-chirality copy srs\* is also retained; many subdominant compressions clear the line; below-waterline noise is discarded.*

## What the waterline forces

The waterline is what the substrate's plurality looks like from the observer's side:

- **Chirality is both-handed.** The chirality of the substrate's lattice quotient has both hands above the waterline simultaneously — mirror-image patterns, equally compressible. The framework predicts the chirality plurality structurally. (Compare [Chapter 3](03-the-cover-that-holds-chirality.md) — srs-z is the cover that *holds* both chiralities so the mass operator can couple them.)
- **All three generations.** The $C_3$ generation symmetry has all three labels above the waterline simultaneously — three-fold cyclic recurrences, each compressible at the same rate.
- **All girth-cycle windings.** Closed-walk windings on the substrate's cycles all clear the waterline together; the framework's $V_{cb}$ prediction is the geometric series over winding numbers, not a single dominant term:

$$|V_{cb}| = \sum_{n \geq 1} (2/3)^{8n} = \frac{(2/3)^8}{1 - (2/3)^8} = \frac{256}{6305} \approx 0.04060$$

matching PDG-2024 exclusive (Belle) at $+0.00\sigma$.

None of these are anomalies to clean up. They are what observer-recurrence-filtering *says* is there.

## "Uniqueness" is dominance

The framework's "uniqueness" claims — $k_* = 3$ trivalent nodes, $d = 3$ spatial dimensions, the srs lattice — are **dominance** claims. The dominant compression is unique and well above the waterline; subdominant compressions exist but contribute negligibly. Both readings — "uniquely dominant" and "subdominant retentions exist" — are correct simultaneously.

## What recurrence-filtering does NOT import

- **Thermodynamics is not imported** to motivate compression. Source coding is information-theoretic; the entropy here is Shannon entropy on register descriptions, not Boltzmann entropy on physical states. The framework's later thermodynamic content — Landauer's principle, the observer's energy functional, the arrow of time — is *derived* downstream once the structural chain is established. Importing thermodynamics here would be circular.
- **(B) is not an engineering choice.** It is the operational definition of *observer*. The framework's commitment is not "we choose to study finite-register observers"; it is "*observer* means finite register, and dynamical existence registered by such an observer requires recurrence-filtering anyway."

## Next

The next chapter (in progress) walks the **structural chain to complex Hilbert space**: under (A) + (B) + (I) plus standard mathematics, the observer's natural function space on $F_{\mathrm{inv}}(E)$ is forced to be complex (not real, not quaternionic). The argument is in [`docs/framework/narrative_spine.md`](https://github.com/tekcorman/standard-model-derivation/blob/main/docs/framework/narrative_spine.md) §5.

---

*This chapter is a condensation of [`docs/framework/narrative_spine.md`](https://github.com/tekcorman/standard-model-derivation/blob/main/docs/framework/narrative_spine.md) §4 (the rigorous version).*
