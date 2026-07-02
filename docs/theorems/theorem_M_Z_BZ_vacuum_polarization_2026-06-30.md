# M_Z: the BZ-integrated Z-current vacuum polarization — the shell does NOT bracket; M_Z is a forced oblique residual (2026-06-30)

**Status: COMPLETE HONEST RESULT (not a closure, not a failure).** The planned M_Z
attack — build the **Brillouin-zone-integrated Z-current (T₃ − s²Q) vacuum
polarization** and ask whether the SM's `0.810` fraction of the chiral-shell step
*falls out* — was carried out. **It does not.** The genuine BZ integral gives the shell
at `R = 0.2046` of its Γ-point template value, not `0.810`. M_Z is therefore confirmed
to be a **forced substrate-vs-SM oblique difference (a real ~4%-relative residual)**,
the framework's intrinsic precision floor on the EW oblique — exactly the honest prior.

Reproducer / deliverable: `proofs/foundations/M_Z_BZ_integrated_vacuum_polarization_2026-06-30.py`.
Supersedes the bracket framing of `theorem_M_Z_shell_vertexbox_closure_2026-06-29.md`
(the bracket's upper arm is now shown to be a Γ-evaluation artifact).

## 1. What was tested
The M_Z tree→pole oblique = **Perron singlet (δ_r) + shell vertex/box (the Q-winding)**.
The shell template (theorem 2026-06-29) reads
`shell = (Σw²)·(1/2|E|)·s⁴·F·α₁` with **F = Im(h)/|h|² evaluated at the single k-point Γ**
(F_Γ = √7/4). That gives the bracket:

| oblique | value | M_Z |
|---|---|---|
| δ_r (Perron singlet, LIVE) | 0.3384% | **+8.1σ** under |
| + chiral shell @Γ (Σw²=2, F=√7/4) | 0.3614% | **−1.9σ** over |
| SM tree→pole target | 0.3570% | = δ_r + **0.8123**·(chiral step) |

The closure question: does that **0.810** fall out of the genuine **BZ integral** of the
Z-current vacuum polarization, rather than the single-k-point template?

## 2. The forced object (no fit, basis-free)
On `directed_edges()`'s own ordering (so `[B(0), P] = 0`, verified), build the C₃ dart
permutation `P` (σ=(1 3 2) on vertices, (n₁n₂n₃)→(n₃n₁n₂) on homology) and the
winding-charge operator `W = (P − P²)/(i√3)` (eigenvalues {0, +1, −1}; Tr W² = 8). Then

```
shell_BZ = <Σ w² · F>_BZ · (1/2|E|) · s⁴ · α₁
<Σ w² · F>_BZ = ∫_BZ d³k  Σ_{n: Im λ_n(k) > 0}  |⟨l_n|W|r_n⟩|²  ·  Im(λ_n)/|λ_n|²
R = <Σ w² · F>_BZ  /  [Σ w² · F]_Γ ,      [Σ w² · F]_Γ = 2·(√7/4)
```

- **Chirality = the Im(λ) > 0 hemisphere.** At Γ the two hemispheres *cancel exactly*
  (h gives +2·√7/4, h̄ gives −2·√7/4); the chiral shell IS one hemisphere. This is the
  clean, band-tracking-free definition of "chiral."
- **Basis-free cross-check.** Because `B(−k) = conj(B(k))`, the hemisphere integral
  equals `(1/2) ∫_BZ Σ_{all n} w²·|Im λ|/|λ|²`. The two agree to all printed digits at
  every grid — so `R` is not an artifact of band classification or the chiral cutoff.
- **F is the genuine k-dependent spectral functional.** Away from the symmetric points
  `|λ(k)|² ≠ 2` (Ramanujan saturation is special to Γ, P), so F(λ(k)) = Im(λ)/|λ|² is no
  longer pinned to √7/4. The whole point of the BZ integral is to let it vary.
- **s² drops out of R** (it is the external EW input `sin²θ_W`, entering template and BZ
  integral identically as the s⁴ prefactor). R is a pure substrate number.

The Γ value of the integrand exactly reproduces the template (R → 1 as integrand → Γ).

## 3. Result
```
R = 0.2046     (converged: 0.20454 → 0.20458 over ngrid 12…44, basis-free cross-check identical)
```
**The entire contribution is the shell band**: the |λ|=1 band contributes 0.000 (its
modes are real at Γ — F=0 — and carry no chiral Q-current weight across the BZ). So
"shell vertex/box" is the correct object and the normalization is apples-to-apples.

`0.810` does **not** fall out (off by −75%). Consequences:

```
BZ shell           = R · 0.0230%               = +0.0047%
substrate oblique  = δ_r + shell_BZ = 0.3431%   (SM tree→pole = 0.3570%)
=> substrate UNDER-predicts the SM oblique by 0.0139%  (3.9% relative)
M_Z residual:  δ_r alone        +8.1σ
               + BZ shell        +6.1σ   (NOT closed)
```

The Γ "bracket" was an **artifact of evaluating F at its Brillouin-zone maximum (Γ)** —
the high-symmetry point where |λ|²=2 is minimal and Im λ is maximal, so F there is the
BZ peak. The genuine integral is ~5× smaller and does **not** bracket the SM: the
substrate **under-predicts** the EW oblique. There is no "0.8 of the chiral step" — that
was a coincidence of comparing the BZ maximum to the BZ average.

## 4. Robustness / why 0.205 is the forced number (and 0.810 is firmly excluded)
- diagonal per-mode F (the literal BZ generalization of the template): **0.2046**
- ½·Σ_all w²|F| (basis-free, no band cut, no chirality ambiguity): **0.2046** (identical)
- shell-band only / |λ|=1-band only: **0.2046 / 0.000** (all weight in the shell)
- full two-propagator bubble `Σ_{m,n}|⟨l_m|W|r_n⟩|²·½(F_m+F_n)` (interband, the genuine
  field-theory current–current correlator): **R ≈ 0.56–0.59**, but it does **not converge**
  (exceptional-point sensitivity of the non-normal B at near-degeneracies), is **not** the
  framework's template object, and is **still not 0.810**.

So under the framework's own (well-conditioned, basis-free) object R = 0.205; under the
more complete but ill-conditioned bubble R ≈ 0.57. **No natural definition reaches 0.810.**
The diagonal per-mode F is the forced choice (basis-free identity holds; full bubble is
ill-defined for non-normal B).

## 5. Verdict
The M_Z residual is a **forced substrate-vs-SM oblique difference** — the substrate's
Q-current vacuum polarization, integrated honestly over the Brillouin zone, predicts the
EW oblique to **~4% relative** (δ_r + shell_BZ = 0.343% vs SM 0.357%), leaving M_Z at
**+6σ**. This is the framework's intrinsic precision floor on the oblique, **not** a
missing term that zeroes out and **not** a fit. The `0.810` is **not** forced by the
T₃ − s²Q structure; it was the ratio of the BZ maximum to the BZ average, a coincidence.

The live single-term prediction (Perron-only δ_r, +8.1σ) stands as the forced result; the
forced next term (BZ shell, +0.0047%) improves it to +6.1σ but does not close it.

**Honest prior confirmed** (`docs/incomplete_equations_todo.md` ★ NEXT ATTACK): "likely a
confirmed residual; closure only if T₃−s²Q forces 0.810." It does not. M_Z is the last of
the framework's σ-levers and it bottoms out as a genuine ~few-% oblique residual.

**Lesson banked:** a k-point-representative template evaluated at a *high-symmetry point*
systematically over-estimates a BZ integral (Γ is the spectral extremum). An apparent
"bracket" between a Perron read and a Γ-shell read can be an evaluation artifact, not a
real bound — integrate over the zone before claiming a bracket.
