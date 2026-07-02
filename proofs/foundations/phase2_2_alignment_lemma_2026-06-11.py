#!/usr/bin/env python3
"""Phase 2.2 stage 2 — THE ALIGNMENT LEMMA (the panel's named K2 target).

Lemma (Hermitian-positive C3-circulant sqrt(M) under R3 Fourier duality):
Let M >= 0 be the mass operator, diagonal in the generation basis (R3), and
sqrt(M) its CANONICAL POSITIVE root. Write the character (Fourier) data
z_alpha = (1/3) sum_j sqrt(m_j) w^{-j alpha}.

 (i)  FORWARD (alignment = Hermiticity): because sqrt(m_j) are real >= 0,
      automatically z_0 = mean(sqrt m) real >= 0 and z_2 = conj(z_1).
      The conjugate-aligned phase structure (0, +delta, -delta) is FORCED;
      delta = arg z_1 is the ONLY phase freedom. The former K2 residue
      ("a chosen per-channel phase") dissolves: it was Hermiticity.
 (ii) CONVERSE (completion): given channel magnitudes |z| = (sqrt w0,
      sqrt w1, sqrt w1) from the Born weights, the Hermitian-positive
      completion exists iff delta lies in the positivity window
      (all f_j = z0 + 2|z1| cos(delta + 2 pi j/3) >= 0). For the P-saddle
      weights (1/2, 1/4, 1/4): eps = 2|z1|/z0 = sqrt2, window |delta| <=
      pi/12, and Q = (1 + eps^2/2)/3 = 2/3 IDENTICALLY across the window
      (symbolic identity sum f^2 = 3 z0^2 (1 + eps^2/2), sum f = 3 z0).
(iii) MIXED-STATE VALIDITY: the construction consumes ONLY channel
      magnitudes — the panel's rho = I/8 control is the lemma's PREDICTION
      (no state coherence is used anywhere), resolving the "coherence lives
      in the read" objection: there is no coherence claim; the phases come
      from Hermiticity of the observable, not from the state.
 (iv) SADDLE DICHOTOMY corollary: at the Gamma/H weights (1/3,1/3,1/3),
      eps = 2 and the positivity window DEGENERATES to the measure-zero
      points delta = 0 (mod 2pi/3) with spectrum sqrt(m) prop. (3, 0, 0) —
      no robust Koide structure at the neutrino saddles (logged; physical
      interpretation NOT promoted).

Status note: this discharges the K2-class residue named by the 2026-06-11
P2 panel. The two ~1-bit identifications (mass = Born weight at P;
uniform-over-CSCO) remain priced; verdict upgrade requires panel
ratification per the frozen spec (clause: verdicts at ultracode).
"""
import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

RNG = np.random.default_rng(20260611)
TOL = 1e-12
FAILURES = []
W = np.exp(2j * np.pi / 3)


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main():
    print("=" * 72)
    print(" PHASE 2.2 stage 2 -- THE ALIGNMENT LEMMA (Hermiticity, not choice)")
    print("=" * 72)

    # (i) forward: alignment is automatic for any real nonneg sqrt-masses
    ok = True
    for _ in range(20):
        f = RNG.random(3) * 10  # arbitrary nonneg sqrt-masses
        z = [np.mean(f * np.array([W ** (-j * al) for j in range(3)])) for al in range(3)]
        ok &= abs(z[0].imag) < TOL and z[0].real >= -TOL and abs(z[2] - np.conj(z[1])) < TOL
    gate("AL1 (i) forward: z0 real>=0, z2 = conj(z1) AUTOMATIC for any masses",
         ok, "z2 = conj(z1) is reality/Hermiticity; z0 >= 0 is POSITIVITY of the "
             "canonical root; only freedom is arg z1 (ratification-corrected)")

    # (ii) symbolic: sum f^2 = 3 z0^2 (1 + eps^2/2); Q = (1+eps^2/2)/3, delta-free
    z0s, eps, dl = sp.symbols('z0 eps delta', positive=True, real=True)
    j = sp.symbols('j', integer=True)
    fj = [z0s * (1 + eps * sp.cos(dl + 2 * sp.pi * jj / 3)) for jj in range(3)]
    sum_f2 = sp.simplify(sum(x**2 for x in fj))
    sum_f = sp.simplify(sum(fj))
    Qsym = sp.simplify(sum_f2 / sum_f**2)
    gate("AL2a symbolic: Q = (1 + eps^2/2)/3 identically (delta cancels)",
         sp.simplify(Qsym - (1 + eps**2 / 2) / 3) == 0, f"Q = {Qsym}")
    gate("AL2b eps = sqrt2 => Q = 2/3 exactly, window-uniform",
         sp.simplify(Qsym.subs(eps, sp.sqrt(2)) - sp.Rational(2, 3)) == 0, "")
    # window boundary for eps = sqrt2: min_j f_j = 0 at |delta| = pi/12
    fmin = lambda d: min(1 + np.sqrt(2) * np.cos(d + 2 * np.pi * jj / 3) for jj in range(3))
    gate("AL2c positivity window |delta| <= pi/12 (boundary f_min = 0)",
         abs(fmin(np.pi / 12)) < 1e-12 and fmin(np.pi / 12 - 1e-6) > 0
         and fmin(np.pi / 12 + 1e-3) < 0,
         f"f_min(pi/12) = {fmin(np.pi/12):.2e}; 2/9 = {2/9:.4f} < pi/12 = {np.pi/12:.4f}")

    # (a, ratification-ordered) negative-z0 branch: positivity forces the sign
    neg_ok = True
    for d in np.linspace(0, 2 * np.pi / 3, 2001):
        f_neg = [-np.sqrt(0.5) + 2 * np.sqrt(0.25) * np.cos(d + 2 * np.pi * jj / 3)
                 for jj in range(3)]
        if min(f_neg) >= 0:
            neg_ok = False
    gate("AL2d negative-z0 branch admits NO positive completion (sign forced)",
         neg_ok, "0 admissible deltas on the negative branch")

    # (c, ratification-ordered) outside-window self-inconsistency: the canonical
    # root at delta = pi/6 yields weights != (1/2,1/4,1/4) -- the premise
    # self-breaks outside the window (no silent Q drift).
    d_out = np.pi / 6
    f_out = [1 + np.sqrt(2) * np.cos(d_out + 2 * np.pi * jj / 3) for jj in range(3)]
    sq = np.sqrt(np.abs(f_out))           # canonical root of |f|^2 masses
    z_out = [abs(np.mean(sq * np.array([np.exp(-2j * np.pi * jj * al / 3)
             for jj in range(3)]))) ** 2 for al in range(3)]
    z_out = np.array(z_out) / sum(z_out)
    gate("AL2e outside-window self-inconsistency: implied weights != (1/2,1/4,1/4)",
         np.abs(z_out - np.array([0.5, 0.25, 0.25])).max() > 1e-3,
         f"implied weights at delta=pi/6: {np.round(z_out, 4)}")

    # (iii) mixed-state validity: weights -> operator family, no coherence input
    wts = np.array([0.5, 0.25, 0.25])
    z = np.sqrt(wts)
    delta = 2 / 9
    f = [z[0] + 2 * z[1] * np.cos(delta + 2 * np.pi * jj / 3) for jj in range(3)]
    Q = sum(x**2 for x in f) / sum(f) ** 2
    gate("AL3 (iii) construction from WEIGHTS alone (mixed-state valid) -> Q = 2/3",
         abs(Q - 2 / 3) < TOL,
         "the rho = I/8 control is the lemma's PREDICTION, not a defect")

    # (iv) saddle dichotomy: eps = 2 -> window degenerates to delta = 0 (mod 2pi/3)
    fmin2 = lambda d: min(1 + 2 * np.cos(d + 2 * np.pi * jj / 3) for jj in range(3))
    interior_empty = all(fmin2(d) < -1e-9 for d in np.linspace(1e-3, 2 * np.pi / 3 - 1e-3, 500))
    f0 = [1 + 2 * np.cos(0 + 2 * np.pi * jj / 3) for jj in range(3)]
    gate("AL4 (iv) Gamma/H weights: window = {delta = 0 mod 2pi/3} only; spectrum (3,0,0)",
         abs(fmin2(0.0)) < 1e-12 and interior_empty
         and np.allclose(sorted(f0), [0, 0, 3], atol=1e-12),
         "no robust Koide at the neutrino saddles (measure-zero, doubly-degenerate-massless)")

    print("\n  K2 residue DISCHARGED: the aligned phases are Hermiticity of the")
    print("  mass observable; the state contributes magnitudes (Born weights)")
    print("  only. Remaining priced: mass=Born-weight id (~1 bit) +")
    print("  uniform-over-CSCO id (~1 bit). Verdict upgrade -> panel.")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print(" RESULT: ALL GATES PASS -- alignment lemma established")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
