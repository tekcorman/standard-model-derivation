#!/usr/bin/env python3
"""
needB_R4_pin_delta_target_2026-05-16.py

Need-B research target, Route R4 (bounded sub-step):
PIN the K-rational δ target the data requires — TRIANGULATION, NOT CLOSURE.

Per the scoping doc `an internal working note
_target_2026-05-16.md` §6/§7 and the load-bearing anti-numerology guardrail
(§5): this probe computes the phase δ (and the Koide ε) that the OBSERVED
down-quark mass triple requires, tests K=Q(√2,√3,√5)-rationality, and checks
the structural candidates {lepton 2/9, arg(h_P)/4}. It PINS a target for the
real routes R1/R2/R3. It is NOT a derivation of δ; reproducing the data here
is by construction (inversion), not evidence.

Inversion (exact, per generation→j ordering). Framework circulant-Koide form
  √m_j = M₀·(1 + ε·cos(2πj/3 + δ)),  j=0,1,2.
With ω=e^{2πi/3}:  Σ_j √m_j ω^j = M₀·ε·(3/2)·e^{-iδ}   (the "1" part: Σω^j=0).
⇒  M₀ = mean(√m_j);  ε = 2|F1|/(3 M₀) = 2|F1|/Σ√m_j;  δ = -arg(F1).
KEY: |F1| is invariant under relabeling ⇒ **ε is ordering-invariant** (a clean
robust deliverable). δ shifts by ±2π/3 (cyclic) / δ→-δ (reflection) under the
6 orderings ⇒ δ carries the data-anchored LABELING ambiguity (disclosed, not
hidden — it is exactly the non-blocking residue the unification reframed).

Self-check: the SAME inversion on (m_e,m_μ,m_τ) MUST reproduce the framework
lepton δ = 2/9 under the framework ordering, or the method/convention is wrong.
"""

import cmath, math

OMEGA = cmath.exp(2j * math.pi / 3)
TWO_NINTH = 2.0 / 9.0                                   # lepton δ (theorem-grade)
H_P = (math.sqrt(3) + 1j * math.sqrt(5)) / 2            # Row P52
ARG_H_OVER_4 = math.atan2(H_P.imag, H_P.real) / 4.0     # documented near-match
K_BASIS = {"√2": math.sqrt(2), "√3": math.sqrt(3), "√5": math.sqrt(5)}


def invert(masses):
    """masses in fixed (j=0,1,2) order → (M0, eps, delta_rad)."""
    r = [math.sqrt(m) for m in masses]
    F1 = sum(r[j] * OMEGA ** j for j in range(3))
    M0 = sum(r) / 3.0
    eps = 2.0 * abs(F1) / sum(r)
    delta = (-cmath.phase(F1)) % (2 * math.pi)
    return M0, eps, delta


def all_orderings_delta(masses):
    """6 perms → set of δ (rad, principal) — the labeling ambiguity."""
    from itertools import permutations
    out = []
    for p in permutations(range(3)):
        _, eps, d = invert([masses[i] for i in p])
        # fold to [0, 2π/3) since the form has 2π/3 + sign structure
        out.append((round(eps, 6), round(d % (2 * math.pi / 3), 6)))
    return sorted(set(out))


def k_rational_probe(x, max_den=24):
    """Nearest simple K=Q(√2,√3,√5) value to x (rad). Reporting only."""
    best = None
    for name, v in [("0", 0.0), ("2/9", TWO_NINTH), ("arg(h_P)/4", ARG_H_OVER_4)]:
        d = abs(x - v)
        if best is None or d < best[2]:
            best = (name, v, d)
    # small rationals q and q·(√k) probes
    for q_num in range(1, 13):
        for q_den in range(2, max_den + 1):
            for kn, kv in [("1", 1.0)] + list(K_BASIS.items()):
                val = (q_num / q_den) * kv
                if val > math.pi:
                    continue
                d = abs(x - val)
                if d < best[2]:
                    best = (f"{q_num}/{q_den}·{kn}", val, d)
    return best


def main():
    print("=" * 78)
    print("Need-B R4 — PIN the δ target (TRIANGULATION, NOT CLOSURE)")
    print("=" * 78)
    print("Guardrail: this inverts data→δ. Matching data here is BY")
    print("CONSTRUCTION, not evidence. δ is CLOSED only if R1/R2/R3 DERIVE")
    print("it structurally AND it reproduces lepton 2/9 by the same route.")
    print(f"\nReference candidates:  lepton δ = 2/9 = {TWO_NINTH:.6f} rad"
          f"  |  arg(h_P)/4 = {ARG_H_OVER_4:.6f} rad"
          f"  (gap {100*abs(TWO_NINTH-ARG_H_OVER_4)/TWO_NINTH:.2f}%)")

    # ---- SELF-CHECK: leptons must reproduce δ = 2/9 ---------------------
    m_e, m_mu, m_tau = 0.51099895e-3, 0.1056583755, 1.77686  # GeV
    print("\n[SELF-CHECK] lepton inversion (must hit δ = 2/9 under some")
    print("              ordering, else method/convention is wrong):")
    lep = all_orderings_delta([m_e, m_mu, m_tau])
    lep_eps = lep[0][0]
    hit = min(abs(d - TWO_NINTH) for _, d in lep)
    print(f"  ε(lepton, ordering-invariant) = {lep_eps:.6f}  (√2 = "
          f"{math.sqrt(2):.6f}; ε² = {lep_eps**2:.5f} vs 2)")
    print(f"  δ folded values: {sorted({d for _,d in lep})}")
    print(f"  closest to 2/9 = {TWO_NINTH:.6f}: Δ = {hit:.6f} rad "
          f"({'PASS' if hit < 0.02 else 'FAIL — method/ordering wrong'})")

    # ---- DOWN-QUARK scenarios (scale systematic disclosed) --------------
    # PDG-2024 MS-bar central; RG-run figures illustrative (dominant syst).
    scenarios = {
        "S1  μ=2 GeV  (m_b run→2GeV≈4.90)": (4.67e-3, 93.4e-3, 4.90),
        "S2  μ=m_b    (m_d,m_s run→m_b)":   (2.82e-3, 55.0e-3, 4.18),
        "S3  μ=M_Z    (all→M_Z)":           (2.75e-3, 55.0e-3, 2.90),
        "S4  GJ-textured leptons @GUT "
        "(m_d=3m_e, m_s=m_μ/3, m_b=m_τ)":   (3*m_e, m_mu/3, m_tau),
    }
    print("\n[DOWN-SECTOR] data-implied (ε, δ) per scenario "
          "(ε ordering-invariant; δ folded to [0,2π/3)):")
    print(f"  {'scenario':46s} {'ε²':>7s} {'δ(rad)':>9s}  nearest-K")
    deltas = []
    for name, mss in scenarios.items():
        _, eps, _ = invert(list(mss))
        folds = sorted({d for _, d in all_orderings_delta(list(mss))})
        # report the δ-fold closest to the structural candidates
        dstar = min(folds, key=lambda d: min(abs(d-TWO_NINTH),
                                             abs(d-ARG_H_OVER_4)))
        kn, kv, kd = k_rational_probe(dstar)
        deltas.append(dstar)
        print(f"  {name:46s} {eps**2:7.4f} {dstar:9.6f}  "
              f"≈{kn} ({kv:.5f}, Δ{kd:.4f})")

    # ---- VERDICT (honest; pin-not-close) -------------------------------
    print("\n" + "=" * 78)
    print("VERDICT — what R4 PINS (and what it does NOT)")
    print("=" * 78)
    eps2 = [invert(list(m))[1] ** 2 for m in scenarios.values()]
    print(f"• ε²_down (ordering-invariant) across scenarios: "
          f"[{min(eps2):.3f}, {max(eps2):.3f}].")
    near2 = any(abs(e - 2.0) < 0.3 for e in eps2)
    eps_msg = ("consistent with ε²≈2 in some scenarios" if near2 else
               "NOT ≈2 — the ε²=2 form does not fit raw down masses; the "
               "δ-pin is conditional on the (GJ-rotated/scale) form, which "
               "is itself part of Need-B")
    print(f"  Framework assumes ε²=2 (theorem-grade for the form). "
          f"Data: {eps_msg}.")
    band = (min(deltas), max(deltas))
    print(f"• δ_down band (struct-closest fold, scale systematic dominant): "
          f"[{band[0]:.4f}, {band[1]:.4f}] rad.")
    in_29 = band[0] - 0.03 <= TWO_NINTH <= band[1] + 0.03
    in_h4 = band[0] - 0.03 <= ARG_H_OVER_4 <= band[1] + 0.03
    msg_29 = "within band±syst" if in_29 else "OUTSIDE band"
    msg_h4 = ("within band±syst — SURVIVES as the R1/R2/R3 target" if in_h4
              else "OUTSIDE band — arg(h)/4 hypothesis in TENSION")
    print(f"  lepton 2/9   = {TWO_NINTH:.4f}: {msg_29}")
    print(f"  arg(h_P)/4   = {ARG_H_OVER_4:.4f}: {msg_h4}")
    print("\n• DOMINANT SYSTEMATICS (honest): (1) RG scale/running of the")
    print("  down masses; (2) the generation→j LABELING (data-anchored,")
    print("  non-blocking, reframed by the unification — δ defined mod the")
    print("  ordering). (3) Whether the framework's down form is raw-mass")
    print("  ε²=2 or GJ-rotated — UNRESOLVED, and itself a Need-B sub-question.")
    print("\n• STATUS: R4 = bounded sub-step COMPLETE. It PINS a target band")
    print("  + tests candidates. It does NOT close Need-B (no structural")
    print("  derivation of δ; reproducing data is by construction). Hand to")
    print("  R1 (triplet screw-axis Wigner-D) / R3 (diagonal G_NB reading)")
    print("  with this band as the number to hit.")
    print("=" * 78)


if __name__ == "__main__":
    main()
