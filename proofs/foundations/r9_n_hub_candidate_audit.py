#!/usr/bin/env python3
"""
R-9 closure attempt — N_hub structural candidate audit.

Question: does any low-K-complexity dimensionless combination of framework
primitives reach the empirical N_hub ≈ 8.395×10⁶⁰ within sub-1σ
precision (~0.02%)?

If YES with a unique winner backed by a structural defining equation
→ candidate for upgrading Row P17 from anchor-dependent to
  UNIQUE-THEOREM-GRADE-STRUCTURAL.
If NO unique winner OR no candidate in band → honest negative; the
  framework's N value is genuinely external in the sense that no
  small-K combination of primitives reaches it.

Reference: `proofs/foundations/r9_srs_z_polynomial_derivation.py` (R-9
template applied to Wyckoff free parameter); scoping doc
an internal working note.
"""

import math
import itertools
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Empirical target
# ---------------------------------------------------------------------------

# Anchor-derived N_hub (G_F → BZJ inversion; see predictions/N_hub.py).
G_F   = 1.1663787e-5
M_P   = 1.22089e19
delta = 2.0 / 9.0
alpha_1 = (2.0 / 3.0) ** 8
v_GF  = 1.0 / math.sqrt(math.sqrt(2) * G_F)
dark  = 1.0 - (5.0 / 12.0) * alpha_1 / (1.0 - alpha_1)
N_hub_obs = (delta**2 * M_P * dark / (math.sqrt(2) * v_GF)) ** 4

# Anchor-propagation precision: dN/N ≈ 4·dM_P/M_P + 4·dv/v
#   M_P ~ 50 ppm (CODATA 2018 — derived from G_N at 22 ppm + ℏ, c exact).
#   the value of the adopted N_hub inherits ~0.25 ppm from the measured G_F (the calibrating observable).
# Total: ~200 ppm = 0.02%.
sigma_rel = 2.0e-4   # ~0.02% one-sigma in N_hub anchor

ln_N = math.log(N_hub_obs)
log10_N = math.log10(N_hub_obs)


# ---------------------------------------------------------------------------
# Framework primitive basis
# ---------------------------------------------------------------------------

@dataclass
class Primitive:
    name: str
    value: float
    ln_val: float

    @classmethod
    def make(cls, name, value):
        return cls(name=name, value=value, ln_val=math.log(value))


PRIMITIVES = [
    Primitive.make("2", 2.0),
    Primitive.make("3", 3.0),
    Primitive.make("k*", 3.0),                    # alias of 3
    Primitive.make("g", 10.0),                    # girth
    Primitive.make("d_spatial", 3.0),             # alias of 3
    Primitive.make("delta", 2.0 / 9.0),           # = 2/9
    Primitive.make("1/delta", 9.0 / 2.0),
    Primitive.make("alpha_1", alpha_1),           # (2/3)^8
    Primitive.make("1/alpha_1", 1.0 / alpha_1),
    Primitive.make("5/12", 5.0 / 12.0),
    Primitive.make("12/5", 12.0 / 5.0),
    Primitive.make("8/sqrt(pi)", 8.0 / math.sqrt(math.pi)),  # M_P/M_substrate (Drude)
    Primitive.make("sqrt(pi)/8", math.sqrt(math.pi) / 8.0),
    Primitive.make("pi", math.pi),
    Primitive.make("1/pi", 1.0 / math.pi),
    Primitive.make("pi/64", math.pi / 64.0),       # G_UV·M_sub^2
    Primitive.make("64/pi", 64.0 / math.pi),
    Primitive.make("sqrt(2)", math.sqrt(2.0)),
    Primitive.make("sqrt(3)", math.sqrt(3.0)),
    Primitive.make("sqrt(5)", math.sqrt(5.0)),
    Primitive.make("Gamma(5/4)", math.gamma(5.0 / 4.0)),
    Primitive.make("e", math.e),                   # transcendental control
    Primitive.make("4", 4.0),                      # N_atoms primitive cell
    Primitive.make("8", 8.0),                      # N_orbit
    Primitive.make("2/3", 2.0 / 3.0),              # bare survival ratio
    Primitive.make("3/2", 1.5),
]

BASIS_SIZE = len(PRIMITIVES)
BIT_PER_PRIM = math.log2(BASIS_SIZE)


def gamma2_bits(num_factors: int, max_abs_exp: int) -> float:
    """
    γ.2 algebraic-K-complexity analogue for multiplicative expressions.
    Each factor: log2(BASIS_SIZE) bits to pick the primitive.
    Each exponent: log2(2·|exp|+1) bits to encode signed integer.
    """
    if num_factors == 0:
        return 0.0
    return num_factors * BIT_PER_PRIM + num_factors * math.log2(2 * max(max_abs_exp, 1) + 1)


# ---------------------------------------------------------------------------
# Family A — single-primitive integer powers   N = p^k
# ---------------------------------------------------------------------------

def family_A():
    cands = []
    for p in PRIMITIVES:
        if p.value <= 1.0 + 1e-9:
            continue
        # k = ln(N)/ln(p); take nearest integer and check residual
        k_real = ln_N / p.ln_val
        for k in [math.floor(k_real), math.ceil(k_real)]:
            if k < 1:
                continue
            value = p.value ** k
            sigma = abs(value - N_hub_obs) / (N_hub_obs * sigma_rel)
            bits = gamma2_bits(1, abs(k))
            cands.append({
                "family": "A",
                "expr":   f"{p.name}^{k}",
                "value":  value,
                "sigma":  sigma,
                "bits":   bits,
            })
    return cands


# ---------------------------------------------------------------------------
# Family B — two-primitive products  N = p1^a · p2^b
# ---------------------------------------------------------------------------

def family_B(max_abs_exp=200):
    """
    Sweep a, b over integer ranges; match ln N to a·ln(p1) + b·ln(p2).
    For each (p1, p2), find integer (a, b) closest to the linear fit.
    Then refine: enumerate small (a, b) deviations and report sub-1σ matches.
    """
    cands = []
    primitives = [p for p in PRIMITIVES if abs(p.ln_val) > 1e-9]
    for p1, p2 in itertools.combinations(primitives, 2):
        # Solve: a·ln(p1) + b·ln(p2) = ln(N)
        # 1-parameter family in continuous (a, b); enumerate b ∈ [-max, max]
        for b in range(-max_abs_exp, max_abs_exp + 1):
            a_real = (ln_N - b * p2.ln_val) / p1.ln_val
            for a in [math.floor(a_real), math.ceil(a_real)]:
                if abs(a) > max_abs_exp or a == 0 and b == 0:
                    continue
                if abs(a) > max_abs_exp:
                    continue
                ln_val = a * p1.ln_val + b * p2.ln_val
                if abs(ln_val - ln_N) > math.log(2):  # crude pre-filter > factor of 2
                    continue
                value = math.exp(ln_val)
                sigma = abs(value - N_hub_obs) / (N_hub_obs * sigma_rel)
                if sigma > 50:  # report only candidates within 50σ
                    continue
                bits = gamma2_bits(2, max(abs(a), abs(b)))
                cands.append({
                    "family": "B",
                    "expr":   f"{p1.name}^{a} * {p2.name}^{b}",
                    "value":  value,
                    "sigma":  sigma,
                    "bits":   bits,
                })
    return cands


# ---------------------------------------------------------------------------
# Family C — three-primitive products  N = p1^a · p2^b · p3^c
# ---------------------------------------------------------------------------

def family_C(max_abs_exp=120, sigma_cap=10):
    cands = []
    primitives = [p for p in PRIMITIVES if abs(p.ln_val) > 1e-9]
    triples = list(itertools.combinations(primitives, 3))
    for p1, p2, p3 in triples:
        # For each (b, c) in a moderate range, solve a from ln(N) - b·ln(p2) - c·ln(p3)
        for b in range(-max_abs_exp, max_abs_exp + 1, 4):
            for c in range(-max_abs_exp, max_abs_exp + 1, 4):
                a_real = (ln_N - b * p2.ln_val - c * p3.ln_val) / p1.ln_val
                for a in [math.floor(a_real), math.ceil(a_real)]:
                    if abs(a) > max_abs_exp:
                        continue
                    ln_val = a * p1.ln_val + b * p2.ln_val + c * p3.ln_val
                    if abs(ln_val - ln_N) > 0.05:  # within ~5%
                        continue
                    value = math.exp(ln_val)
                    sigma = abs(value - N_hub_obs) / (N_hub_obs * sigma_rel)
                    if sigma > sigma_cap:
                        continue
                    bits = gamma2_bits(3, max(abs(a), abs(b), abs(c)))
                    cands.append({
                        "family": "C",
                        "expr":   f"{p1.name}^{a} * {p2.name}^{b} * {p3.name}^{c}",
                        "value":  value,
                        "sigma":  sigma,
                        "bits":   bits,
                    })
    return cands


# ---------------------------------------------------------------------------
# Family D — single-primitive non-integer powers (rational exponents)
# ---------------------------------------------------------------------------

def family_D(max_denom=12):
    """
    Test N = p^(m/n) for small rational m/n.
    """
    cands = []
    for p in PRIMITIVES:
        if p.value <= 1.0 + 1e-9:
            continue
        target = ln_N / p.ln_val   # the real-valued "m/n"
        for n in range(1, max_denom + 1):
            m_real = target * n
            for m in [math.floor(m_real), math.ceil(m_real)]:
                if m == 0:
                    continue
                exponent = m / n
                value = p.value ** exponent
                sigma = abs(value - N_hub_obs) / (N_hub_obs * sigma_rel)
                if sigma > 5:
                    continue
                # Bit cost: 1 primitive + (log2(2|m|+1) + log2(2n+1)) bits for rational
                bits = BIT_PER_PRIM + math.log2(2 * abs(m) + 1) + math.log2(2 * n + 1)
                cands.append({
                    "family": "D",
                    "expr":   f"{p.name}^({m}/{n})",
                    "value":  value,
                    "sigma":  sigma,
                    "bits":   bits,
                })
    return cands


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

def main():
    print("=" * 96)
    print("R-9 closure attempt — N_hub structural candidate audit")
    print("=" * 96)
    print(f"\n  Empirical target:  N_hub (adopted) = {N_hub_obs:.6e}  (value pinned via the measured G_F)")
    print(f"  ln(N_hub_obs)             = {ln_N:.6f}")
    print(f"  log10(N_hub_obs)          = {log10_N:.4f}")
    print(f"  Anchor 1σ relative:       ~{sigma_rel:.0e}  (~{sigma_rel*100:.2f}%)")
    print(f"  Sub-1σ match band:        N ∈ [{N_hub_obs*(1-sigma_rel):.4e}, {N_hub_obs*(1+sigma_rel):.4e}]")
    print(f"\n  Basis: {BASIS_SIZE} primitives  ({BIT_PER_PRIM:.2f} bits/primitive selection)")

    all_cands = []
    print("\n" + "-" * 96)
    print("  Family A — single-primitive integer powers")
    print("-" * 96)
    A = family_A()
    A_sub3 = [c for c in A if c["sigma"] < 3]
    A.sort(key=lambda c: c["sigma"])
    for c in A[:8]:
        print(f"    σ={c['sigma']:8.2f}  bits={c['bits']:5.2f}  {c['expr']:30s}  N={c['value']:.4e}")
    all_cands.extend(A)

    print("\n" + "-" * 96)
    print("  Family B — two-primitive integer-power products  (sub-50σ only)")
    print("-" * 96)
    B = family_B(max_abs_exp=200)
    B.sort(key=lambda c: c["sigma"])
    print(f"    [{len(B)} sub-50σ candidates total]")
    for c in B[:10]:
        print(f"    σ={c['sigma']:8.2f}  bits={c['bits']:5.2f}  {c['expr']:50s}  N={c['value']:.4e}")
    all_cands.extend(B)

    print("\n" + "-" * 96)
    print("  Family D — single-primitive rational powers (small denominator)")
    print("-" * 96)
    D = family_D(max_denom=12)
    D.sort(key=lambda c: c["sigma"])
    print(f"    [{len(D)} sub-5σ candidates]")
    for c in D[:10]:
        print(f"    σ={c['sigma']:8.2f}  bits={c['bits']:5.2f}  {c['expr']:30s}  N={c['value']:.4e}")
    all_cands.extend(D)

    print("\n" + "-" * 96)
    print("  Family C — three-primitive integer-power products  (sub-10σ only)")
    print("-" * 96)
    C = family_C(max_abs_exp=120, sigma_cap=10)
    C.sort(key=lambda c: c["sigma"])
    print(f"    [{len(C)} sub-10σ candidates total — showing 12 lowest σ + 12 lowest bits]")
    for c in C[:12]:
        print(f"    σ={c['sigma']:8.4f}  bits={c['bits']:5.2f}  {c['expr']:60s}  N={c['value']:.4e}")
    print()
    by_bits = sorted(C, key=lambda c: (c["bits"], c["sigma"]))
    for c in by_bits[:12]:
        print(f"    bits={c['bits']:5.2f}  σ={c['sigma']:8.4f}  {c['expr']:60s}  N={c['value']:.4e}")
    all_cands.extend(C)

    print("\n" + "=" * 96)
    print("  VERDICT — sub-1σ candidates by bit cost")
    print("=" * 96)
    sub1 = [c for c in all_cands if c["sigma"] < 1.0]
    sub1.sort(key=lambda c: c["bits"])
    print(f"\n  Total sub-1σ candidates: {len(sub1)}")
    if not sub1:
        print("  (NONE — no low-K combination of primitives reaches N_hub within ~0.02%)")
    else:
        for c in sub1[:25]:
            print(f"    bits={c['bits']:5.2f}  σ={c['sigma']:6.4f}  [{c['family']}]  {c['expr']:60s}  N={c['value']:.4e}")

    # MDL closure analysis
    print("\n" + "-" * 96)
    print("  MDL CLOSURE ANALYSIS")
    print("-" * 96)
    if not sub1:
        print("""
  No sub-1σ candidate at any K-complexity in the families surveyed.
  Verdict: outcome band (C) — N is structurally external in the sense
  that no small-K combination of framework primitives reaches the
  empirical value within anchor precision.
""")
    else:
        # Bits to encode N to anchor precision (sigma_rel = 2e-4) from scratch:
        bits_to_encode_N_raw = math.log2(N_hub_obs * sigma_rel)
        print(f"  Bits to encode N to anchor precision (sigma_rel = {sigma_rel}):")
        print(f"    log2(N · σ_rel) = log2({N_hub_obs:.3e} · {sigma_rel}) = {bits_to_encode_N_raw:.2f} bits")
        print()
        cheapest = sub1[0]
        margin = bits_to_encode_N_raw - cheapest["bits"]
        print(f"  Cheapest sub-1σ candidate: {cheapest['expr']}")
        print(f"    bits = {cheapest['bits']:.2f},  σ = {cheapest['sigma']:.4f}")
        print(f"    Compression margin vs raw-anchor: {margin:.2f} bits")
        print()
        # Closure threshold: "MDL prefers structural" needs structural cheaper
        # AND uniquely below alternatives by enough to be selection-decisive.
        same_bits = [c for c in sub1 if c["bits"] <= cheapest["bits"] + 1.0]
        print(f"  Candidates within 1 bit of cheapest: {len(same_bits)}")
        if len(same_bits) > 1:
            print("  → MDL DEGENERATE: multiple candidates at similar K-complexity")
            print("    Outcome band (B) — gap is genuine; near-fits are post-hoc.")
        elif margin > 12.0:
            print("  → CANDIDATE WINNER: bits << raw-anchor + uniquely cheapest at this complexity.")
            print("    Outcome band (A): structural-derivation candidate (subject to verifying")
            print("    the candidate has a derivable defining equation, not just numerical match).")
        else:
            print("  → MARGIN INSUFFICIENT: structural compression too small to displace anchor.")
            print("    Outcome band (B) — gap is genuine.")


if __name__ == "__main__":
    main()
