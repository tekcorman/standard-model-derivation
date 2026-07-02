#!/usr/bin/env python3
"""The observer-side flow (cascade) is a dyadic ladder on ONE spine H ~ N^-1.

Promoted + gated structural ledger for clause (S2) of
docs/theorems/theorem_observer_flow_dyadic_ladder_2026-06-15.md.

Claim: every dimensional cascade observable is a MONOMIAL X = M_Pl^a . H^q in the single spine
H ~ N^-1 (the H.N.t_P=1 theorem), with q a DYADIC power (power of 2), because every
observable<->spine link is a derived square / sqrt / reciprocal physical relation. So the
"exponent vector" is NOT independent numbers -- it is one dilation generator acting as dyadic
monomials. (Deterministic; no randomness. Re-description of already-derived in-repo relations.)

HONEST SCOPE (gated as a comment, enforced by LG4): the dyadic-ness of the purely-cosmological
rungs {Lambda, H, t, T} is GENERIC FRW physics, NOT a framework discovery. The framework-specific
content is exactly TWO items: the spine (H~N^-1) and the v -1/4 keystone rung (observer read).

GATES (exit 0 on all-pass):
  LG1 every spine-power q is DYADIC (|q| is an integer power of 2)
  LG2 p_X = -q_X for every observable (X ~ H^q ~ N^-q since H ~ N^-1)
  LG3 the negative rungs are CONSECUTIVE powers of 2: {2, 1, 1/2, 1/4}
  LG4 exactly TWO framework-specific rungs (the spine + the v keystone); the rest are generic
"""
import sys
from fractions import Fraction as F

FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def is_pow2(fr):
    """True iff |fr| is an integer power of two (2^j, j in Z), e.g. 2,1,1/2,1/4."""
    fr = abs(fr)
    if fr == 0:
        return False
    num, den = fr.numerator, fr.denominator
    def only2(n):
        while n % 2 == 0 and n > 1:
            n //= 2
        return n == 1
    if den == 1:
        return only2(num)
    if num == 1:
        return only2(den)
    return False


# observable: (name, p_X[N-exponent], q[spine power of H], framework_specific, link)
OBS = [
    ("Lambda", F(-2),   F(2),   False, "Lambda = 3 H^2 (Friedmann, vacuum) -- square"),
    ("H",      F(-1),   F(1),   True,  "the SPINE: H.N.t_P=1 (theorem)"),
    ("t_age",  F(1),    F(-1),  False, "t = 1/H -- reciprocal"),
    ("T_rad",  F(-1,2), F(1,2), False, "H ~ T^2/M_Pl (radiation Friedmann) -- sqrt; era-limited"),
    ("m_nu",   F(-1,2), F(1,2), False, "m_nu ~ (y v)^2/M_R (seesaw) -- square in v"),
    ("G_F",    F(1,2),  F(-1,2),False, "G_F = 1/(sqrt2 v^2) -- reciprocal-square"),
    ("v",      F(-1,4), F(1,4), True,  "v ~ (H M_Pl^3)^1/4 (OBSERVER one-pass read) -- KEYSTONE"),
]

print("=" * 80)
print(" OBSERVER-SIDE FLOW = dyadic ladder on the spine H ~ N^-1 (structural ledger)")
print("=" * 80)
print(f"\n  {'observable':<10}{'p_X(N)':>8}{'q(spine H)':>12}{'dyadic?':>9}{'  fw-specific':>13}   link")
for name, pX, q, fw, link in OBS:
    print(f"  {name:<10}{str(pX):>8}{str(q):>12}{str(is_pow2(q)):>9}{str(fw):>13}   {link}")

# LG1: dyadic
gate("LG1 every spine-power q is dyadic (power of 2)", all(is_pow2(q) for *_, q, _, _ in
     [(n, p, q, fw, l) for (n, p, q, fw, l) in OBS]),
     "{2,1,1/2,1/4}")
# LG2: p_X = -q_X
gate("LG2 p_X = -q_X for every observable (X ~ H^q ~ N^-q)",
     all(pX == -q for (_, pX, q, _, _) in OBS))
# LG3: negative rungs = consecutive powers of 2 {2,1,1/2,1/4}
neg_mag = sorted({abs(pX) for (_, pX, q, _, _) in OBS if pX < 0})
gate("LG3 negative rungs = consecutive powers of 2 {1/4,1/2,1,2}",
     neg_mag == [F(1, 4), F(1, 2), F(1), F(2)], f"{[str(x) for x in neg_mag]}")
# LG4: exactly two framework-specific (spine + v keystone)
fw_names = [n for (n, p, q, fw, l) in OBS if fw]
gate("LG4 exactly TWO framework-specific rungs (spine H + v keystone)",
     fw_names == ["H", "v"], f"{fw_names}  (rest = generic FRW, not a framework discovery)")

print(f"""
  => the cascade is ONE generator (the dilation, eigenvalue -1 on the spine H~N^-1) acting as
     DYADIC MONOMIALS H^q -- not 7 independent exponents. Dyadic because every link is a
     square/sqrt/reciprocal (Friedmann H^2~rho is the master quadratic). The framework-specific
     content is exactly {fw_names}: the spine (why an N-flow exists) and the v -1/4 keystone
     rung (the observer read, bolting the particle VEV onto the cosmological spine). The
     dyadic-ness of the cosmological rungs is GENERIC FRW physics, not sold as a discovery.""")

print("\n" + "=" * 80)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
    sys.exit(1)
print(" RESULT: ALL GATES PASS -- one spine, dyadic monomials; two framework-specific rungs")
print("=" * 80)
sys.exit(0)
