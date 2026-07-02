"""
proofs/foundations/dark_feshbach_alpha_GUT_g_audit_2026-05-11.py

RE-AUDIT: is 5/12 = α_GUT·g the same structural object as the existing
marginal-Hashimoto-mode derivation c = (2(|E|−|V|)+1)/(2|E|), or a distinct
second derivation?

The two forms (both = 5/12 for srs):
  Form 1 (existing): c = (2(|E|−|V|)+1)/(2|E|) = (2·2+1)/12 = 5/12
                       — fraction of Hashimoto modes in the marginal cycle space
  Form 2 (new):     c = α_GUT·g = (1/24)·10 = 10/24 = 5/12
                       = g/(|V|·|E|) = (|V|+|E|)/(|V|·|E|) = 1/|V| + 1/|E|
                       — using α_GUT = 1/(|V|·|E|) and g = |V|+|E| = k*²+1

Test:
  1. Are the two forms equal as FUNCTIONS of (|V|, |E|, g), or only at the
     srs values?
  2. Does Form 2 have a MECHANISM (a reason the dark correction should equal
     α_GUT·g), or is it just a numerical re-expression?
  3. Verdict: distinct second derivation, same structural object, or
     re-expression-without-mechanism.
"""

import math
import sys
from pathlib import Path
from fractions import Fraction

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    print("=" * 100)
    print("RE-AUDIT: 5/12 = α_GUT·g vs marginal-Hashimoto-mode derivation")
    print("=" * 100)
    print()

    V, E, k_star, g = 4, 6, 3, 10

    print(f"  srs primitives: |V| = {V}, |E| = {E}, k* = {k_star}, g = {g}")
    print(f"  srs-specific relations:")
    print(f"    k* = 2|E|/|V| = {2*E}/{V} = {2*E//V} ✓")
    print(f"    g = k*² + 1 = {k_star**2}+1 = {k_star**2+1} ✓ (Moore-bound saturation)")
    print(f"    g = |V| + |E| = {V}+{E} = {V+E} ✓ (Euler-like for K_4 quotient)")
    print(f"    |V|·|E| = {V*E} = 2^k*·k* = {2**k_star}·{k_star} = {2**k_star*k_star} ✓ (Fock×directions)")
    print(f"    |E| − |V| = {E-V}; cycle rank β_1 = |E|−|V|+1 = {E-V+1}")
    print()

    # Form 1: marginal Hashimoto modes
    print(f"  Form 1 (existing derivation): c = (2(|E|−|V|)+1)/(2|E|)")
    num1 = 2*(E-V)+1
    den1 = 2*E
    c1 = Fraction(num1, den1)
    print(f"    = (2·{E-V}+1)/{den1} = {num1}/{den1} = {c1}")
    print(f"    Mechanism: the Hashimoto operator B on 2|E| directed edges has")
    print(f"    a mode decomposition; the 'marginal cycle space' (modes that")
    print(f"    contribute to the dark correction) has dimension 2(|E|−|V|)+1")
    print(f"    = {num1}. The dark correction strength = (marginal modes)/(total")
    print(f"    directed edges) = {num1}/{den1} = 5/12. This is a SPECTRAL count.")
    print(f"    Source: F7 / dark-correction apparatus (theorem_dark_5_12_spectral).")
    print()

    # Form 2: α_GUT·g
    print(f"  Form 2 (new): c = α_GUT·g where α_GUT = 1/(|V|·|E|) = 1/{V*E}")
    alpha_GUT = Fraction(1, V*E)
    c2 = alpha_GUT * g
    print(f"    = (1/{V*E})·{g} = {g}/{V*E} = {c2}")
    print(f"    = g/(|V|·|E|) = (|V|+|E|)/(|V|·|E|) [using g = |V|+|E|]")
    print(f"    = 1/|V| + 1/|E| = 1/{V} + 1/{E} = {Fraction(1,V)+Fraction(1,E)}")
    print(f"    'Mechanism': ??? — α_GUT is the unification coupling (inverse Fock×")
    print(f"    directions count); g is the girth. Why would the dark correction")
    print(f"    equal (unification coupling) × (girth)? No mechanism identified.")
    print()

    # Test 1: equal as functions of (|V|, |E|, g)?
    print("=" * 100)
    print("Test 1 — are Form 1 and Form 2 equal as functions of (|V|, |E|, g)?")
    print("=" * 100)
    print()
    print(f"  Form 1 = Form 2:")
    print(f"    (2(|E|−|V|)+1)/(2|E|) = g/(|V|·|E|)")
    print(f"    Cross-multiply by |V|·|E|: (2(|E|−|V|)+1)·|V|/(2) = g")
    print(f"    i.e., |V|·(2(|E|−|V|)+1) = 2g")
    print(f"    i.e., 2|V|·|E| − 2|V|² + |V| = 2g")
    lhs = 2*V*E - 2*V**2 + V
    rhs = 2*g
    print(f"    For srs: 2·{V}·{E} − 2·{V}² + {V} = {2*V*E} − {2*V**2} + {V} = {lhs}")
    print(f"             2g = 2·{g} = {rhs}")
    print(f"    {lhs} == {rhs}? {'YES ✓' if lhs == rhs else 'NO ✗'}")
    print()
    print(f"  → The two forms agree for srs, but this REQUIRES the constraint")
    print(f"    2|V|·|E| − 2|V|² + |V| = 2g, which is NOT a generic-graph identity.")
    print(f"    For srs it holds because of the srs-specific relations")
    print(f"    (|E| = |V|k*/2, g = k*²+1, |V| = 2(k*²+1)/(k*+2), all from k*=3).")
    print()
    # Show it fails for a hypothetical other graph
    print(f"  Check on a hypothetical k*=4 graph (|V|=?, |E|=2|E|... say |V|=6, |E|=12, g=17):")
    V2, E2, g2 = 6, 12, 17  # k*=4 → g=k*²+1=17 by Moore saturation; |E|=|V|k*/2 needs |V|=2E/k*=6
    lhs2 = 2*V2*E2 - 2*V2**2 + V2
    rhs2 = 2*g2
    print(f"    2|V|·|E| − 2|V|² + |V| = {2*V2*E2} − {2*V2**2} + {V2} = {lhs2}")
    print(f"    2g = {rhs2}")
    print(f"    Equal? {'YES' if lhs2 == rhs2 else 'NO ✗ — the identity is srs-specific'}")
    print(f"    (Also: does |V|=6 satisfy |V| = 2(k*²+1)/(k*+2) = 2·17/6 = 34/6? No — so")
    print(f"     k*=4 doesn't even give a consistent crystal-net parameter set this way.")
    print(f"     The srs parameters are tightly constrained by k*=3.)")
    print()

    # Test 2: mechanism?
    print("=" * 100)
    print("Test 2 — does Form 2 (α_GUT·g) have a mechanism?")
    print("=" * 100)
    print()
    print(f"  Form 1's mechanism: dark correction = (marginal Hashimoto modes)/(total")
    print(f"    directed edges). This is a clear spectral-counting argument: the")
    print(f"    Hashimoto operator's mode structure splits into 'tree' modes (tied to")
    print(f"    |V|) and 'cycle' modes; the marginal cycle space has dim 2(|E|−|V|)+1;")
    print(f"    the dark correction picks up this fraction. The mechanism is in the")
    print(f"    F7 / dark-correction-theorem apparatus.")
    print()
    print(f"  Form 2 ('α_GUT·g'): is there a reason the dark correction = (unification")
    print(f"    coupling) × (girth)? α_GUT = 1/(|V|·|E|) counts Fock states × directions.")
    print(f"    g counts the smallest cycle. The product g/(|V|·|E|) = (smallest cycle)/")
    print(f"    (Fock×directions). There is NO MECHANISM connecting 'dark correction' to")
    print(f"    'cycle-length / Fock-direction-count'. It's a numerical re-expression")
    print(f"    that holds because of the srs-specific constraint above.")
    print()
    print(f"  (Compare: cos²(arg h_P) = sin²θ_W was also 'true because k*=3' with no")
    print(f"   shared mechanism. The 5/12 = α_GUT·g case is the same flavor — except")
    print(f"   here BOTH sides are substrate-counting quantities, so it's an INTERNAL")
    print(f"   redundancy of the substrate-counting algebra rather than a")
    print(f"   substrate-quantity-equals-physical-observable coincidence.)")
    print()

    # Verdict
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"""
  5/12 = α_GUT·g is STRUCTURALLY TRUE in the framework (both sides derive
  from substrate primitives), but it is NOT a new derivation of the dark
  Feshbach factor — it is a NUMERICAL RE-EXPRESSION within the
  substrate-counting algebra, holding by virtue of the srs-specific
  constraint 2|V|·|E| − 2|V|² + |V| = 2g (a consequence of k*=3 forcing
  |V|=4, |E|=6, g=10).

  - Form 1 (marginal Hashimoto modes / directed edges) is THE derivation —
    it has a clear spectral-counting mechanism.
  - Form 2 (α_GUT·g = g/(|V|·|E|) = 1/|V| + 1/|E|) is a re-expression with
    NO identified mechanism (no reason the dark correction should equal
    unification-coupling × girth).
  - The two are NOT 'two independent derivations' — Form 2 has no
    derivation chain; it's an observed equality.
  - Same flavor as cos²(arg h_P) = sin²θ_W: a real numerical fact, forced
    by the substrate commitment (here srs/k*=3), but not a structural
    identity linking two mechanisms. The difference: 5/12 = α_GUT·g links
    counting-to-counting (internal algebra redundancy), whereas
    cos²(arg h_P) = sin²θ_W linked spectral-to-rep-theory (more surprising).

  RECOMMENDATION:
  - Keep the marginal-Hashimoto-mode derivation as THE derivation of c = 5/12.
  - Record '5/12 = α_GUT·g = 1/|V| + 1/|E|' in the substrate-formula catalog
    as a noted internal identity (useful for cross-checks), flagged as a
    re-expression, not a derivation.
  - Do NOT claim a 'second derivation of the dark Feshbach factor'.
  - GENERAL LESSON: the substrate-counting algebra (primitives |V|=4, |E|=6,
    k*=3, g=10, tightly constrained by k*=3 via k* = 2|E|/|V|, g = k*²+1,
    g = |V|+|E|, |V|·|E| = 2^k*·k*) has many internal numerical
    coincidences. Most 'X = Y' identities found by pattern-matching are
    re-expressions, not new mechanisms. To be a new derivation, an identity
    needs an independent mechanism on BOTH sides — not just numerical equality.
""")


if __name__ == "__main__":
    main()
