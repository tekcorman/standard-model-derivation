"""
explore_t02 — the von Neumann algebra TYPE: does srs have intrinsic time? PURE MATH, walled.

The bare object's observable algebra is the Bloch band algebra L^∞(T³) ⊗ M_4(C) — a direct integral of
the fiber factors M_4. We show this is TYPE I: it carries a faithful normal trace (⇒ NOT type III) and
minimal projections (⇒ type I, not II). Consequence (Connes): on a type-I/semifinite algebra every
modular flow is INNER, so the canonical map ℝ → Out(M) is trivial ⇒ NO intrinsic (state-independent)
time. The t01 flow is therefore state-DEPENDENT. Intrinsic (type III) time would require added structure
(second quantization + a KMS state ⇒ an Araki–Woods factor).
"""
import numpy as np

rng = np.random.default_rng(0)
rmat = lambda: rng.standard_normal((4, 4)) + 1j*rng.standard_normal((4, 4))
x, y = rmat(), rmat()

# (1) faithful normal TRACE on the fiber M_4  =>  semifinite  =>  NOT type III
tr = np.trace
print("(1) Trace on the fiber M_4 (the matrix trace τ):")
print(f"    tracial  τ(xy)=τ(yx) ? {np.isclose(tr(x@y), tr(y@x))};   "
      f"faithful  τ(x*x)>0 for x≠0 ? {tr(x.conj().T@x).real > 0}")
print(f"    => a faithful normal trace exists  =>  the algebra is SEMIFINITE  =>  NOT type III.")

# (2) MINIMAL projections  =>  type I (not II)
v = rng.standard_normal(4) + 1j*rng.standard_normal(4); v /= np.linalg.norm(v)
p = np.outer(v, v.conj())                                   # rank-1 projector
is_proj = np.allclose(p@p, p) and np.allclose(p, p.conj().T)
abelian = all(np.allclose(p@(rmat())@p, (v.conj()@rmat()@v)*p) for _ in range(3)) or \
          all(np.allclose(p@M@p, (v.conj()@M@v)*p) for M in [rmat() for _ in range(20)])
print("\n(2) Minimal projections:")
print(f"    rank-1 p: is a projection ? {is_proj};   p·M_4·p = C·p  (p abelian/minimal) ? {abelian}")
print(f"    => minimal projections exist  =>  TYPE I (not type II, which has a trace but NO minimal proj).")

# (3) consequence: type I  =>  modular flow INNER  =>  state-dependent, not intrinsic
print("\n(3) Consequence (Connes' modular theory):")
print("    On a type-I / semifinite algebra, σ_t^ω = Ad((Dω/Dτ)^{it}) with (Dω/Dτ)^{it} INSIDE the algebra")
print("    (inner). So the canonical homomorphism ℝ → Out(M) = Aut/Inn is TRIVIAL.")
print("    => NO intrinsic (state-independent) time. The t01 modular flow is STATE-DEPENDENT — a choice of")
print("       state, not forced by the geometry.")

print("\n--- finding (time bridge, walled) ---")
print("  The srs band algebra L^∞(T³)⊗M_4 is TYPE I (faithful trace ⇒ not III; minimal projections ⇒ not II).")
print("  Structural reason: srs is an AMENABLE crystal (ℤ³-periodic) — Bloch theory makes it a direct integral")
print("  of matrix factors — so it is type I DESPITE the positive NB-walk entropy log 2 (amenability, not")
print("  entropy, fixes the vN type). Hence TIME IS STATE-DEPENDENT (inner flow), NOT intrinsic.")
print("  This parallels the matter bridge: the geometry gives the MECHANISM (the modular flow) but not a")
print("  forced/canonical time. Intrinsic (type III) time would need ADDED structure — second quantization")
print("  (the CAR field algebra) + a non-tracial KMS state ⇒ an Araki–Woods type III_λ factor. NOT done here.")
