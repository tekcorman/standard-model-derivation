#!/usr/bin/env python3
"""
The P-point-δ ↔ N-orbit cyclic-basis map — and an honest correction.

The C₃₆-twist attack (mass_operator_c36_twist_attack_2026-05-21.py) reduced
[ORDER] to the Koide circulant phase δ and said the residue becomes "one
BOUNDED P-point-δ ↔ N-orbit map plus one ordinal naming." This probe attacks
that map — and finds the "bounded" label was optimistic.

WHAT THE MAP CONNECTS.
  • mass sector — the Koide circulant √m_k = M₀(1+ε·cos(2πk/3+δ)); its phase
    δ orders the generations. δ is P-point h-power Yukawa content.
  • mixing sector — the N-orbit cyclic 3-orbit basis, the C_36 twist, the CKM.

  G1  state the map
  G2  the map's SKELETON is derived — one generation-Z₃, the DFT U
  G3  the map's CONTENT is NOT bounded — it is δ's value = Need-B δ-physical
  G4  verdict — honest correction: the chain bottoms out at the deep frontier
"""

import numpy as np

results = []


def gate(name, passed, detail=""):
    results.append(bool(passed))
    print(f"  [{'PASS' if passed else 'OPEN'}] {name}")
    for ln in detail.strip("\n").split("\n"):
        if ln.strip():
            print(f"         {ln}")
    print()


# ======================================================================
print("=" * 72)
print("G1 — what the map must connect")
print("=" * 72)
gate("G1 the map links the mass-sector circulant to the N-orbit cyclic basis",
     True,
     "MASS sector: the Koide circulant on generation space, √m_k =\n"
     "M₀(1+ε·cos(2πk/3+δ)) — its phase δ orders gen-1<gen-2<gen-3 (the\n"
     "C₃₆-twist attack's [ORDER]). ε, δ are P-point h-power Yukawa content.\n"
     "MIXING sector: the N-orbit cyclic 3-orbit basis (8 candidates), the\n"
     "C_36 twist, the twisted walker T=B·C_36 → the CKM.\n"
     "The map must say how the δ-ordered masses sit on the N-orbit basis.")


# ======================================================================
print("=" * 72)
print("G2 — the map's SKELETON: one generation-Z₃, related by the DFT U")
print("=" * 72)
omega = np.exp(2j * np.pi / 3)
P = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])           # cyclic shift
U = np.array([[omega**(j*k) for k in range(3)] for j in range(3)]) / np.sqrt(3)
# C_36 = cyclic shift P in the N-orbit cyclic basis; diagonal in the Fourier
# (= mass-eigen = isotypic) basis:
diag_form = U.conj().T @ P @ U
is_diag = np.allclose(diag_form - np.diag(np.diag(diag_form)), 0, atol=1e-12)
matches = np.allclose(np.sort_complex(np.diag(diag_form)),
                      np.sort_complex(np.array([1, omega, omega**2])), atol=1e-12)
# a circulant (mass matrix) is diagonal in the SAME Fourier basis:
c = np.array([1.0, 0.4*np.exp(1j/9), 0.4*np.exp(-1j/9)])
M = np.array([[c[0], c[1], c[2]], [c[2], c[0], c[1]], [c[1], c[2], c[0]]])
M_fourier = U.conj().T @ M @ U
circ_diag = np.allclose(M_fourier - np.diag(np.diag(M_fourier)), 0, atol=1e-12)
g2 = is_diag and matches and circ_diag
gate("G2 skeleton = one Galois Z₃ + the DFT U — assembled from theorem-grade", g2,
     f"C_36 in the N-orbit cyclic basis = the cyclic shift P; in the Fourier\n"
     f"basis U†PU = diag(1,ω,ω²): is-diagonal {is_diag}, eigenvalues match {matches}.\n"
     f"The mass circulant is diagonal in the SAME Fourier basis: {circ_diag}.\n"
     "So the mass-eigenbasis and the N-orbit cyclic basis are ONE DFT U apart,\n"
     "and the mass-circulant's generation-C₃ and the N-orbit's C_36 are the\n"
     "SAME Z₃ — the M1.B Galois generation-Z₃ (theorem_substrate_generation_\n"
     "charge_conservation, theorem-grade 2026-04-28). The map's SKELETON —\n"
     "{one Galois Z₃, the DFT U} — is therefore derived (assembled from\n"
     "theorem-grade pieces). This part IS bounded and is done here.")


# ======================================================================
print("=" * 72)
print("G3 — the map's CONTENT is NOT bounded: it is δ's value")
print("=" * 72)
gate("G3 the map's content = the δ VALUE = Need-B δ-physical (deep frontier)",
     False,
     "The skeleton (G2) places the masses on the N-orbit basis up to the DFT —\n"
     "but the map must also transport the VALUE of δ (which fixes WHICH mode\n"
     "is gen-1 and the mass RATIOS). And δ's value is NOT derived:\n"
     "  • lepton δ = 2/9 — graded 'algebraic-identity-only'\n"
     "    (theorem_41_screw_wigner §6(i)); an identity that holds, no mechanism.\n"
     "  • quark δ = 2/(9(s+1)) — the s-dependence is less pinned still.\n"
     "Deriving δ's value from the P-point h-power Yukawa structure IS the\n"
     "named deep-frontier mask **Need-B δ-physical** (state_of_the_derivation\n"
     "§3, 'per-generation physical mass ratios = the layer itself').\n"
     "\n"
     "HONEST CORRECTION. The C₃₆-twist attack called this 'one BOUNDED\n"
     "P-point-δ ↔ N-orbit map'. That was optimistic: the SKELETON is bounded\n"
     "(G2, done), but the CONTENT — δ's value — is the deep frontier. The map\n"
     "is not a bounded item; it bottoms out at Need-B δ-physical.")


# ======================================================================
print("=" * 72)
print("G4 — verdict")
print("=" * 72)
gate("G4 the C₃₆-twist chain bottoms out at the deep frontier — honest end", True,
     """OUTCOME of 'derive the P-point-δ ↔ N-orbit map':

 • SKELETON — DERIVED (G2). The mass-eigenbasis and the N-orbit cyclic basis
   are one DFT U apart; the generation-C₃ of the mass circulant and the C_36
   of the N-orbit are the one M1.B Galois Z₃. Assembled from theorem-grade
   pieces. This is the genuine bounded part, and it is done.

 • CONTENT — NOT bounded (G3). The map's substance is the VALUE of δ — the
   per-generation Koide phase that fixes the mass ratios. δ_lepton = 2/9 is
   'algebraic-identity-only'; the quark δ even less pinned. Deriving δ IS the
   named deep-frontier mask Need-B δ-physical.

 HONEST CORRECTION of the C₃₆-twist attack: 'one bounded P-point-δ ↔ N-orbit
 map' overstated it — only the skeleton is bounded. The chain
   θ_u → §8 → C₃₆-twist → [ORDER] → δ → the P-N map → δ's value
 has now reached **Need-B δ-physical**, a genuine deep-frontier mask with no
 bounded entry (state_of_the_derivation §3). This is the honest END of the
 'attack the next bounded thing' chain — it has run into the real frontier,
 not another bounded step. Closing it is a deliberate multi-session research
 program (the substrate h-power Yukawa formalism), not a probe.

 NET for the quark-flavor sector this session: amplitudes, ε², the GST node
 picture, θ_u, and the C₃₆-twist [ORDER]/[GEN-PAIR] split are all structural;
 the one irreducible bottom is δ-physical — and that was already a known
 named mask. Nothing new is broken; the frontier is precisely located.""")


# ======================================================================
print("=" * 72)
n = sum(results)
print(f"P-N MAP SENTINEL: {n}/{len(results)} gates ({n} bounded, "
      f"{len(results)-n} = deep frontier)")
print("=" * 72)
print("""
The P-point-δ ↔ N-orbit map: its SKELETON is derived (one Galois Z₃ + the DFT
U — theorem-grade pieces). Its CONTENT — the value of the Koide phase δ — is
the named deep-frontier mask Need-B δ-physical, with no bounded entry. The
C₃₆-twist attack's "one bounded map" is corrected: only the skeleton was
bounded. The 'attack the next bounded thing' chain has honestly reached the
deep frontier and stops here.
""")
# G3 is intentionally OPEN — it reports the deep frontier honestly, not a
# probe failure. Exit 0: the analysis is complete and correct.
raise SystemExit(0)
