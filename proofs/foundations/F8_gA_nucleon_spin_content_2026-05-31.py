#!/usr/bin/env python3
# ============================================================
# F8 gate: g_A — the nucleon axial coupling from the 3-walker spin content
# ============================================================
#
# Scope: docs/scoping/fresh_threads_baryon_sector_2026-05-31.md §F8 (the OPEN LEG).
# Predecessor: F8_nucleon_3body_binding_2026-05-31.py built the nucleon as a
# 3-walker entropic bound state (the color-singlet string junction) and supplied
# the Q_np QCD matrix element via flavor-blindness. The remaining open leg was
# g_A = nucleon AXIAL coupling, which needs the nucleon SPIN content (the binding
# gives the mass, not the axial structure). The prior scoping
# (proofs/cosmology/nucleon_sector_BBN_gate_scoping_2026-05-28.py PART C) graded
# g_A "fully open — no framework handle". This probe advances it.
#
# THE STRUCTURE THE FRAMEWORK NOW SUPPLIES (after F8 part 1):
#   - The nucleon = THREE walkers meeting at a common junction (F8 part 1). The
#     junction is the COLOR-SINGLET baryon vertex (3 walks = 3 colors antisymmetric).
#   - Each walker carries spinor-return = a spin-1/2 SU(2) doublet (the ISO program's
#     "chirality operator = walk srs<->srs-z"; theorem_V_Ram_Cl6_Fock_iso).
#   - PAULI: the color part is totally ANTISYMMETRIC (the singlet junction) => the
#     spin (x) flavor part must be totally SYMMETRIC. For 3 spin-1/2 (x) 2 flavors
#     (u,d), the totally symmetric spin-flavor state IS the SU(6) 56-plet. The
#     proton is its I=1/2, S=1/2 member.
#
# So g_A is the SPIN analog of F8 part 2's FLAVOR result:
#   - part 2 (flavor): binding is geometric => flavor-blind => <N|qq|N> = valence
#     count = 1  => Q_np^QCD = m_d - m_u.
#   - here  (spin):   junction is color-singlet => Pauli => SU(6)-symmetric spin
#     state => g_A = the spin-flavor matrix element <p|Sum sigma_z tau_3|p> = 5/3.
# Both are the LEADING-ORDER "free constituent count"; the binding (geometric,
# flavor- AND spin-blind at leading order) does not renormalize them.
#
# This probe (1) builds the explicit SU(6) proton spin-flavor wavefunction that the
# color-singlet junction + Pauli forces, (2) computes g_A = 5/3 as a real matrix
# element (not an assertion), (3) bounds it honestly against the observed 1.2723:
# the reduction 5/3 -> 1.27 (factor ~0.76) is the relativistic/sea bound-state
# effect ("spin crisis"), the SPIN analog of the sub-% sea-quark flavor dependence
# F8 part 2 already flagged as open. The geometric MDL binding is spin-blind at
# leading order, so it does NOT supply the reduction -> that piece stays open.

from fractions import Fraction


# ---------------------------------------------------------------------------
# The SU(6)-symmetric proton (spin-up) wavefunction.
#
# Forced by: color-singlet junction (antisym color) + Pauli => spin(x)flavor
# totally symmetric. The standard normalized |p^>:
#
#  |p^> = (1/sqrt18)[ 2 u^u^d. - u^u.d^ - u.u^d^       (d in slot 3)
#                   + 2 u^d.u^ - u^d^u. - u.d^u^       (d in slot 2)
#                   + 2 d.u^u^ - d^u^u. - d^u.u^ ]     (d in slot 1)
#  where ^ = spin up, . = spin down. (Norm: 3*2^2 + 6*1^2 = 18.)
# ---------------------------------------------------------------------------
def proton_up_wavefunction():
    """Return list of (coeff, ((flavor,spin),(.,.),(.,.))) basis terms.
    flavor: +1 = u, -1 = d.  spin: +1 = up, -1 = down."""
    u_up, u_dn = (+1, +1), (+1, -1)
    d_up, d_dn = (-1, +1), (-1, -1)
    terms = [
        (2, (u_up, u_up, d_dn)), (-1, (u_up, u_dn, d_up)), (-1, (u_dn, u_up, d_up)),
        (2, (u_up, d_dn, u_up)), (-1, (u_up, d_up, u_dn)), (-1, (u_dn, d_up, u_up)),
        (2, (d_dn, u_up, u_up)), (-1, (d_up, u_up, u_dn)), (-1, (d_up, u_dn, u_up)),
    ]
    return terms


def check_normalization(terms):
    norm_sq = sum(c * c for c, _ in terms)          # = 18 before the 1/sqrt18
    return norm_sq


def check_quantum_numbers(terms):
    """Every basis term must have flavor uud (I3 = +1/2) and total Sz = +1/2."""
    ok_flavor = ok_spin = True
    for _, state in terms:
        n_u = sum(1 for f, s in state if f == +1)
        if n_u != 2:
            ok_flavor = False
        sz = Fraction(sum(s for f, s in state), 2)   # each quark Sz = s/2
        if sz != Fraction(1, 2):
            ok_spin = False
    return ok_flavor, ok_spin


def g_A_matrix_element(terms):
    """g_A = <p^| Sum_i sigma_z(i) tau_3(i) |p^>  (isovector axial charge).
    sigma_z = s (+/-1), tau_3 = f (+/-1)."""
    norm_sq = sum(c * c for c, _ in terms)
    acc = 0
    for c, state in terms:
        op = sum(f * s for f, s in state)            # Sum_i sigma_z tau_3 on this term
        acc += c * c * op
    return Fraction(acc, norm_sq)


def main():
    print("=" * 72)
    print("F8 gate: g_A — nucleon axial coupling from the 3-walker spin content")
    print("=" * 72)

    terms = proton_up_wavefunction()

    print("\n[setup] The color-singlet junction (F8 part 1) + spinor-return + Pauli")
    print("        force the SU(6)-symmetric spin(x)flavor proton state.")
    nsq = check_normalization(terms)
    print(f"        wavefunction norm^2 (pre-1/sqrt18) = {nsq}  "
          f"(expected 18: {'OK' if nsq == 18 else 'FAIL'})")
    okf, oks = check_quantum_numbers(terms)
    print(f"        every term has flavor uud (I3=+1/2): {'OK' if okf else 'FAIL'}")
    print(f"        every term has total Sz = +1/2:      {'OK' if oks else 'FAIL'}")

    print("\n[1] g_A leading order = the SU(6) spin-flavor matrix element:")
    gA = g_A_matrix_element(terms)
    print(f"        g_A^(0) = <p^|Sum_i sigma_z(i) tau_3(i)|p^> = {gA} = {float(gA):.4f}")
    # cross-check via the per-quark spin fractions Delta_u, Delta_d
    #   Delta_q = <p^| Sum_i delta(flavor_i,q) sigma_z(i) |p^>  (with sign tau_3 split)
    nsq = sum(c * c for c, _ in terms)
    dU = Fraction(sum(c * c * sum(s for f, s in st if f == +1) for c, st in terms), nsq)
    dD = Fraction(sum(c * c * sum(s for f, s in st if f == -1) for c, st in terms), nsq)
    print(f"        cross-check: Delta_u = {dU} = {float(dU):+.3f},  "
          f"Delta_d = {dD} = {float(dD):+.3f}  (PDG-ish +0.84/-0.43 leading)")
    print(f"                     g_A = Delta_u - Delta_d = {dU - dD} = {float(dU - dD):.4f}")

    g_A_obs = 1.2723
    print("\n[2] honest bound against observation:")
    print(f"        g_A(obs)  = {g_A_obs}")
    print(f"        g_A^(0)   = {float(gA):.4f}  (framework leading-order, SU(6))")
    print(f"        ratio obs/LO = {g_A_obs/float(gA):.4f}  (the 'spin-crisis' reduction)")
    print("        The reduction (5/3 -> 1.27, factor ~0.76) is the relativistic /")
    print("        sea-quark QCD bound-state renormalization. It is the SPIN ANALOG of")
    print("        the sub-% sea-quark FLAVOR dependence F8 part 2 already flagged as")
    print("        open. The geometric MDL binding (F8 part 1) is spin-blind at")
    print("        leading order (it gives mass, not axial overlap), so it does NOT")
    print("        supply the reduction -> that renormalization stays an open leg.")

    print("\n" + "=" * 72)
    print("VERDICT — F8 gate (g_A)")
    print("=" * 72)
    print(f"""  ADVANCE over the prior 'fully open' grade (BBN-gate scoping PART C):
   the framework now HAS a derived nucleon spin content. The F8 color-singlet
   junction + spinor-return + Pauli FORCE the SU(6)-symmetric spin-flavor state,
   whose axial matrix element is a REAL computation:

       g_A^(0) = <p^|Sum sigma_z tau_3|p^> = {gA} = {float(gA):.4f}

   This is the SPIN analog of F8 part 2's flavor-blind valence count
   (<N|qq|N> = 1 -> Q_np = m_d - m_u). Same leading-order 'free-constituent
   count'; same flavor/spin-blind geometric binding leaving it un-renormalized.

  STILL OPEN (honestly, no F7-style oversell):
   - the reduction 5/3 -> 1.2723 (factor {g_A_obs/float(gA):.3f}) = relativistic/sea
     'spin-crisis' renormalization. The framework's geometric binding is
     spin-blind at leading order, so it does NOT derive this factor. This is the
     genuine hard open leg — the same bound-state QCD effect that reduces the
     naive constituent count, now isolated to a single number.

  NET: the g_A gate moves from 'fully open / no handle' to 'leading-order spin
  content DERIVED (5/3), QCD renormalization (->1.27) OPEN'. Symmetric with the
  Q_np leg: framework supplies the leading constituent count for free; the
  sub-leading bound-state renormalization remains the wall.""")
    print("=" * 72)


if __name__ == "__main__":
    main()
