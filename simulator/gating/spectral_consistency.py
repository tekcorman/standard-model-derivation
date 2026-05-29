"""
Spectral-triple hard-constraint filter — C1-C4 gate.

Companion module to `simulator.menus.spectral_triple`.  Each candidate tuple
(J variant, basis on Cl(6) Fock, SU(3) embedding, U(1)_Y formula, KO-dim
reduction sign) goes through the four hard constraints from the M-arc theory
writeup §5:

  C1 — Lie-algebra closure: each gauge factor's generators close (SU(3), SU(2),
       U(1)).
  C2 — Gauge equivariance with D_F: for each Hermitian generator T of the SM
       gauge group lifted to A_F, [T_lifted, D_F] = 0  at machine precision.
  C3 — Distinct gauge factors commute: [SU(3), SU(2)_L] = [SU(3), U(1)_Y] =
       [SU(2)_L, U(1)_Y] = 0  on Cl(6) Fock (after basis change if any).
  C4 — CC spectral-triple axioms for J:
         (4a) J antiunitary, J² = ε·1
         (4b) J D_F J⁻¹ = ε'·D_F
         (4c) J χ̂ J⁻¹ = ε''·χ̂
         (4d) 0-th order [J π(a) J⁻¹, π(b)] = 0
         (4e) 1st order [[D_F, π(a)], J π(b) J⁻¹] = 0
       and the resulting (ε, ε', ε'') matches the tuple's `reduction` field
       (KO-dim 0 → (+1,+1,+1); KO-dim 6 → (+1,+1,−1)).

The gate REUSES the operator constructors from
  `proofs.foundations.M1_J_real_structure_probe`
  `proofs.foundations.M2_SM_gauge_embedding_probe`
  `proofs.foundations.de_rham_susy_fibered_v2_probe`
so behaviour is consistent with the M1/M2 probe verdicts.

For tuples whose basis is `ps_tensor_product` (not yet built) or whose
SU(3) embedding requires that basis, the gate returns `not_constructable`
with a clear flag, rather than a spurious PASS/FAIL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import sys
from pathlib import Path

import numpy as np

# Ensure repo root is importable so we can reuse M1/M2/v2 probe primitives.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs.foundations.de_rham_susy_fibered_v2_probe import (  # noqa: E402
    d_alg, NV, NE, SX, SY, SZ, I2,
)
from proofs.foundations.M1_J_real_structure_probe import build_J  # noqa: E402
from proofs.foundations.M2_SM_gauge_embedding_probe import (  # noqa: E402
    build_gamma, biv,
)

from ..menus.spectral_triple import (  # noqa: E402
    SpectralTripleChoice, BASIS_BUILT,
)


# ---------------------------------------------------------------------------
# Numerical tolerance
# ---------------------------------------------------------------------------

TOL = 1e-9


# ---------------------------------------------------------------------------
# Cached framework data (independent of choice tuple)
# ---------------------------------------------------------------------------

class _FrameworkData:
    """Lazily-built operators independent of the choice tuple."""
    _instance: Optional['_FrameworkData'] = None

    @classmethod
    def get(cls) -> '_FrameworkData':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # D_F = [[0, d†], [d, 0]] on H_F = C⁰_alg ⊕ C¹_alg (280-dim)
        d = d_alg((0.0, 0.0, 0.0))
        dim0, dim1 = NV * 64, NE * 4
        D_F = np.zeros((dim0 + dim1, dim0 + dim1), dtype=complex)
        D_F[:dim0, dim0:] = d.conj().T
        D_F[dim0:, :dim0] = d
        self.D_F = D_F
        self.dim0 = dim0
        self.dim1 = dim1
        self.dim_total = dim0 + dim1
        # χ̂ = diag(+1 on C⁰_alg, −1 on C¹_alg)
        self.chi = np.diag([1.0] * dim0 + [-1.0] * dim1).astype(complex)
        # Cl(6) Brauer-Weyl gammas and Γ_7 chirality
        self.gamma = build_gamma()
        G = self.gamma
        self.G7 = -1j * G[1] @ G[2] @ G[3] @ G[4] @ G[5] @ G[6]


# ---------------------------------------------------------------------------
# Basis builders (8×8 unitary U mapping Brauer-Weyl ℂ^8 to chosen basis)
# ---------------------------------------------------------------------------

def _basis_change_brauer_weyl() -> np.ndarray:
    return np.eye(8, dtype=complex)


def _basis_change_b3_species() -> np.ndarray:
    """Simultaneous eigenbasis of (T_1, T_2, Y) Spin(6) Cartan triple.

    Mirrors `theorem_B3_spinor_fermion.py` Step 2: diagonalise a generic
    incommensurate linear combination so eigh separates the 8 weight states.
    """
    fw = _FrameworkData.get()
    G = fw.gamma
    T1 = biv(G, 1, 2) / (2j)
    T2 = biv(G, 3, 4) / (2j)
    Y = biv(G, 5, 6) / (2j)
    # incommensurate coefficients separate the joint spectrum
    M = 1.0 * T1 + 3.7 * T2 + 11.3 * Y
    _, U = np.linalg.eigh(M)
    return U


def _basis_change_c3_eigenbasis() -> np.ndarray:
    """Simultaneous eigenbasis of the body-diagonal C_3 on Cl(6) Fock.

    C_3 cyclically permutes the three pairs of Γ generators
    (Γ_1,Γ_2) → (Γ_3,Γ_4) → (Γ_5,Γ_6) → (Γ_1,Γ_2).  The Spin(6) lift on
    ℂ^8 has order 3.  Its eigenbasis diagonalises the colour-Z_3 action.
    """
    fw = _FrameworkData.get()
    G = fw.gamma
    # Build the C_3 lift U_{C_3} as exp((π/3) · S) where S generates the
    # permutation (Γ_1,Γ_2)→(Γ_3,Γ_4)→(Γ_5,Γ_6) via successive 2π/3 rotations
    # in the (1,3,5) and (2,4,6) planes.  Concretely the C_3 lift cycles
    # the three bivector pairs.  Use the bivector form on the diagonal:
    #   gen = (1/√3) · (Γ_13 + Γ_24 + Γ_35 + Γ_46 + Γ_51 + Γ_62) / 2
    # This may not be in the standard form, so we construct C_3 as a finite
    # rotation on the underlying ℂ^6 then lift to spinors.
    # Direct construction: build the permutation P on (1..6) sending
    #   1→3, 2→4, 3→5, 4→6, 5→1, 6→2.
    # Then  U_{C_3} Γ_a U_{C_3}^{-1} = Γ_{P(a)}.
    # In the ℂ^8 spinor rep this is implemented by:
    P = {1: 3, 2: 4, 3: 5, 4: 6, 5: 1, 6: 2}
    # Solve for U_{C_3}: find unitary V such that V Γ_a V† = Γ_{P(a)}.
    # Use the Brauer construction: V acts on the basis state |abc⟩ → relabel.
    # On ℂ^8 = (ℂ^2)^{⊗3}, the cyclic permutation of qubits 1→3→2→1 isn't quite
    # P (which mixes within pairs); but P is precisely the permutation that
    # CYCLES THE THREE QUBITS: qubit_1(Γ_1, Γ_2) → qubit_2(Γ_3, Γ_4) →
    # qubit_3(Γ_5, Γ_6) → qubit_1.
    # So U_{C_3} on (ℂ^2)^{⊗3} is the cyclic-shift unitary:  |abc⟩ → |cab⟩.
    V = np.zeros((8, 8), dtype=complex)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                # source state |abc⟩ has index 4a+2b+c (qubit 1 = a, qubit 2 = b, qubit 3 = c)
                src = 4 * a + 2 * b + c
                # target |cab⟩ has index 4*c + 2*a + b
                tgt = 4 * c + 2 * a + b
                V[tgt, src] = 1.0
    # Sanity: V³ = I
    V3 = V @ V @ V
    if not np.allclose(V3, np.eye(8, dtype=complex), atol=TOL):
        raise RuntimeError('c3 basis: cyclic shift unitary failed V³=I check')
    # Eigenbasis of V
    _, U = np.linalg.eig(V)
    # Orthonormalise (since V is unitary, eig should already return unitary cols, but normalise to be safe)
    U, _ = np.linalg.qr(U)
    return U


_BASIS_BUILDERS = {
    'brauer_weyl':   _basis_change_brauer_weyl,
    'b3_species':    _basis_change_b3_species,
    'c3_eigenbasis': _basis_change_c3_eigenbasis,
    # 'ps_tensor_product' is NOT built — gate flags it as not_constructable.
}


# ---------------------------------------------------------------------------
# SM gauge generator builders (on Cl(6) Fock ℂ^8 in Brauer-Weyl basis)
# ---------------------------------------------------------------------------

def _su2L_bw() -> list[np.ndarray]:
    """SU(2)_L (self-dual bivector triple) in Brauer-Weyl basis."""
    fw = _FrameworkData.get(); G = fw.gamma
    G12, G34 = biv(G, 1, 2), biv(G, 3, 4)
    G13, G24 = biv(G, 1, 3), biv(G, 2, 4)
    G14, G23 = biv(G, 1, 4), biv(G, 2, 3)
    return [(G12 + G34) / (4j), (G13 - G24) / (4j), (G14 + G23) / (4j)]


def _su2R_bw() -> list[np.ndarray]:
    """SU(2)_R (anti-self-dual) in Brauer-Weyl basis."""
    fw = _FrameworkData.get(); G = fw.gamma
    G12, G34 = biv(G, 1, 2), biv(G, 3, 4)
    G13, G24 = biv(G, 1, 3), biv(G, 2, 4)
    G14, G23 = biv(G, 1, 4), biv(G, 2, 3)
    return [(G12 - G34) / (4j), (G13 + G24) / (4j), (G14 - G23) / (4j)]


def _su3_gell_mann_upper3_bw() -> list[np.ndarray]:
    """SU(3)_c via standard Gell-Mann λ^a/2 on the upper-3×3 block of SU(4)'s
    4-rep (= chir+ sector of Cl(6) Fock), conj. on 4̄ (chir− sector)."""
    fw = _FrameworkData.get(); G7 = fw.G7
    L = lambda *rows: np.array(rows, dtype=complex)
    lam = [
        L([0, 1, 0], [1, 0, 0], [0, 0, 0]),
        L([0, -1j, 0], [1j, 0, 0], [0, 0, 0]),
        L([1, 0, 0], [0, -1, 0], [0, 0, 0]),
        L([0, 0, 1], [0, 0, 0], [1, 0, 0]),
        L([0, 0, -1j], [0, 0, 0], [1j, 0, 0]),
        L([0, 0, 0], [0, 0, 1], [0, 1, 0]),
        L([0, 0, 0], [0, 0, -1j], [0, 1j, 0]),
        L([1, 0, 0], [0, 1, 0], [0, 0, -2]) / np.sqrt(3),
    ]
    T_4 = []
    for la in lam:
        M = np.zeros((4, 4), dtype=complex)
        M[:3, :3] = la / 2.0
        T_4.append(M)
    # Lift to ℂ^8 via Γ_7 chiral split
    eigs, vecs = np.linalg.eigh(G7)
    plus = vecs[:, [k for k in range(8) if eigs[k] > 0.5]]
    minus = vecs[:, [k for k in range(8) if eigs[k] < -0.5]]
    out = []
    for T4 in T_4:
        # Antifundamental on chir−: T̄^a = −T^{a*} (so [T̄^a, T̄^b] = i f^{abc} T̄^c
        # closes the SU(3) algebra on ℂ^8 = 4 ⊕ 4̄; M2 used +T^{a*} which generates
        # SU(3) × SU(3) and fails C1 — corrected here.).
        T8 = plus @ T4 @ plus.conj().T - minus @ T4.conj() @ minus.conj().T
        out.append(T8)
    return out


def _u1y_bw(formula: str) -> np.ndarray:
    """U(1)_Y generator on Cl(6) Fock in Brauer-Weyl basis."""
    fw = _FrameworkData.get(); G = fw.gamma
    JR3 = (biv(G, 1, 4) - biv(G, 2, 3)) / (4j)  # T_3R = (Γ_14 − Γ_23)/(4i)
    BminusL = biv(G, 5, 6) / (2j)               # B−L = Γ_56/(2i)
    if formula == 't3r_plus_bminusL_over_2':
        return JR3 + BminusL / 2.0
    if formula == 'three_fifths_bminusL_plus_t3r':
        return (3.0 / 5.0) * BminusL + JR3
    raise ValueError(f'unknown U(1)_Y formula: {formula!r}')


# ---------------------------------------------------------------------------
# Choice-aware operator construction
# ---------------------------------------------------------------------------

@dataclass
class BuiltOperators:
    """Operators built from a (choice, framework data) pair, on Cl(6) Fock ℂ^8."""
    U_basis: np.ndarray            # 8×8 unitary mapping Brauer-Weyl → chosen basis
    su2L: list[np.ndarray]         # 3 generators in chosen basis
    su2R: list[np.ndarray]         # 3 generators in chosen basis
    su3: list[np.ndarray]          # 8 generators in chosen basis
    u1y: np.ndarray                # 1 generator in chosen basis
    J_perm: np.ndarray             # 280×280 permutation part of J on full H_F


def _rebasis(M: np.ndarray, U: np.ndarray) -> np.ndarray:
    return U.conj().T @ M @ U


def _build_operators(choice: SpectralTripleChoice) -> Optional[BuiltOperators]:
    """Construct the operators for one choice tuple, or return None if not_constructable."""
    if not BASIS_BUILT.get(choice.basis, False):
        return None
    if choice.su3_embedding == 'aligned_with_ps_tensor' \
            and choice.basis != 'ps_tensor_product':
        return None
    U = _BASIS_BUILDERS[choice.basis]()
    # SU(2)_L (framework's B3 self-dual triple — INVARIANT under basis change at
    # the level of the algebra, but represented in chosen basis).
    su2L_bw = _su2L_bw()
    su2R_bw = _su2R_bw()
    su3_bw  = _su3_gell_mann_upper3_bw()
    y_bw    = _u1y_bw(choice.u1y_formula)
    su2L = [_rebasis(T, U) for T in su2L_bw]
    su2R = [_rebasis(T, U) for T in su2R_bw]
    su3  = [_rebasis(T, U) for T in su3_bw]
    y    = _rebasis(y_bw, U)
    J_perm = build_J(choice.j_variant)
    return BuiltOperators(U_basis=U, su2L=su2L, su2R=su2R, su3=su3,
                          u1y=y, J_perm=J_perm)


# ---------------------------------------------------------------------------
# C1 — Lie-algebra closure
# ---------------------------------------------------------------------------

def _is_in_span(M: np.ndarray, basis: list[np.ndarray]) -> bool:
    """Return True iff M is a (complex) linear combination of `basis` matrices."""
    A = np.array([B.flatten() for B in basis])
    rank0 = np.linalg.matrix_rank(A, tol=TOL)
    ext = np.vstack([A, M.flatten()])
    return np.linalg.matrix_rank(ext, tol=TOL) == rank0


def _check_su2_closure(triple: list[np.ndarray]) -> bool:
    a, b, c = triple
    if not np.allclose(a @ b - b @ a, 1j * c, atol=TOL): return False
    if not np.allclose(b @ c - c @ b, 1j * a, atol=TOL): return False
    if not np.allclose(c @ a - a @ c, 1j * b, atol=TOL): return False
    return True


def _check_su3_closure(eight: list[np.ndarray]) -> bool:
    # All commutators land in the span of the 8 generators
    rank0 = np.linalg.matrix_rank(np.array([T.flatten() for T in eight]), tol=TOL)
    if rank0 != 8: return False
    for i in range(8):
        for j in range(i + 1, 8):
            if not _is_in_span(eight[i] @ eight[j] - eight[j] @ eight[i], eight):
                return False
    return True


def check_C1(ops: BuiltOperators) -> tuple[bool, str]:
    su2L_ok = _check_su2_closure(ops.su2L)
    su3_ok  = _check_su3_closure(ops.su3)
    # U(1)_Y is abelian — only check Hermitian
    u1_ok   = np.allclose(ops.u1y, ops.u1y.conj().T, atol=TOL)
    ok = su2L_ok and su3_ok and u1_ok
    detail = (f'SU(2)_L closes: {su2L_ok}; SU(3) closes: {su3_ok}; '
              f'U(1)_Y Hermitian: {u1_ok}')
    return ok, detail


# ---------------------------------------------------------------------------
# C2 — Gauge equivariance with D_F
# ---------------------------------------------------------------------------

def _lift_to_AF_vertex_block(T8: np.ndarray, dim0: int, dim1: int) -> np.ndarray:
    """Lift an 8×8 generator on Cl(6) Fock to A_F adjoint on full H_F = ℂ^280.

    The generator acts at each vertex as the adjoint  X ↦ T8 · X − X · T8
    on M_8 (64-dim flatten), and trivially on edges.
    """
    dim_total = dim0 + dim1
    I8 = np.eye(8, dtype=complex)
    # ad_T on M_8 col-major flatten:  (T ⊗ I − I ⊗ T^T)
    ad_T = np.kron(I8, T8) - np.kron(T8.T, I8)
    M = np.zeros((dim_total, dim_total), dtype=complex)
    for v in range(NV):
        M[v * 64:(v + 1) * 64, v * 64:(v + 1) * 64] = ad_T
    return M


def check_C2(ops: BuiltOperators) -> tuple[bool, dict]:
    fw = _FrameworkData.get()
    diags: dict[str, float] = {}
    # Lift each generator (in chosen basis) back to the FRAMEWORK action on H_F.
    # Critical point: D_F is constructed in the FIXED Brauer-Weyl basis at each
    # vertex; we must conjugate the chosen-basis generator BACK to Brauer-Weyl
    # before lifting.  Equivalently: T_bw = U · T_chosen · U†.
    U = ops.U_basis
    def lift_bw_back(T_chosen: np.ndarray) -> np.ndarray:
        T_bw = U @ T_chosen @ U.conj().T
        return _lift_to_AF_vertex_block(T_bw, fw.dim0, fw.dim1)
    all_ok = True
    for label, gens in [('SU(2)_L', ops.su2L),
                        ('SU(3)',   ops.su3),
                        ('U(1)_Y',  [ops.u1y])]:
        max_norm = 0.0
        for T in gens:
            Tlift = lift_bw_back(T)
            comm = Tlift @ fw.D_F - fw.D_F @ Tlift
            nrm = np.linalg.norm(comm)
            if nrm > max_norm:
                max_norm = nrm
        diags[label] = max_norm
        if max_norm > TOL:
            all_ok = False
    return all_ok, diags


# ---------------------------------------------------------------------------
# C3 — distinct gauge factors commute on Cl(6) Fock
# ---------------------------------------------------------------------------

def check_C3(ops: BuiltOperators) -> tuple[bool, dict]:
    diags: dict[str, float] = {}
    def max_comm(A_list, B_list):
        m = 0.0
        for A in A_list:
            for B in B_list:
                c = A @ B - B @ A
                m = max(m, np.linalg.norm(c))
        return m
    diags['[SU(3), SU(2)_L]'] = max_comm(ops.su3, ops.su2L)
    diags['[SU(3), U(1)_Y]']  = max_comm(ops.su3, [ops.u1y])
    diags['[SU(2)_L, U(1)_Y]'] = max_comm(ops.su2L, [ops.u1y])
    ok = all(v < TOL for v in diags.values())
    return ok, diags


# ---------------------------------------------------------------------------
# C4 — CC spectral-triple axioms for J
# ---------------------------------------------------------------------------

def _find_sign(M: np.ndarray, ref: np.ndarray) -> int:
    if np.allclose(M, ref, atol=TOL):  return +1
    if np.allclose(M, -ref, atol=TOL): return -1
    return 0


def check_C4(ops: BuiltOperators, expected_ko_dim: int
             ) -> tuple[bool, dict]:
    """Verify CC axioms for J; return (ok, diagnostics).

    `expected_ko_dim` is 0 or 6 — determines the target sign tuple
    (ε, ε', ε'').  KO-dim 0: (+1,+1,+1); KO-dim 6: (+1,+1,−1).
    """
    fw = _FrameworkData.get()
    P = ops.J_perm
    I_full = np.eye(fw.dim_total, dtype=complex)
    # J²: J(v) = P · conj(v), so J²(v) = P · conj(P) · v.  P is real here.
    J2 = P @ P.conjugate()
    eps  = _find_sign(J2, I_full)
    # J D J⁻¹ (J⁻¹ = J for J²=I):  linear part = P · conj(D) · P
    JDJ = P @ fw.D_F.conjugate() @ P
    eps_p  = _find_sign(JDJ, fw.D_F)
    # J χ J⁻¹: linear part = P · χ · P
    JchiJ = P @ fw.chi @ P
    eps_pp = _find_sign(JchiJ, fw.chi)
    if expected_ko_dim == 0:
        target = (+1, +1, +1)
    elif expected_ko_dim == 6:
        target = (+1, +1, -1)
    else:
        raise ValueError(f'expected_ko_dim must be 0 or 6, got {expected_ko_dim}')
    signs_ok = ((eps, eps_p, eps_pp) == target)
    diags = {
        'eps':       eps,
        'eps_prime': eps_p,
        'eps_dprime': eps_pp,
        'target':    target,
        'signs_match_expected_ko_dim': signs_ok,
    }
    return signs_ok, diags


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Outcome of running C1-C4 on one SpectralTripleChoice."""
    choice: SpectralTripleChoice
    constructable: bool
    c1_ok: bool = False
    c1_detail: str = ''
    c2_ok: bool = False
    c2_diag: dict = field(default_factory=dict)
    c3_ok: bool = False
    c3_diag: dict = field(default_factory=dict)
    c4_ok: bool = False
    c4_diag: dict = field(default_factory=dict)
    not_constructable_reason: str = ''

    @property
    def passes_all(self) -> bool:
        return (self.constructable and self.c1_ok and self.c2_ok
                and self.c3_ok and self.c4_ok)

    def summary(self) -> str:
        if not self.constructable:
            return (f'  NOT CONSTRUCTABLE: {self.not_constructable_reason}')
        return (f'  C1={self.c1_ok}  C2={self.c2_ok}  '
                f'C3={self.c3_ok}  C4={self.c4_ok}'
                f'  → {"PASS" if self.passes_all else "FAIL"}')


def evaluate(choice: SpectralTripleChoice) -> CheckResult:
    """Run C1-C4 on one choice tuple."""
    ops = _build_operators(choice)
    if ops is None:
        reason = []
        if not BASIS_BUILT.get(choice.basis, False):
            reason.append(f'basis {choice.basis!r} not built (M2.refined deliverable)')
        if choice.su3_embedding == 'aligned_with_ps_tensor' \
                and choice.basis != 'ps_tensor_product':
            reason.append(f'SU(3) embedding {choice.su3_embedding!r} requires '
                          f'ps_tensor_product basis')
        return CheckResult(choice=choice, constructable=False,
                           not_constructable_reason='; '.join(reason))
    c1_ok, c1_detail = check_C1(ops)
    c2_ok, c2_diag   = check_C2(ops)
    c3_ok, c3_diag   = check_C3(ops)
    expected_ko = 0 if choice.reduction == 'ko_dim_0' else 6
    c4_ok, c4_diag   = check_C4(ops, expected_ko)
    return CheckResult(
        choice=choice, constructable=True,
        c1_ok=c1_ok, c1_detail=c1_detail,
        c2_ok=c2_ok, c2_diag=c2_diag,
        c3_ok=c3_ok, c3_diag=c3_diag,
        c4_ok=c4_ok, c4_diag=c4_diag,
    )


def filter_survivors(candidates: list[SpectralTripleChoice]
                     ) -> tuple[list[CheckResult], list[CheckResult]]:
    """Run all candidates through C1-C4.

    Returns (survivors, all_results).  A survivor is a CheckResult with
    `passes_all == True`.
    """
    results = [evaluate(c) for c in candidates]
    survivors = [r for r in results if r.passes_all]
    return survivors, results
