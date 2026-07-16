#!/usr/bin/env python3
"""
proofs/foundations/M0_convention_control_2026-07-07.py

M0 — THE CONVENTION CONTROL (the 2-mode toy). Pre-registered in
internal research notes (committed 3ba1633
BEFORE this file). Station M0-C.

PURPOSE (charting our own course): we do NOT import the Peschel/Araki entanglement-
Hamiltonian formula from conventional physics. We DERIVE it here for OUR quasi-free
states by EXACT DIAGONALIZATION of an explicit many-body density matrix, and we RECORD
the exact sign/branch convention that makes it hold. Everything below is exact linear
algebra on a 4-mode fermionic Fock space (16-dim), region A = the first 2 modes.

The four controls (all must LOCK or the station VOIDs):
  C-GAUSSIAN : the exact reduced state rho_A is the exponential of a QUADRATIC form
               (Wick's theorem for our quasi-free state) -- verified by reconstruction.
  C-PESCHEL  : h_A = log((1 - C_A)/C_A) reproduces the exact rho_A eigenvalues (<1e-9).
               Records the OWNED convention: rho_A = e^{-K}/Z, K = sum h_ij a_i^dag a_j.
  C-FIRSTLAW : d<K> = dS to first order under state perturbation (ratio -> 1, resid O(d^2)).
  C-KMS      : rho_A is KMS at beta=1 wrt its OWN modular flow sigma_t = e^{iKt}. e^{-iKt}
               (the beta=1 tautology -- 'temperature' = the scale relating K-units to
               physical energy, fixed downstream by FLOW-ID, NOT an adopted thermometer).

NO substrate object, NO J, NO F1 region, NO target parameter appears here. This is the
pure math control. Poisons (never invoked): 2a1^5, 2a1^3, 5/12, 0.197, 2pi/ln2.
"""
import numpy as np
from scipy.linalg import expm, logm

np.set_printoptions(precision=6, suppress=True)

ok_all = True
def check(name, cond, detail=""):
    global ok_all
    status = "PASS" if cond else "FAIL"
    if not cond:
        ok_all = False
    print(f"  [{status}] {name}{('  -- ' + detail) if detail else ''}")
    return cond

# ---------------------------------------------------------------------------
# 0. Jordan-Wigner fermion operators on N modes (exact many-body, 2^N-dim).
#    Convention (OWNED, stated explicitly):
#      basis per mode: |0> = (1,0) EMPTY, |1> = (0,1) OCCUPIED
#      a  = [[0,1],[0,0]]  (a|1> = |0>, a|0> = 0)
#      a^dag = [[0,0],[1,0]],  n = a^dag a = diag(0,1)
#      a_p = ( Z_0 (x) ... (x) Z_{p-1} ) (x) a (x) I ... (x) I   (JW string of Z=diag(1,-1))
#    Mode ordering: leftmost tensor factor = mode 0. Region A = modes {0,1} is the
#    INITIAL SEGMENT, so the naive tensor partial-trace over B = {2,3} gives the
#    correct FERMIONIC reduced state (no JW-string ambiguity for an initial block).
# ---------------------------------------------------------------------------
I2 = np.eye(2)
Z2 = np.diag([1.0, -1.0])
a1 = np.array([[0.0, 1.0], [0.0, 0.0]])   # single-mode annihilation

def kron_list(mats):
    out = np.array([[1.0]])
    for m in mats:
        out = np.kron(out, m)
    return out

def annihilation(p, N):
    mats = [Z2] * p + [a1] + [I2] * (N - 1 - p)
    return kron_list(mats)

N = 4
A_modes = [0, 1]          # region A (initial segment)
B_modes = [2, 3]
dimA = 2 ** len(A_modes)   # 4
dimB = 2 ** len(B_modes)   # 4

a = [annihilation(p, N) for p in range(N)]
adag = [op.conj().T for op in a]

# CAR sanity: {a_i, a_j^dag} = delta_ij, {a_i, a_j} = 0
car_ok = True
for i in range(N):
    for j in range(N):
        anti = a[i] @ adag[j] + adag[j] @ a[i]
        expected = np.eye(2 ** N) if i == j else np.zeros((2 ** N, 2 ** N))
        if not np.allclose(anti, expected, atol=1e-12):
            car_ok = False
        anti2 = a[i] @ a[j] + a[j] @ a[i]
        if not np.allclose(anti2, np.zeros((2 ** N, 2 ** N)), atol=1e-12):
            car_ok = False
print("== M0-C: convention control (2-mode region, 4-mode toy) ==")
check("CAR algebra {a_i,a_j^dag}=delta, {a_i,a_j}=0", car_ok)

# ---------------------------------------------------------------------------
# 1. A pure QUASI-FREE state: a Slater determinant of 2 filled single-particle
#    orbitals, chosen (deterministically) to ENTANGLE region A with region B.
#    b_alpha^dag = sum_i U[i,alpha] a_i^dag ; |psi> = b_0^dag b_1^dag |vac>.
#    U's columns = an orthonormal set of single-particle orbitals from a fixed
#    Hermitian generator (fully deterministic, no RNG).
# ---------------------------------------------------------------------------
def build_state(theta):
    # Fixed Hermitian single-particle generator G; U = exp(i*theta*G) rotates the
    # orbitals so region A and B mix (theta controls the entanglement / perturbation).
    G = np.array([
        [0.0, 0.7, 0.3, 0.1],
        [0.7, 0.0, 0.5, 0.2],
        [0.3, 0.5, 0.0, 0.6],
        [0.1, 0.2, 0.6, 0.0],
    ])
    U = expm(1j * theta * G)
    filled = U[:, [0, 1]]          # fill orbitals 0 and 1
    vac = np.zeros(2 ** N, dtype=complex); vac[0] = 1.0
    # b_alpha^dag applied to vacuum
    psi = vac
    for alpha in range(filled.shape[1]):
        bdag = sum(filled[i, alpha] * adag[i] for i in range(N))
        psi = bdag @ psi
    psi = psi / np.linalg.norm(psi)
    return psi

def corr_matrix(psi):
    # C_ij = <psi| a_i^dag a_j |psi>   (OWNED index convention)
    C = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            C[i, j] = psi.conj() @ (adag[i] @ (a[j] @ psi))
    return C

def reduced_rhoA(psi):
    rho = np.outer(psi, psi.conj())                     # 16x16 pure
    rho = rho.reshape(dimA, dimB, dimA, dimB)           # [A,B,A',B']
    rhoA = np.trace(rho, axis1=1, axis2=3)              # trace over B
    return rhoA                                          # 4x4

theta0 = 0.6
psi0 = build_state(theta0)
C0 = corr_matrix(psi0)
rhoA0 = reduced_rhoA(psi0)

check("rho_A Hermitian, unit trace", np.allclose(rhoA0, rhoA0.conj().T, atol=1e-12)
      and abs(np.trace(rhoA0) - 1) < 1e-12)

C_A = C0[np.ix_(A_modes, A_modes)]                       # restricted 2x2 correlation
zeta = np.linalg.eigvalsh(C_A)                           # occupation eigenvalues in (0,1)
check("C_A eigenvalues in (0,1) (region genuinely mixed)",
      np.all(zeta > 1e-9) and np.all(zeta < 1 - 1e-9),
      detail=f"zeta = {np.sort(zeta)}")

# ---------------------------------------------------------------------------
# 2. Region-A fermion operators (2 modes -> 4-dim), for building K and e^{-K}.
# ---------------------------------------------------------------------------
def annihilation_A(p, nA):
    mats = [Z2] * p + [a1] + [I2] * (nA - 1 - p)
    return kron_list(mats)

nA = len(A_modes)
aA = [annihilation_A(p, nA) for p in range(nA)]
aAdag = [op.conj().T for op in aA]

def quadratic_op(hmat):
    # K = sum_{ij} (h^T)_ij a_i^dag a_j   on region-A Fock space (4-dim).
    # OWNED CONVENTION (derived, not imported): with C_ij = <a_i^dag a_j> a COMPLEX
    # Hermitian matrix, the many-body K that reproduces the state uses the TRANSPOSE of
    # the single-particle h = log((1-C)/C). Reason: diagonalizing C = sum_k zeta_k|u_k><u_k|,
    # the eigen-MODES of K are the CONJUGATES |u_k*>, so K = (sum_k eps_k|u_k><u_k|)^* = h^T
    # (h Hermitian => h^* = h^T). For real symmetric C this is invisible; for complex C it
    # is essential. The 2-mode control exists precisely to pin this sign/transpose down.
    M = hmat.T
    K = np.zeros((dimA, dimA), dtype=complex)
    for i in range(nA):
        for j in range(nA):
            K += M[i, j] * (aAdag[i] @ aA[j])
    return K

# ---------------------------------------------------------------------------
# C-PESCHEL: h_A = log((1 - C_A)/C_A); rho_pred = e^{-K}/Z must equal exact rho_A.
#   (OWNED convention recorded: minus sign in the exponent, h = log((1-C)/C).)
# ---------------------------------------------------------------------------
IA = np.eye(nA)
h_A = logm((IA - C_A) @ np.linalg.inv(C_A))
K = quadratic_op(h_A)
rho_pred = expm(-K)
rho_pred = rho_pred / np.trace(rho_pred)

# eigenvalue-set comparison (the invariant, basis-free statement)
ev_exact = np.sort(np.linalg.eigvalsh(rhoA0).real)
ev_pred = np.sort(np.linalg.eigvalsh(rho_pred).real)
check("C-PESCHEL: rho_A eigenvalues = e^{-K}/Z eigenvalues (<1e-9)",
      np.allclose(ev_exact, ev_pred, atol=1e-9),
      detail=f"max|dev| = {np.max(np.abs(ev_exact - ev_pred)):.2e}")
# and the full operator (both are simultaneously diagonalizable here; compare directly)
check("C-PESCHEL(operator): rho_A == e^{-K}/Z as matrices (<1e-9)",
      np.allclose(rhoA0, rho_pred, atol=1e-9),
      detail=f"max|dev| = {np.max(np.abs(rhoA0 - rho_pred)):.2e}")

# ---------------------------------------------------------------------------
# C-GAUSSIAN: independent cross-check that rho_A is exp of a QUADRATIC form.
#   From the exact rho_A directly: K_exact = -log(rho_A). It must be quadratic,
#   i.e. equal quadratic_op(h') for h' = log((1-C_A)/C_A). We already built that;
#   here we confirm -log(rho_A) has NO constant/quartic pieces beyond K + const.
# ---------------------------------------------------------------------------
K_exact = -logm(rhoA0)
# Remove the scalar (log Z) piece by matching traces against K.
const = (np.trace(K_exact) - np.trace(K)) / dimA
check("C-GAUSSIAN: -log(rho_A) = K + (log Z) I  (rho_A is Gaussian/quadratic)",
      np.allclose(K_exact, K + const * np.eye(dimA), atol=1e-9),
      detail=f"max|dev| = {np.max(np.abs(K_exact - (K + const*np.eye(dimA)))):.2e}")

# entropies two ways
S_direct = -np.sum(ev_exact * np.log(ev_exact))
S_peschel = -np.sum(zeta * np.log(zeta) + (1 - zeta) * np.log(1 - zeta))
check("entropy: -Tr rho_A log rho_A == -sum[z ln z + (1-z)ln(1-z)] (<1e-9)",
      abs(S_direct - S_peschel) < 1e-9,
      detail=f"S = {S_direct:.9f} (nats)")

# ---------------------------------------------------------------------------
# C-FIRSTLAW: d<K> = dS to first order in a state perturbation.
#   K is FIXED (from theta0). Perturb the state theta0 -> theta0 + delta.
#   Delta<K> = Tr[(rho_A(delta) - rho_A(0)) K];  DeltaS = S(delta) - S(0).
#   First law => Delta<K> - DeltaS = O(delta^2) (relative entropy, >=0).
# ---------------------------------------------------------------------------
def K_expect_and_S(theta, Kfixed):
    psi = build_state(theta)
    rA = reduced_rhoA(psi)
    Kexp = np.trace(rA @ Kfixed).real
    ev = np.clip(np.linalg.eigvalsh(rA).real, 1e-15, None)
    S = -np.sum(ev * np.log(ev))
    return Kexp, S

K0_exp, S0 = K_expect_and_S(theta0, K)
ratios = []
print("  C-FIRSTLAW: delta -> (Delta<K> - DeltaS) should vanish ~ delta^2")
for delta in [1e-1, 1e-2, 1e-3]:
    Kd, Sd = K_expect_and_S(theta0 + delta, K)
    dK = Kd - K0_exp
    dS = Sd - S0
    gap = dK - dS                      # = relative entropy S(rho_d || rho_0) >= 0, O(delta^2)
    ratios.append(gap / delta ** 2)
    print(f"      delta={delta:.0e}:  Delta<K>={dK:+.6e}  DeltaS={dS:+.6e}  "
          f"gap={gap:.3e}  gap/delta^2={gap/delta**2:.4f}")
# first-order agreement: gap/delta shrinks linearly (=> gap is O(delta^2))
gap_over_delta = []
for delta in [1e-2, 1e-3, 1e-4]:
    Kd, Sd = K_expect_and_S(theta0 + delta, K)
    gap_over_delta.append((Kd - K0_exp - (Sd - S0)) / delta)
check("C-FIRSTLAW: (Delta<K>-DeltaS)/delta -> 0 (first law holds to first order)",
      abs(gap_over_delta[-1]) < 1e-3 and abs(gap_over_delta[-1]) < abs(gap_over_delta[0]),
      detail=f"gap/delta at 1e-2,1e-3,1e-4 = {[f'{g:.2e}' for g in gap_over_delta]}")
check("C-FIRSTLAW: relative entropy >= 0 (gap non-negative, all deltas)",
      all(r >= -1e-9 for r in ratios))

# ---------------------------------------------------------------------------
# C-KMS: rho_A is KMS at beta=1 wrt its OWN modular flow sigma_t(x)=e^{iKt} x e^{-iKt}.
#   Two-point KMS: <a_i sigma_{t=i*beta}(a_j^dag)> = <a_j^dag a_i>  at beta=1.
#   sigma_{i}(a_j^dag) = e^{-K} a_j^dag e^{K}  (analytic continuation t -> i).
# ---------------------------------------------------------------------------
rhoA_state = rho_pred                                   # = e^{-K}/Z
eK = expm(K); emK = expm(-K)
kms_ok = True
for i in range(nA):
    for j in range(nA):
        lhs = np.trace(rhoA_state @ (aA[i] @ (emK @ aAdag[j] @ eK))).real  # <a_i sigma_i(a_j^dag)>
        rhs = np.trace(rhoA_state @ (aAdag[j] @ aA[i])).real               # <a_j^dag a_i>
        if abs(lhs - rhs) > 1e-9:
            kms_ok = False
check("C-KMS: two-point KMS relation holds at beta=1 (state's OWN modular point)",
      kms_ok)

# ---------------------------------------------------------------------------
print()
print("== OWNED CONVENTION (recorded for the substrate probe) ==")
print("  state:        quasi-free (Gaussian) vacuum; C_ij = <a_i^dag a_j>")
print("  reduced RDM:  rho_A = e^{-K}/Z,   K = sum_ij (h_A^T)_ij a_i^dag a_j")
print("  ent. Hamil.:  h_A = log((I - C_A) C_A^{-1})   [MINUS sign in exp; TRANSPOSE in K]")
print("                (transpose is essential for COMPLEX Hermitian C_A; the control")
print("                 caught it -- a blind import of h (no transpose) FAILS here)")
print("  occ<->energy: zeta = 1/(e^{eps}+1),  eps = log((1-zeta)/zeta)")
print("  entropy:      S = -sum[ zeta ln zeta + (1-zeta) ln(1-zeta) ]  (nats)")
print("  modular flow: sigma_t(x) = e^{iKt} x e^{-iKt};  KMS at beta=1 (own flow)")
print("  first law:    Delta<K> = DeltaS + O(delta^2)  (rel. entropy >= 0)")
print()
print("RESULT:", "ALL CONTROLS LOCK -> convention OWNED, proceed to substrate probe"
      if ok_all else "CONTROL FAILED -> VOID (premise falls; do not trust substrate read)")
import sys
sys.exit(0 if ok_all else 1)
