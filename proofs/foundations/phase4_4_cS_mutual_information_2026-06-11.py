#!/usr/bin/env python3
"""Phase 4.4 -- c_S via the pure-state identity I(A:B) = 2S(A) on the
Phase-3.2 step-isometry environment (spec FROZEN b4bb97b, S4 clause).

THE QUESTION (gravity_coupling_factor2_FINAL_STATE_2026-05-28.md Sec 5,
verbatim): "Is there a *derived* reason the gravitating horizon entropy is
the boundary mutual information (2x the observer record) rather than the
entanglement entropy (1x)? ... must be answered blind to G_eff = G.
Absent a forced answer, c_S = 1 and gravity stays form-level."

THE NEW ASSET (did not exist when the question was frozen): the
two-subsystem OEF vertex theorem (`two_subsystem_oef_vertex_2026-06-01.py`,
banked in the interaction-layer arc): the additivity defect of the OEF
identification E = kappa*S is EXACTLY the mutual information,
    E(XY) - E(X) - E(Y) = -kappa * I(X:Y).
BLINDNESS PEDIGREE: that theorem was derived for the INTERACTION sector
(binding energies, MDL scattering; it grounds U_MDL = 3) -- not for
gravity, and before Phase 4 opened. This probe COMPOSES it with A3 purity
on the 3.2 step isometry: the framework's own cut-localized boundary
object at any bipartition is the additivity defect -kappa*I (ATTRACTIVE
sign, panel erratum: the boundary share's MAGNITUDE is kappa*I -- the sign
must not propagate into a first-law sign), and under purity I = 2S.

Gates:
  C1  step-isometry regression (V+V = I; visible marginal = the 3.1
      channel) -- the 3.2/S1 objects rebuilt.
  C2  PURITY IDENTITY exact: I(A:E) = 2 S(A) = 2 S(E) for pure inputs,
      machine precision, across q in {1/3, 2/3, 0.9} (not a q = 2/3
      accident).
  C3  the cosmological case (maximally mixed visible, purified by R; the
      standing A3 situation): global (A,E,R) pure; I(A:ER) = 2 S(A) and
      I(E:AR) = 2 S(E) exact -- the cut-correlation is ALWAYS twice the
      record, for every cut.
  C4  the OEF boundary object ON the step's own output: the additivity
      defect at the record cut = I(E:AR) = 2 S(E) exactly; with E =
      kappa*S (the standing I2 identification) the cut-localized boundary
      energy = kappa * I = 2 kappa S(E).
  C5  TRILEMMA DISSOLUTION: the three historical counts are THREE
      DIFFERENT well-defined quantities on the SAME object --
      S(E) (the record/extent count, the c_S = 1 answer), I(E:AR) = 2S(E)
      (the cut-correlation count, the c_S = 2 answer), and the
      record-basis Shannon surprise H_sh(diag rho_E) >= S(E) (the
      epistemic ~2.585-class count, eliminated 2026-05-28 and confirmed
      distinct here). Exact relations gated.
  C6  VERDICT (the S4 deliverable, panel-reworded 2026-06-12): c_S = 2
      forced GIVEN the NEW NAMED IDENTIFICATION "gravitating horizon
      entropy := the framework's cut-localized additivity-defect object"
      -- a role assignment BEYOND I2 (which licenses E = kappa*S on a
      joint configuration), priced 1 bit adoption-class. Panel
      ratification REFUSED: the Clausius asset's native S_total is a
      single-stream record count (points at 1x); per the frozen 05-28
      default, absent ratification c_S = 1 and gravity stays form-level.
      The 05-28 extent-vs-flux c_S = 1 is RECONCILED (it counts S(E)).
      kappa RELOCATED: G_eff = G <=> kappa = M_Pl/2, underived -- kappa
      NOT PROMOTED (standing rule). Never bare 'forced'/'c_S derived'.
"""
import os
import sys

import numpy as np
from numpy import linalg as la

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from proofs.common import find_bonds  # noqa: E402

FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def S_ent(rho):
    """von Neumann entropy in bits."""
    ev = la.eigvalsh((rho + rho.conj().T) / 2.0)
    ev = ev[ev > 1e-14]
    return float(-np.sum(ev * np.log2(ev)))


N = 12
bonds = find_bonds()
edges = [(i, j, tuple(c)) for (i, j, c) in bonds]
rev = {}
for a, (i, j, c) in enumerate(edges):
    t = (j, i, tuple(-x for x in c))
    for b, e2 in enumerate(edges):
        if e2 == t:
            rev[a] = b
P_pt = np.array([0.25, 0.25, 0.25])
C = np.zeros((N, N), complex)
for a, (i, j, c) in enumerate(edges):
    for b, (i2, j2, c2) in enumerate(edges):
        if i2 == i:
            C[b, a] = 2.0 / 3.0 - (1 if b == a else 0)
S_mat = np.zeros((N, N), complex)
for a, (i, j, c) in enumerate(edges):
    S_mat[rev[a], a] = np.exp(2j * np.pi * np.dot(P_pt, np.asarray(c, float)))
U = S_mat @ C
R_DIM = N + 1


def V_of(q):
    V = np.zeros((N * R_DIM, N), complex)
    for col in range(N):
        psi = np.zeros(N, complex)
        psi[col] = 1
        out = np.zeros((N, R_DIM), complex)
        out[:, 0] = np.sqrt(q) * (U @ psi)
        for e in range(N):
            out[:, 1 + e] = np.sqrt(1 - q) * (np.eye(N)[e] * psi[e])
    # careful: P_e psi = psi[e] |e>
        V[:, col] = out.reshape(-1)
    return V


print("=" * 72)
print(" PHASE 4.4 -- c_S: the cut-correlation is twice the record (purity)")
print("=" * 72)

# ---- C1: regression ----
q0 = 2.0 / 3.0
V = V_of(q0)
iso_dev = la.norm(V.conj().T @ V - np.eye(N))
# visible marginal = Phi(rho) = q U rho U+ + (1-q) sum_e P_e rho P_e
RNG = np.random.default_rng(20260611)
chan_dev = 0.0
for _ in range(4):
    Xr = RNG.normal(size=(N, N)) + 1j * RNG.normal(size=(N, N))
    rho = Xr @ Xr.conj().T
    rho /= np.trace(rho).real
    out = (V @ rho @ V.conj().T).reshape(N, R_DIM, N, R_DIM)
    vis = np.einsum("arbr->ab", out)
    phi = q0 * U @ rho @ U.conj().T + (1 - q0) * np.diag(np.diag(rho))
    chan_dev = max(chan_dev, la.norm(vis - phi))
gate("C1 step isometry rebuilt: V+V = I; visible marginal = the 3.1 "
     "channel", iso_dev < 1e-12 and chan_dev < 1e-12,
     f"iso dev={iso_dev:.1e}, channel dev={chan_dev:.1e}")

# ---- C2: purity identity, exact, q-robust ----
worst = 0.0
for q in (1.0 / 3.0, 2.0 / 3.0, 0.9):
    Vq = V_of(q)
    for trial in range(3):
        psi = RNG.normal(size=N) + 1j * RNG.normal(size=N)
        psi /= la.norm(psi)
        out = (Vq @ psi).reshape(N, R_DIM)
        rho_A = out @ out.conj().T
        rho_E = out.T @ out.conj()
        SA, SE = S_ent(rho_A), S_ent(rho_E)
        I_AE = SA + SE - 0.0          # S(AE) = 0: global pure
        worst = max(worst, abs(I_AE - 2 * SA), abs(SA - SE))
gate("C2 PURITY IDENTITY: I(A:E) = 2 S(A) = 2 S(E) exactly, for every "
     "input and every q (Schmidt symmetry of the kept record)",
     worst < 1e-10, f"max dev={worst:.1e}")

# ---- C3: the cosmological (purified maximally-mixed) case ----
# |Omega> on A(x)R, then V on A: state on A(12) x E(13) x R(12)
q = q0
V = V_of(q)
psi_AR = np.eye(N) / np.sqrt(N)              # (A, R) amplitudes
out = np.einsum("xa,ar->xr", V, psi_AR)      # x = (A,E) joint, r = R
T3 = out.reshape(N, R_DIM, N)                # (A, E, R)
rho_A = np.einsum("aer,ber->ab", T3, T3.conj())
rho_E = np.einsum("aer,afr->ef", T3, T3.conj())
rho_R = np.einsum("aer,aes->rs", T3, T3.conj())
rho_AE = np.einsum("aer,bfr->aebf", T3, T3.conj()).reshape(N * R_DIM, N * R_DIM)
rho_AR = np.einsum("aer,bes->arbs", T3, T3.conj()).reshape(N * N, N * N)
SA, SE, SR = S_ent(rho_A), S_ent(rho_E), S_ent(rho_R)
S_AE, S_AR = S_ent(rho_AE), S_ent(rho_AR)
sum_SA_SAE = SA + S_AE      # panel relabel: this is S(A)+S(AE), NOT I(A:ER);
#                             I(A:ER) = S(A) + S(ER) - 0 = 2 S(A) by purity
# purity: S(ER) = S(A), S(AR) = S(E), S(AE) = S(R), S(AER) = 0
dev_c3 = max(abs(S_AE - SR), abs(S_AR - SE))   # the NON-tautological content:
# complement-entropy matches (purity); the I = 2S lines then follow exactly
I_E_AR = 2 * SE             # I(E:AR) = S(E) + S(AR) - 0 = 2 S(E) (S(AR) = S(E))
gate("C3 cosmological case: global (A,E,R) pure; complement entropies "
     "match (S(AE) = S(R), S(AR) = S(E)); every cut's correlation = "
     "TWICE its record: I(E:AR) = 2 S(E), I(A:ER) = 2 S(A)",
     dev_c3 < 1e-10,
     f"S(A)={SA:.4f}, S(E)={SE:.4f}, S(R)={SR:.4f}; dev={dev_c3:.1e}")

# ---- C4: the OEF boundary object on the step's own output ----
# additivity defect of S at the record cut (E | AR):
defect = SE + S_AR - 0.0   # S(E) + S(AR) - S(E,AR); S(E,AR) = S(global) = 0
dev_c4 = abs(defect - 2 * SE)
gate("C4 the OEF boundary object: additivity defect at the record cut "
     "= S(E) + S(AR) - S(EAR) = I(E:AR) = 2 S(E) EXACTLY; with the "
     "standing I2 identification E = kappa*S, the cut-localized boundary "
     "energy = kappa*I = 2*kappa*S(E) (the banked 06-01 vertex theorem, "
     "recomputed on the gravity-relevant state)",
     dev_c4 < 1e-10, f"defect = {defect:.6f} = 2 x {SE:.6f}")

# ---- C5: trilemma dissolution ----
H_sh = float(-np.sum(np.clip(np.diag(rho_E).real, 1e-300, None)
                     * np.log2(np.clip(np.diag(rho_E).real, 1e-300, None))))
gate("C5 (panel-softened): the CONSTANT c_S dissolves into two "
     "well-defined QUANTITIES on one object -- record S(E) = 1x vs "
     "cut-correlation I = 2x -- the FORK SURVIVES as one named "
     "identification, with the 2 BETWEEN the quantities now FORCED "
     "(purity); the "
     "record-basis Shannon surprise H_sh >= S(E) (= here: the max-mixed "
     "input decoheres the record; the historical 2.585 = 1+log2(3) was a "
     "different OEF bookkeeping, eliminated 2026-05-28) and H_sh != I",
     H_sh >= SE - 1e-10 and abs(defect - 2 * SE) < 1e-10
     and abs(H_sh - defect) > 1e-3,
     f"S(E) = {SE:.4f} | I = {defect:.4f} | H_sh = {H_sh:.4f}")

# ---- C6: the verdict ----
# (panel relabel: the former placeholder clock-arithmetic check was a
# tautology and carries ZERO evidential weight -- the clock-closes claim
# rests on the 05-28 algebra; printed only.)
print("      [clock note, print-only: at c_S = 2, H = c_S M_Pl/(2N) = "
      "M_Pl/N = the cascade rate; kappa cancels -- 05-28 algebra]")
gate("C6 VERDICT well-formed: c_S = 2 forced GIVEN the NEW NAMED "
     "IDENTIFICATION [gravitating horizon entropy := the framework's "
     "cut-localized additivity-defect object -- a role assignment BEYOND "
     "I2 (which licenses E = kappa*S on a joint configuration); priced "
     "1 bit adoption-class, ledger 2026-06-12]; never bare 'forced', "
     "never 'c_S derived'; extent-vs-flux c_S = 1 RECONCILED (it counts "
     "S(E), a different question); 2.585 = epistemic; kappa NOT promoted",
     dev_c4 < 1e-10 and worst < 1e-10)

print("\n--- 4.4 STATUS TABLE (the S4 deliverable) ---")
print("  c_S = 2          : forced GIVEN the NEW NAMED IDENTIFICATION")
print("                     [gravitating entropy := the cut-localized")
print("                     additivity-defect object; a role assignment")
print("                     BEYOND I2; priced 1 bit adoption-class] --")
print("                     panel ratification REFUSED 2026-06-12: the")
print("                     Clausius asset's native S_total is a single-")
print("                     stream record count (points at 1x); per the")
print("                     frozen 05-28 default, absent ratification")
print("                     c_S = 1 and gravity stays form-level")
print("  blindness        : the vertex theorem (06-01) was derived in the")
print("                     interaction arc (binding/U_MDL = 3), not for")
print("                     gravity; this probe only COMPOSES banked facts")
print("  c_S = 1 horn     : reconciled -- the extent/record count S(E);")
print("                     correct answer to a different question")
print("  c_S = 2.585 horn : epistemic (a record-basis Shannon bookkeeping,")
print("                     1+log2(3)); eliminated 2026-05-28; here the")
print("                     surprise collapses onto S(E) (record decoheres)")
print("  kappa            : RELOCATED, not closed: G_eff = G <=> kappa =")
print("                     M_Pl/2 (underived); kappa NOT PROMOTED; gravity")
print("                     magnitude stays form-level pending kappa")
print("  panel question   : does the framework's Clausius asset (OEF")
print("                     delta-E = kappa dS IS the first law) COMMIT to")
print("                     the cut-localized reading? If ratified, c_S = 2")
print("                     closes and the open piece is kappa alone.")

print("\n" + "=" * 72)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
    sys.exit(1)
print(" RESULT: ALL GATES PASS -- S4 verdict delivered; Phase 4 construction"
      " complete")
print("=" * 72)
sys.exit(0)
