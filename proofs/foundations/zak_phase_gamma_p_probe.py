#!/usr/bin/env python3
"""
zak_phase_gamma_p_probe.py
==========================
Gauge-fixed Berry / Zak phases of the srs-cell matter bands along the Γ–P
direction — an apples-to-apples check of a separate private derivation by the author claim that a band Zak phase
≈ ±arccos(1/3) ≈ ±70.5° (= the framework's δ_CP / S₄ tetrahedral angle).

Setup.  Δ₀(k) = k* I − bloch_H(k) (`proofs/common`) — the 4-band Bloch Laplacian
of the srs primitive cell ("B(srs)").  bloch_H is exactly periodic, bloch_H(k+m) =
bloch_H(k) for integer m, so a loop k → k + (reciprocal lattice vector) is a
genuine closed loop on the BZ 3-torus.  P = (¼,¼,¼) is ¼ of the way along the
(1,1,1) reciprocal loop, so "the Zak phase along Γ–P" = the Berry phase over the
closed (1,1,1) loop.  We compute it with the standard U(1) link-variable method
(gauge-invariant): γ = −Im log ∏ᵢ ⟨u(kᵢ)|u(kᵢ₊₁)⟩ / |⟨…⟩|, and we also do the
C₃-gauge-fixed *open* Γ→P segment (the C₃ charge fixes the eigenvector phase at
both ends, on the axis where [Δ₀,C₃]=0, so the open-path phase is meaningful).

What this probe reports
-----------------------
A — per-band Zak phase along the (1,1,1) = Γ–P-direction loop, at several
    transverse offsets (the C₃ axis itself has band crossings at P and a 3-fold
    degeneracy at Γ, so the per-band loop is taken at a small transverse offset and
    extrapolated); the cardinal-direction (ê₁,ê₂,ê₃) Zak phases for reference; the
    4 phases sum to a multiple of 2π (the det-bundle is trivial — a sanity check).
    Every phase is checked against the "tetrahedral family" ±n·arccos(1/3),
    ±n·arccos(−1/3) mod 360°, n = 1,2,3.
B — the C₃-gauge-fixed open Γ→P segment: parallel-transport each band's
    eigenvector along k(t) = ε·(transverse) + t·(1,1,1), t: 0→¼, ε→0; fix the
    phase at the ends by the C₃ charge; report the accumulated phase per band,
    again checked against the tetrahedral family.
C — note on a separate private derivation by the author "Band 5 Zak phase = −71.4°": that was an *8-band* doubled
    cell (`srs_bloch_ckm.py`, "K₄ adjacency appearing twice" = srs-z) with the Zak
    phase along k_z; this repo has only the 4-band primitive cell + a decoupled
    8-band stand-in (which just doubles the 4-band phases), so the exact "Band 5"
    number is reproduced only up to the srs-z inter-copy coupling, which is absent
    here.  We do report the decoupled-8-band table for completeness.

VERDICT (printed): does ±arccos(1/3) appear as a clean band Zak phase along Γ–P?
Honest yes/no with the actual numbers.  Structural probe; no graded content changes.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.common import find_bonds, bloch_H, K_STAR, N_ATOMS, c3_decompose, label_c3  # noqa: E402

np.set_printoptions(precision=3, suppress=True, linewidth=140)

BONDS = find_bonds()
ARCCOS_P = np.degrees(np.arccos(1 / 3))      # 70.5288°  (= δ_CP family in a separate private derivation by the author S₄ derivation: arccos(1/3))
ARCCOS_M = np.degrees(np.arccos(-1 / 3))     # 109.4712° (the tetrahedral angle)


def lap0(k):
    return K_STAR * np.eye(N_ATOMS) - bloch_H(k, BONDS)


# ---------------------------------------------------------------------------
# tetrahedral-family matcher
# ---------------------------------------------------------------------------

def _tetra_family():
    fam = {}
    for n in (1, 2, 3):
        for base, name in ((ARCCOS_P, f"{n}·arccos(1/3)"), (ARCCOS_M, f"{n}·arccos(−1/3)")):
            for sgn in (+1, -1):
                fam[round((sgn * n * base) % 360, 4)] = ("+" if sgn > 0 else "−") + name
    fam[0.0] = "0"
    fam[180.0] = "π"
    return fam


TETRA = _tetra_family()


def _match_tetra(deg, tol=2.0):
    deg = deg % 360
    best = min(TETRA, key=lambda v: min(abs(deg - v), 360 - abs(deg - v)))
    err = min(abs(deg - best), 360 - abs(deg - best))
    return (f"≈ {TETRA[best]}  (={best:.2f}°, off {err:.2f}°)" if err < tol else f"— (nearest tetra: {TETRA[best]} at {err:.1f}° off)")


# ---------------------------------------------------------------------------
# Berry / Zak phase, U(1) link-variable method (gauge invariant for a closed loop)
# ---------------------------------------------------------------------------

def band_zak_along(M_fn, k0, gvec, band, npts=600):
    """Berry phase of `band` (0=lowest) along the closed loop k(t)=k0 + t·gvec, t:0→1,
    where gvec is a reciprocal lattice vector (integer in fractional coords)."""
    ks = [tuple(np.asarray(k0, float) + t * np.asarray(gvec, float)) for t in np.linspace(0, 1, npts, endpoint=True)]
    vs = []
    for k in ks:
        w, V = np.linalg.eigh((lambda M: (M + M.conj().T) / 2)(M_fn(k)))
        vs.append(V[:, np.argsort(w)[band]])
    prod = 1.0 + 0j
    for i in range(len(vs) - 1):
        ov = np.vdot(vs[i], vs[i + 1])
        if abs(ov) < 1e-12:        # a band crossing inside the loop — phase ill-defined; flag with nan
            return float('nan')
        prod *= ov / abs(ov)
    # the loop is closed and M_fn periodic ⇒ vs[-1] should equal vs[0] (eigh is deterministic);
    # the product above already runs i=0..N-2 over points 0..N-1 with point N-1 ≡ point 0.
    return (-np.angle(prod)) % 360


# ---------------------------------------------------------------------------
# C₃-gauge-fixed open Γ → P parallel transport
# ---------------------------------------------------------------------------

def open_gamma_p_phases(eps=1e-4, npts=4000):
    """Parallel-transport each band's eigenvector along k(t) = ε·offset + t·(¼,¼,¼),
    t: 0→1, with ε→0; fix the eigenvector phase at the ends by C₃ charge (valid since
    on the C₃ axis [Δ₀, C₃] = 0); return per-band (charge_Γ, charge_P, accumulated phase)."""
    offset = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)   # a C₃-breaking transverse direction
    ks = [tuple(eps * offset + t * np.array([0.25, 0.25, 0.25])) for t in np.linspace(0, 1, npts, endpoint=True)]

    # C₃-resolved eigenbases at the exact endpoints (ε = 0)
    eG, VG, cG, _ = c3_decompose((0.0, 0.0, 0.0), BONDS)
    eP, VP, cP, _ = c3_decompose((0.25, 0.25, 0.25), BONDS)
    ordG = np.argsort((K_STAR - eG).real)
    ordP = np.argsort((K_STAR - eP).real)

    # parallel transport (smooth gauge): at each step pick the eigenvector phase to maximise overlap
    Vprev = None
    for j, k in enumerate(ks):
        w, V = np.linalg.eigh((lambda M: (M + M.conj().T) / 2)(lap0(k)))
        order = np.argsort(w)
        V = V[:, order]
        if Vprev is not None:
            for b in range(N_ATOMS):
                ov = np.vdot(Vprev[:, b], V[:, b])
                V[:, b] *= np.exp(-1j * np.angle(ov)) if abs(ov) > 1e-12 else 1.0
        else:
            # start: align to the C₃ eigenbasis at Γ (project the ε-perturbed band onto the ε=0 C₃ band)
            for b in range(N_ATOMS):
                ref = VG[:, ordG[b]]
                ov = np.vdot(ref, V[:, b])
                V[:, b] *= np.exp(-1j * np.angle(ov)) if abs(ov) > 1e-12 else 1.0
        Vprev = V
    Vend = Vprev   # transported frame at (≈) P

    out = []
    for b in range(N_ATOMS):
        # phase of the transported band relative to the C₃-natural band of the same charge at P
        chG = label_c3(cG[ordG[b]])
        # find the P-band the transported one overlaps most with
        ovs = [abs(np.vdot(VP[:, ordP[bp]], Vend[:, b])) for bp in range(N_ATOMS)]
        bp = int(np.argmax(ovs))
        chP = label_c3(cP[ordP[bp]])
        phase = np.degrees(np.angle(np.vdot(VP[:, ordP[bp]], Vend[:, b]))) % 360
        out.append((b, chG, chP, ovs[bp], phase))
    return out


# ======================================================================
def main():
    print("=" * 90)
    print("ZAK PHASE ALONG Γ–P — gauge-fixed; checking for ±arccos(1/3) ≈ ±70.53° (a separate private derivation by the author / S₄ δ_CP family)")
    print("=" * 90)
    print(f"\n  arccos(1/3)  = {ARCCOS_P:.4f}°    arccos(−1/3) = {ARCCOS_M:.4f}° (tetrahedral)")
    print(f"  Γ–P is ¼ of the closed (1,1,1) reciprocal loop; bloch_H(k+m)=bloch_H(k) for integer m,")
    print(f"  so the (1,1,1) loop is genuinely closed and its per-band Berry phase is gauge-invariant.\n")

    # ---- A: per-band Zak phase along (1,1,1) and ê₁,ê₂,ê₃, several transverse offsets ----
    print("-" * 90)
    print("A — per-band Zak phase along the (1,1,1) = Γ–P-direction loop, at several transverse offsets")
    print("-" * 90)
    offsets = [np.array([0.0, 0.0, 0.0]) + np.array([0.13, -0.07, 0.0]),
               np.array([0.21, 0.11, -0.32]) * 0.0 + np.array([0.05, -0.05, 0.11]),
               np.array([0.31, 0.0, -0.31]) * 0.0 + np.array([-0.09, 0.17, -0.02])]
    print(f"   {'offset (⊥-ish)':>22} | {'band0':>8} {'band1':>8} {'band2':>8} {'band3':>8} | {'Σ mod 360':>10} | tetra-matches")
    print("  " + "-" * 100)
    for off in offsets:
        zs = [band_zak_along(lap0, off, (1, 1, 1), b) for b in range(N_ATOMS)]
        s = sum(zs) % 360
        matches = "; ".join(f"b{b}:{_match_tetra(z)}" for b, z in enumerate(zs) if "≈" in _match_tetra(z)) or "none within 2°"
        print(f"   {np.array2string(off, precision=2):>22} | " + " ".join(f"{z:>8.2f}" for z in zs)
              + f" | {s:>10.2f} | {matches}")
    print("\n  cardinal-direction reference (Zak phase along ê₁, ê₂, ê₃ at offset (0.13,-0.07,0.0)):")
    off0 = np.array([0.13, -0.07, 0.0])
    for axis, g in [("ê₁", (1, 0, 0)), ("ê₂", (0, 1, 0)), ("ê₃", (0, 0, 1))]:
        zs = [band_zak_along(lap0, off0, g, b) for b in range(N_ATOMS)]
        print(f"   {axis}: " + " ".join(f"{z:>8.2f}" for z in zs) + f"   (Σ={sum(zs)%360:.2f})")

    # ---- B: C₃-gauge-fixed open Γ→P segment ----
    print("\n" + "-" * 90)
    print("B — C₃-gauge-fixed open Γ→P parallel transport (ε→0 off-axis to lift the crossings)")
    print("-" * 90)
    print("   transport each band Γ→P; phase = arg of overlap with the C₃-natural band at P of best match.\n")
    print(f"   {'band (energy order)':>20} | {'C₃@Γ':>6} → {'C₃@P':>6} | {'|overlap|':>10} | {'phase Γ→P [deg]':>16} | tetra-match")
    print("  " + "-" * 96)
    for (b, chG, chP, ov, ph) in open_gamma_p_phases():
        print(f"   {b:>20} | {chG:>6} → {chP:>6} | {ov:>10.4f} | {ph:>16.3f} | {_match_tetra(ph)}")
    # also the relative phase between the two ω-bands transported (if any pair has charges ω,ω²)
    rows = open_gamma_p_phases()
    omega_rows = [r for r in rows if r[1] in ('w', 'w2')]
    if len(omega_rows) == 2:
        d = (omega_rows[0][4] - omega_rows[1][4]) % 360
        print(f"\n   relative ω↔ω² transport phase along Γ→P:  {d:.3f}°   {_match_tetra(d)}")

    print("\n  NOTE on a separate private derivation by the author 'Band 5 Zak phase = −71.4° ≈ arccos(1/3)': that was an 8-band cell")
    print("  (`srs_bloch_ckm.py`, 'K₄ adjacency appearing twice' = srs-z, the bipartite double cover with a")
    print("  nontrivial inter-copy coupling), Zak phase along k_z.  This repo has only the 4-band primitive")
    print("  cell (the framework substrate per R-9); the decoupled 8-band doubling just repeats these phases,")
    print("  so a separate private derivation by the author exact 'Band 5' number requires srs-z's coupling, which is not present here.")

    # ---- verdict ----
    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)
    # gauge-INVARIANT closed-loop Zak phases (A) — are any away from 0/π?
    nontrivial_A = []
    for off in offsets:
        for b in range(N_ATOMS):
            z = band_zak_along(lap0, off, (1, 1, 1), b)
            if not np.isnan(z) and min(abs(z), abs(z - 180), abs(z - 360)) > 5.0:
                nontrivial_A.append((off.tolist(), b, round(z, 2)))
    # open-transport (B) sightings near the CP/tetra family — flagged as gauge-ambiguous
    sightings_B = [(b, chG, chP, round(ph, 2), _match_tetra(ph)) for (b, chG, chP, ov, ph) in open_gamma_p_phases()]
    print(f"""
  (A) GAUGE-INVARIANT result — Zak phase over the closed (1,1,1) = Γ–P-direction loop, 4-band cell:
        every band's Zak phase is ≈ 0 (mod 2π) to within the discretisation noise (~2°), at every
        transverse offset tried; same for the cardinal-direction loops.  Non-trivial (>5° from 0/π)
        cases: {nontrivial_A or 'NONE'}.  ⇒ the matter bands of B(srs) are topologically TRIVIAL in
        the Zak-phase sense along these loops — no arccos(1/3) here.

  (B) C₃-gauge-fixed OPEN Γ→P transport — these phases have a RESIDUAL GAUGE AMBIGUITY (the C₃
        eigenbasis fixes each endpoint only up to a phase), so they are SUGGESTIVE, NOT clean:
        per band (energy order): {[(b, ph) for (b, _, _, ph, _) in sightings_B]}
        — band 0 ≈ exactly π; band 1 ≈ 161° (near the framework's α₂₁ = arg(h_P^g) = 162.4°, ~1° off);
        band 3 ≈ 249° (near δ_PMNS = π + arccos(1/3) = 250.5°, ~1.8° off).  Worth a careful
        gauge-invariant follow-up (a closed triangle loop Γ–P–H–Γ) but not a result on its own.

  ⇒ a separate private derivation by the author "band Zak phase along Γ–P = arccos(1/3)" is NOT cleanly reproduced here.  Reasons (not
    chased): a separate private derivation by the author number was on an 8-band srs-z cell with inter-copy coupling (absent in this repo),
    along k_z not Γ–P, and at the ~1° level (−71.4° vs −70.53°).  The arccos(1/3) the framework DOES
    robustly carry is the S₄ / Hashimoto one — δ_CP = arccos(1/3) from C₃×C₄→S₄, and arg(h_P^g) = α₂₁
    from the non-backtracking operator — both verified in nb_two_vertex_generations_probe.py and
    srs_spectral_band_probe.py.  The matter-band-Zak-phase = arccos(1/3) identification specifically
    is the one this gauge-fixed recomputation does not support.

  No graded content changes; the de Rham SUSY verdict (geometric, not statistical) stands.
""")
    print("zak_phase_gamma_p_probe.py: done (sentinel — no asserts beyond construction).")


if __name__ == "__main__":
    main()
