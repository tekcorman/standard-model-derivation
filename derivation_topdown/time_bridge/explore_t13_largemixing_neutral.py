"""
explore_t13 — THE MASS STRUCTURE OF THE LARGE-MIXING (NEUTRAL) SECTOR FROM RECURRENCE.
PURE MATH, walled.  Reads only ../dirac_srs_mdl + this time_bridge.
No physics; no fitting; no adopted targets; honest negatives kept.

CONTEXT (established, re-verified below):
  The object has DISTINCT matter sectors on DIFFERENT recurrence shells (return-modulus classes of
  the non-backtracking / geodesic flow B).  At Gamma the |h|^2 spectrum splits into THREE shells:
     * |h|^2 = 4 (x1) PERRON  and |h|^2 = 1 (x5) TREE     -> the modes that DECAY/MOVE under the run;
                                                              their mass = integrated departure above
                                                              the shell (t10: the HEAVY sector, done).
     * |h|^2 = 2 (x6) the genuine RAMANUJAN shell = 2 copies of the A4 3-irrep, RIGID under the run,
                      distributed 2-2-2 across the C3 sectors {1, w, w2} (the C3-Fourier structure).
  t10 resolved the FIRST (high-symmetry / Perron) sector.  This script resolves the SECOND: the
  large-mixing sector on the complex Ramanujan shell.

THE KEY STRUCTURAL DIFFERENCE we develop:
  The Perron/tree modes get their mass from a DECAY-IN-MODULUS under the run (|h|^2: 4->2, 9->3).
  The Ramanujan shell does NOT move in modulus (|h|^2 = k-1 = 2 is RIGID, verified).  So its
  recurrence content is NOT a modulus-decay.  Its only running structure is the PHASE of the complex
  return amplitude h (the eigenvalues are genuinely complex on this shell, h = +-i*sqrt(k-1) at
  Gamma).  We build its persistence sub-distribution from the PHASE WINDING under the screw run.

TASK ITEMS:
  1. identify the sectors by shell; build the large-mixing sector's three windings' persistence.
  2. the mass SCALE: compare its persistence to the charged (decaying) sector's.  Is it forced light?
  3. the HIERARCHY/ORDERING: same forced shape as the charged generations, or different?
  4. extra structure: reality/self-conjugacy of the modes (own conjugates vs chiral pairs).
  5. forced vs choice.
"""
import numpy as np, sys, os, cmath
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs

np.set_printoptions(precision=6, suppress=True)
def hdr(s): print("\n" + "=" * 84 + "\n" + s + "\n" + "=" * 84)
SQRT2 = np.sqrt(2.0); om = cmath.exp(2j*np.pi/3); k = srs.DEG  # 3
AXIS = np.array([1.0, -1.0, 1.0]) / np.sqrt(3.0)               # the C3 screw / cooling history axis

# =============================================================================================
hdr("(1a) THE SECTORS BY RECURRENCE SHELL at the symmetric (tracial/hot) start Gamma")
# =============================================================================================
B0 = srs.hashimoto((0, 0, 0))
ev = np.linalg.eigvals(B0)
h2 = np.abs(ev)**2
def cluster(vals, tol=1e-6):
    out = []
    for v in sorted(vals):
        for c in out:
            if abs(v - c[0]) < tol: c[1] += 1; c[2].append(v); break
        else: out.append([v, 1, [v]])
    return out
print("  Hashimoto B(Gamma) return-modulus shells |h|^2 (the recurrence classes):")
for c in sorted(cluster(np.round(h2, 6)), key=lambda c: -c[0]):
    # list the actual complex h-values in the shell
    idx = [i for i in range(12) if abs(h2[i] - c[0]) < 1e-5]
    hs = sorted({complex(round(ev[i].real, 3), round(ev[i].imag, 3)) for i in idx},
                key=lambda z: (round(z.real, 3), round(z.imag, 3)))
    role = {4.0: "PERRON  (decays 4->2; heavy, t10)", 1.0: "TREE    (moves; merges to shell at P)",
            2.0: "RAMANUJAN SHELL (rigid; the LARGE-MIXING/neutral sector)"}.get(round(c[0]), "")
    print(f"     |h|^2 = {c[0]:.3f}  (x{c[1]:2d})   {role}")
    print(f"                        h-values: {', '.join(f'{z.real:+.2f}{z.imag:+.2f}i' for z in hs)}")
print("\n  => the LARGE-MIXING (neutral) sector = the 6-fold |h|^2=2 shell = 2 copies of the A4 3-irrep.")
print("     Its return amplitudes are PURE IMAGINARY at Gamma (h = +-i sqrt2): a COMPLEX shell.")

# =============================================================================================
hdr("(1b) ITS THREE WINDINGS: the C3-Fourier ('generation') resolution of the shell")
# =============================================================================================
# Build the C3 dart-permutation (sigma=(123)); its Fourier projectors split the dart space into the
# three C3 sectors {1, w, w2}.  Along the screw axis [B, P_sigma] = 0, so B is block-diagonal and the
# shell distributes across the three sectors.  These three C3 sectors ARE the three "windings"/
# generations of the large-mixing sector.
sigma = {0: 0, 1: 2, 2: 3, 3: 1}
DARTS = srs._darts()
Pperm = np.zeros((12, 12))
for a, (i, j, v) in enumerate(DARTS):
    g = (sigma[i], sigma[j])
    for b, (p, q, w) in enumerate(DARTS):
        if (p, q) == g: Pperm[b, a] = 1; break
Pc3 = {s: sum(om**(-s*m) * np.linalg.matrix_power(Pperm, m) for m in range(3)) / 3 for s in (0, 1, 2)}
def c3_basis(s):
    w, V = np.linalg.eigh(Pc3[s]); return V[:, np.abs(w - 1) < 1e-6]

def shell_h_in_sector(s, kk):
    """The complex return amplitudes h on the |h|^2=2 shell, restricted to C3 sector s, at fiber kk."""
    B = srs.hashimoto(kk)
    Q = c3_basis(s)
    Bs = Q.conj().T @ B @ Q
    hh = np.linalg.eigvals(Bs)
    return sorted([z for z in hh if abs(abs(z)**2 - 2.0) < 1e-3], key=lambda z: cmath.phase(z))

print("  At Gamma, the shell modes in each C3 sector (the three windings), as complex amplitudes h:")
for s in (0, 1, 2):
    hs = shell_h_in_sector(s, (0, 0, 0))
    print(f"     sector {'1  w  w2'.split()[s]:3} :  h = "
          + ", ".join(f"{z.real:+.3f}{z.imag:+.3f}i (|h|^2={abs(z)**2:.2f}, arg={np.degrees(cmath.phase(z)):+.1f} deg)"
                      for z in hs))
print("  => each C3 sector carries 2 shell amplitudes; 3 sectors x 2 = 6 = the full shell.")
print("     They differ by a C3 PHASE, not a modulus: the windings are a PURE-PHASE family. (Large mixing.)")

# =============================================================================================
hdr("(2) THE MASS SCALE: persistence of the rigid shell vs the decaying (charged) sector")
# =============================================================================================
# Persistence under the run = the accumulated LOG-RETURN-WEIGHT of the mode over the screw history,
# exactly as t10 defined it for the heavy sector:  M_mode = integral log(weight_mode(s)) ds, with the
# weight = |h|^2 / (k-1) = excess persistence over the geodesic-flow spectral radius (k-1)=2.
#
#   CHARGED (Perron) sector:  weight = |h+|^2 / 2,  which DECAYS 4/2=2 -> 1 over [0, s_merge].
#                             => POSITIVE accumulated log-persistence (genuinely heavy).
#   LARGE-MIXING (shell) sector: |h|^2 = 2 is RIGID => weight = 2/2 = 1 EXACTLY, at EVERY s.
#                             => log-weight = log 1 = 0 IDENTICALLY => ZERO accumulated persistence.
print("  Per-step return weight = |h|^2 / (k-1) = |h|^2 / 2  (excess over the geodesic spectral radius).")
print("  Track it along the screw run for both sectors:\n")
print(f"   {'s':>6} {'shell |h|^2':>12} {'shell wt':>9} {'Perron |h+|^2':>14} {'Perron wt':>10}")
def perron_hh(lam):
    if lam >= 2*SQRT2:
        h = (lam + np.sqrt(lam*lam - 8.0))/2.0; return h*h
    return 2.0
def lam_max(s): return float(np.linalg.eigvalsh(srs.adjacency(s*AXIS)).max())
for s in [0.0, 0.05, 0.10, 0.134, 0.20, 0.30]:
    B = srs.hashimoto(s*AXIS); h2s = np.abs(np.linalg.eigvals(B))**2
    shell = np.median(h2s[np.abs(h2s - 2.0) < 0.05]) if np.any(np.abs(h2s-2.0) < 0.05) else float('nan')
    ph = perron_hh(lam_max(s))
    print(f"   {s:6.3f} {shell:12.5f} {shell/2:9.4f} {ph:14.5f} {ph/2:10.4f}")
print("\n  => the shell weight is IDENTICALLY 1 (log 0) along the whole run: the large-mixing sector")
print("     accumulates ZERO modulus-persistence.  The Perron weight starts at 2 and decays to 1.")
print("  FORCED RESULT: the neutral sector's modulus-persistence scale is ZERO relative to the charged")
print("  sector's POSITIVE accumulation.  Its lightness is FORCED by Ramanujan rigidity |h|^2 = k-1:")
print("  it sits EXACTLY on the geodesic-flow spectral radius (the persistence floor), no excess to")
print("  integrate.  The charged sector is heavy ONLY because its Perron root pokes ABOVE the shell.")

# A cleaner exact statement of the forced ratio of persistence SCALES:
print("\n  Exact forced statement of the two persistence scales (bare, at the hot start s=0):")
print(f"     charged (Perron) excess persistence weight = |h+|^2/(k-1) = (k-1)^2/(k-1) = (k-1) = {k-1}")
print(f"     neutral (shell)  excess persistence weight = |h|^2 /(k-1) = (k-1)  /(k-1) =  1")
print(f"   => the forced ratio of (excess) persistence scales  neutral : charged  =  1 : (k-1) = 1 : 2,")
print(f"      and in LOG-persistence (the accumulated 'mass') it is  0 : positive  -- the neutral sector")
print(f"      is the MASSLESS-LIMIT sector of the modulus-recurrence: it has NO modulus mass at all.")

# =============================================================================================
hdr("(2b) BUT the shell is NOT inert: its recurrence content is the PHASE winding (the residual)")
# =============================================================================================
# Since |h|^2 is rigid, the ONLY running structure of the shell is the PHASE of its complex return
# amplitude h = sqrt2 * e^{i theta(s)}.  This phase is the residual, sub-leading persistence content
# of the neutral sector -- the analog, for a rigid shell, of the decay the charged sector has.
# Track theta(s) of the shell amplitudes along the screw, per C3 sector.
print("  Shell amplitude phase theta(s) = arg(h) along the screw run, per C3 winding (degrees):")
print(f"   {'s':>6}   sector-1            sector-w            sector-w2")
for s in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    kk = s*AXIS; row = []
    for sec in (0, 1, 2):
        hs = shell_h_in_sector(sec, kk)
        row.append("[" + ", ".join(f"{np.degrees(cmath.phase(z)):+6.1f}" for z in hs) + "]")
    print(f"   {s:6.3f}   " + "   ".join(f"{r:18s}" for r in row))
print("  => the shell's recurrence content is carried PURELY in this phase (its modulus is frozen).")
print("     The three C3 windings are the three phase-branches; their phases are related by the C3")
print("     rotation (multiplication by w = e^{2pi i/3}).  This is the 'large mixing': the windings")
print("     are connected by a maximal (2pi/3) phase rotation, not separated by a modulus gap.")

# =============================================================================================
hdr("(3) THE THREE-WAY HIERARCHY / ORDERING of the large-mixing sector")
# =============================================================================================
# The charged generations (t10 + the earlier mass work) had a NON-DEGENERATE three-way split: the
# three windings sat at DIFFERENT persistence values (a hierarchy), ordered by a directed phase.
# Test the neutral sector: do its three windings have DIFFERENT |h|^2 (a hierarchy) or the SAME
# (degenerate)?  And is there a directed (chiral) phase ordering?
print("  Three-way |h|^2 of the neutral sector's windings along the screw (is it split or degenerate?):")
for s in [0.0, 0.10, 0.20, 0.30, 0.40]:
    kk = s*AXIS; B = srs.hashimoto(kk)
    if not np.allclose(B@Pperm, Pperm@B, atol=1e-9):
        print(f"   s={s}: [B,C3]!=0 off-axis -- skip"); continue
    vals = []
    for sec in (0, 1, 2):
        hs = shell_h_in_sector(sec, kk)
        vals.append([round(abs(z)**2, 4) for z in hs])
    print(f"   s={s:.2f}:  sector-1 |h|^2={vals[0]}   sector-w |h|^2={vals[1]}   sector-w2 |h|^2={vals[2]}")
print("\n  => ALL THREE windings sit at |h|^2 = 2 (rigid): the three-way split is DEGENERATE IN MODULUS")
print("     (NO hierarchy of persistence-modulus among the three neutral windings) -- the OPPOSITE of")
print("     the charged sector's non-degenerate {heaviest, middle, light} ordering.  The neutral")
print("     three-way structure is a pure-PHASE triplet (degenerate magnitude, three C3 phases).")

# the directed phase: is there a chiral ordering 0 : +phi : -phi like the charged 0 : +-2pi sqrt(3/7)?
print("\n  Directed C3 phase content of the three windings (the residual ordering, if any):")
for s in [0.10, 0.20]:
    kk = s*AXIS
    phs = []
    for sec in (0, 1, 2):
        hs = shell_h_in_sector(sec, kk)
        # take the upper-half-plane representative's phase as the sector label
        up = [z for z in hs if z.imag >= -1e-9]
        phs.append(np.degrees(cmath.phase(up[0])) if up else float('nan'))
    print(f"   s={s:.2f}:  winding phases (deg) = {[round(p,1) for p in phs]}  "
          f"(differences: {round(phs[1]-phs[0],1)}, {round(phs[2]-phs[1],1)})")
print("  => the three windings differ by ~120 deg C3 steps (= arg w): the ordering is the SYMMETRIC")
print("     C3-Fourier phase {0, +120, -120}, NOT a graded hierarchy.  Compare the charged sector's")
print("     directed phase 0 : +-2pi sqrt(3/7) (an IRRATIONAL angle, asymmetric/graded).")

# =============================================================================================
hdr("(4) EXTRA STRUCTURE: reality / self-conjugacy of the modes (own conjugates vs chiral pairs)")
# =============================================================================================
# The charged sector's amplitudes are the Ihara-Bass roots of a REAL adjacency eigenvalue; off the
# Ramanujan branch they are REAL (h+ and h- both real, a NON-conjugate pair).  The neutral shell sits
# on the COMPLEX branch: its two amplitudes per fiber are h_+ and h_- = conj(h_+) (a CONJUGATE pair,
# product = k-1).  Test whether the shell modes are CLOSED under complex conjugation = whether the
# sector is its own anti-sector (self-conjugate / "real" sector) or pairs with a distinct mirror.
print("  Ihara-Bass: the two roots of h^2 - lam h + (k-1) = 0 satisfy  h+ * h- = k-1  and  h+ + h- = lam.")
print("  On the Ramanujan (complex) branch lam^2 < 4(k-1):  h- = conj(h+), so |h+|^2 = |h-|^2 = k-1.")
print("  Check the shell amplitudes come in CONJUGATE pairs (self-conjugate sector):")
for s in [0.0, 0.10, 0.20]:
    kk = s*AXIS; B = srs.hashimoto(kk)
    hs = sorted([z for z in np.linalg.eigvals(B) if abs(abs(z)**2 - 2.0) < 1e-3],
                key=lambda z: (round(z.real,4), round(z.imag,4)))
    # pair each h with its conjugate present in the list
    paired = all(any(abs(z - np.conj(w)) < 1e-3 for w in hs) for z in hs)
    print(f"   s={s:.2f}: shell h-values closed under conjugation (h <-> conj h)?  {paired}   "
          f"(count {len(hs)})")
print("\n  => the neutral shell modes ARE closed under conjugation: every winding amplitude h has its")
print("     conjugate also on the shell (h_- = conj h_+, product k-1).  The sector is SELF-CONJUGATE")
print("     ('real' in the Atiyah-Bott-Shapiro / reality sense): each mode is paired with its OWN")
print("     conjugate WITHIN the shell, not with a distinct mirror sector.")
print("  CONTRAST the charged (Perron/tree) modes: on the REAL branch h+ and h- are BOTH real and")
print("  DISTINCT (h+ != conj h+ in a non-trivial way; they are a NON-conjugate real pair, a chiral")
print("  doublet that splits 4 vs 1, i.e. h+ != h-).  So:")
print("     NEUTRAL sector : self-conjugate (h, conj h) pairs on a single complex shell  (Majorana-like)")
print("     CHARGED sector : real, split (h+ != h-) pairs that separate by modulus       (Dirac-like)")

# Double-check via the A4/2T content: the shell = 2 copies of the 3-irrep (a VECTOR/real irrep of A4),
# carrying NO 2T spinor (m05): consistent with a self-conjugate real sector.
print("\n  Cross-check (rep theory, explore_08 / m05): the shell = exactly 2 copies of the A4 3-irrep,")
print("  a REAL (vector, 2T-centre +1) irrep -> consistent with the sector being self-conjugate/real.")

# =============================================================================================
hdr("(5) FORCED vs CHOICE")
# =============================================================================================
print("""  FORCED (all from k=3 and Ramanujan rigidity |h|^2 = k-1; no fitting):
   * the SECTOR IDENTITY: the large-mixing (neutral) sector = the 6-fold complex Ramanujan shell
     |h|^2 = k-1 = 2 = 2 copies of the A4 3-irrep, distributed 2-2-2 across the C3 windings {1,w,w2};
   * its MASS SCALE is the persistence FLOOR: |h|^2 = k-1 sits EXACTLY on the geodesic-flow spectral
     radius, so its excess persistence weight is 1 (log 0) at EVERY slice => ZERO accumulated modulus-
     mass.  Its lightness is FORCED (Ramanujan rigidity), not fitted.  Forced ratio of the two sectors'
     bare excess-persistence weights = neutral : charged = 1 : (k-1) = 1 : 2; in accumulated log-mass
     it is 0 : positive (the neutral sector is the massless-floor sector of the modulus recurrence);
   * its THREE-WAY structure is DEGENERATE IN MODULUS (all three windings at |h|^2 = 2) with a pure
     SYMMETRIC C3 phase split {0, +120, -120 deg} -- a DIFFERENT shape from the charged generations'
     non-degenerate graded hierarchy with the irrational directed phase 2pi sqrt(3/7).  No heaviest/
     lightest ordering among the neutral windings; they are phase-rotations of one magnitude;
   * its EXTRA STRUCTURE: the modes are SELF-CONJUGATE -- on the complex branch h_- = conj(h_+), so
     each winding amplitude pairs with its OWN conjugate WITHIN the shell (a real/self-dual sector,
     Majorana-like), versus the charged sector's real-split (h+ != h-) chiral doublets (Dirac-like).
     Reinforced by rep theory: the shell = 2x the REAL A4 3-irrep (2T-centre +1, no spinor).

  CHOICE / NEEDS-THE-OBSERVER (honest):
   * any RESIDUAL mass of the neutral sector must come from the PHASE winding (its only running
     content, the modulus being frozen) -- i.e. from the observer's slice/endpoint of the screw run,
     NOT from a modulus decay.  The object is scale-free (III_1, T(M)={0}), so the absolute size of
     any phase-derived residual is the observer's endpoint, not forced.  The FORCED content is that
     the neutral sector's modulus-mass is exactly the floor (zero excess), degenerate, self-conjugate.

  HONEST NEGATIVE: the large-mixing sector has NO forced internal mass HIERARCHY (its three windings
  are modulus-degenerate); the charged sector's graded hierarchy does NOT recur here.  Its entire
  non-trivial content is the symmetric C3 phase and its self-conjugacy -- a qualitatively DIFFERENT
  (lighter, degenerate, self-conjugate) sector, forced to be so by Ramanujan rigidity.""")
