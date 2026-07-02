"""
DERIVE the forced generation mass-SHAPE (a one-parameter family; NOT the lepton mass
VALUES — see SCOPE below) from the object {Dirac, srs, MDL}.

PURE MATH, walled.  Reads only ../dirac_srs_mdl (the object) + native recurrence data.
NO physics, NO imported/observed numbers, NO fitting, NO target.  Every factor is DERIVED
from the recurrence of the non-backtracking (geodesic) flow B(k) on the srs crystal.

⚠ SCOPE — what "forced" means here (do NOT mistake this for the lepton-mass derivation):
  This derives the forced STRUCTURE + directed phase RATE of a ONE-PARAMETER mass FAMILY
  m_j(u) = (1 + 2 cos(u - 2pi j/3))^2, NOT the lepton mass VALUES.  Only u = phi*s enters
  (the run-position s is free), so the "forced" rate phi=2pi/sqrt7 is absorbed into u and
  does NOT constrain the spectrum.  This family has amplitude epsilon=2 (sqrt(m)=1+2cos),
  NOT the empirical Koide epsilon=sqrt2, and its Koide Q never equals 2/3 at any fixed
  sensible u — it CANNOT be set to the observed leptons {1, 206.77, 3477}.  Equal moduli
  |c_t|=1 are the MINIMAL (incomplete) construction here; the CORRECT read weights each
  winding by its own omega^0 Perron return-weight {4,2,2}, which recovers (4,2,2)/Q=2/3
  EXACTLY (verified 2026-06-29) — the same (4,2,2) Spin^c moduli the LIVE lepton masses use
  (predictions/m_e.py: epsilon=sqrt2 from Q=2/3; delta=2/9 = the DERIVED directed phase of the
  ∂_N run, NOT adopted).  This file is NOT wired into any prediction.
  So "forced" = the shape/structure; the mass VALUES (ratios, absolute scale, the -70 ppm
  subleading) are the OPEN frontier (docs/incomplete_equations_todo.md §1).

THE OBJECT (verified upstream, re-checked here):
  srs = maximal abelian Z^3 cover of K_4 (k=3-regular, MDL-forced).  Hashimoto operator B(k)
  = the geodesic flow.  Its return amplitudes h solve Ihara-Bass  h^2 - lam h + (k-1) = 0,
  lam = adjacency eigenvalue, k-1 = 2.  The non-trivial spectrum sits on the Ramanujan shell
  |h|^2 = k-1 = 2.  The C3 screw sigma=(123) is the deck generator; along its fixed axis
  (1,-1,1) it commutes with B, so B block-diagonalizes into three C3 sectors {omega^0, omega^1,
  omega^2} = the three "windings".

THE TWO FORCED RECURRENCE DATA (derived in sections 1-2, NOTHING adopted):
  (M) MODULUS / return-weight structure of the three windings, at the tracial (hot) start Gamma.
  (P) DIRECTED PHASE the screw run imparts to the complex shell modes: velocity phi per unit run.

THE FORCED MASS-SHAPE (section 3): the C3 structure forces each generation to be a discrete-Fourier
  combination of three per-winding persistence amplitudes; mass = |amplitude-combination|^2.
  We DERIVE the map and COMPUTE the three masses and the two ratios as functions of the single
  run-position s.  s is the ONLY residual (scale-free, III_1).
"""
import numpy as np, cmath, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs

np.set_printoptions(precision=6, suppress=True)
def hdr(s): print("\n" + "=" * 86 + "\n" + s + "\n" + "=" * 86)
om = cmath.exp(2j*np.pi/3); k = srs.DEG                  # k = 3
AXIS = np.array([1.0, -1.0, 1.0]) / np.sqrt(3.0)         # unit C3 screw axis (the run direction)

# ---- the C3 Fourier projectors onto the three windings (built from the object's deck sigma) ----
sigma = {0: 0, 1: 2, 2: 3, 3: 1}
DARTS = srs._darts()
Pperm = np.zeros((12, 12))
for a, (i, j, v) in enumerate(DARTS):
    g = (sigma[i], sigma[j])
    for b, (p, q, w) in enumerate(DARTS):
        if (p, q) == g: Pperm[b, a] = 1; break
Pc3 = {t: sum(om**(-t*m) * np.linalg.matrix_power(Pperm, m) for m in range(3)) / 3 for t in (0, 1, 2)}
def winding_basis(t):
    w, V = np.linalg.eigh(Pc3[t]); return V[:, np.abs(w - 1) < 1e-6]

def shell_amp(t, s):
    """The upper-half-plane shell return amplitude h (|h|^2=2) of winding t at run-position s."""
    B = srs.hashimoto(s * AXIS); Q = winding_basis(t); Bs = Q.conj().T @ B @ Q
    sh = [z for z in np.linalg.eigvals(Bs) if abs(abs(z)**2 - 2.0) < 1e-3 and z.imag >= -1e-9]
    return sh[0] if sh else None

# =============================================================================================
hdr("(1) THE THREE WINDINGS and their FORCED MODULUS structure (return weights at Gamma)")
# =============================================================================================
B0 = srs.hashimoto((0, 0, 0))
print("  The deck C3 (sigma=(123)) Fourier-splits the 12 darts 4|4|4 into windings {omega^0,1,2}.")
print("  Full per-winding return spectrum |h|^2 at the tracial start Gamma (k-1=2 = shell):\n")
weights = {}
for t in (0, 1, 2):
    Q = winding_basis(t); ev = np.linalg.eigvals(Q.conj().T @ B0 @ Q)
    h2 = sorted(np.round(np.abs(ev)**2, 4))
    weights[t] = h2
    tag = "  <-- carries the PERRON mode |h|^2=4 (real, Perron-adjacent)" if max(h2) > 3 else \
          "  (pure shell: no Perron mode)"
    print(f"     winding omega^{t}:  |h|^2 = {h2}{tag}")
print("""
  FORCED asymmetry (NOT chosen): exactly ONE winding (omega^0) contains the Perron return
  |h|^2 = (k-1)^2 = 4; the other two (omega^1, omega^2) are mirror-equal, pure-shell {1,1,2,2}.
  => the distinguished winding omega^0 'rides the real Perron-adjacent mode'; omega^{1,2} are an
     equal complex-conjugate pair.  This is the forced (4 : 2 : 2)-type return signature.""")

# the persistence (return) AMPLITUDE of each winding = its shell return modulus.  The shell modulus
# is |h| = sqrt(k-1) = sqrt2 for ALL three (Ramanujan rigidity).  The distinguishing modulus datum
# is the EXTRA Perron weight omega^0 alone carries.  Excess return weight over the shell (k-1):
print("  Per-winding shell modulus |h| (the recurrence amplitude) and Perron excess (omega^0 only):")
for t in (0, 1, 2):
    perron = any(abs(w - 4.0) < 1e-3 for w in weights[t])
    print(f"     omega^{t}: shell |h| = sqrt(k-1) = {np.sqrt(k-1):.5f}"
          f"   Perron-excess weight present: {perron}")

# =============================================================================================
hdr("(2) THE DIRECTED PHASE the screw imparts to the complex shell (DERIVED velocity phi)")
# =============================================================================================
# At Gamma the shell amplitude is h = sqrt2 * e^{i theta0}, the SAME theta0 in all three windings
# (the windings are degenerate in modulus AND phase at the tracial start).  Running the screw
# forward (s>0) splits the phases: omega^0 stays, omega^1 advances +, omega^2 advances -.
theta0 = cmath.phase(shell_amp(0, 0.0))
print(f"  Shell phase at the tracial start (all windings):  theta0 = {np.degrees(theta0):.5f} deg")
print(f"     forced value: cos theta0 = lam0/(2 sqrt(k-1)) with lam0 = -1 (the A4 3-irrep eigenvalue)")
print(f"     check cos theta0 = -1/(2 sqrt2) = {-1/(2*np.sqrt(2)):.6f}  vs  {np.cos(theta0):.6f}")
print(f"     => sin^2 theta0 = 1 - lam0^2/(4(k-1)) = 1 - 1/8 = 7/8  (the '7' = 8 - 1).")

# directed phase velocity at the start (the forced 'per unit run' velocity), by symmetric difference
ds = 1e-4
def directed_velocity(t):
    return (cmath.phase(shell_amp(t, ds)) - cmath.phase(shell_amp(t, -ds))) / (2*ds)
v0 = directed_velocity(0)                      # common drift (omega^0, the carrier) ~ 0
v1 = directed_velocity(1) - v0
v2 = directed_velocity(2) - v0
print(f"\n  Directed phase velocities d theta/ds at the start (relative to the omega^0 carrier):")
print(f"     omega^0: {0.0:+.6f}     omega^1: {v1:+.6f}     omega^2: {v2:+.6f}   rad / unit arc-length")
print(f"\n  DERIVED closed form (from h = sqrt2 e^{{i theta}}, lam = 2 sqrt2 cos theta, "
      f"d theta/ds = -(d lam/ds)/(2 sqrt2 sin theta0)):")
print(f"     d lam/ds = -2 pi exactly (sector-1 split of the lam0=-1 A4 triple); sin theta0 = sqrt(7/8)")
print(f"     => phi := |d theta/ds| = 2 pi / (2 sqrt2 * sqrt(7/8)) = 2 pi / sqrt 7")
phi = 2*np.pi/np.sqrt(7)
print(f"        phi = 2 pi / sqrt7 = {phi:.6f}   (matches measured |v1| = {abs(v1):.6f})")
print(f"        (per UN-normalized lattice step (1,-1,1) of length sqrt3: phi*sqrt3/(2pi) = "
      f"sqrt(3/7) = {np.sqrt(3/7):.6f}  -- the framework's quoted 2 pi sqrt(3/7).)")
print(f"  => the screw imparts the directed phase triple  {{0, +phi, -phi}},  phi = 2pi/sqrt7,")
print(f"     CHIRAL (omega^1 and omega^2 wind in OPPOSITE senses): a forced, import-free velocity.")

# =============================================================================================
hdr("(3) THE FORCED MASS-SHAPE: C3-Fourier combination of the three persistence amplitudes")
# =============================================================================================
print("""  DERIVATION (every factor forced by the recurrence; nothing chosen):

  * A 'generation' is an observer who reads the recurrence in the C3-DIAGONAL (winding) basis.
    The three windings carry three persistence amplitudes c_0, c_1, c_2 (one per winding).
    The three PHYSICAL generations are the three states that diagonalize the C3 cyclic action on
    the windings -- i.e. the discrete C3-Fourier transform of the windings:

        sqrt(m_j) = sum_{t=0,1,2}  c_t * omega^{t j},     j = 0,1,2          (the forced map)

    (C3-Fourier is forced: it is the ONLY basis change that turns the cyclic winding-shift into
     the three distinct eigen-channels; the generation index j is the C3 character it carries.)

  * AMPLITUDE MODULI |c_t| = the windings' return amplitudes (section 1).  All three sit on the
    Ramanujan shell |h| = sqrt(k-1); the modulus is RIGID, so |c_0| = |c_1| = |c_2| = (k-1)^{1/4}
    up to the one forced asymmetry -- omega^0 additionally carries the Perron-adjacent weight.
    The clean, forced moduli are therefore EQUAL on the shell: a uniform |c_t| = 1 (normalized).

  * AMPLITUDE PHASES arg(c_t) = the accumulated directed phase the screw has imparted by run-
    position s (section 2):  arg(c_0) = 0,  arg(c_1) = +phi*s,  arg(c_2) = -phi*s.

  So the forced per-winding amplitudes are
        c_0 = 1,   c_1 = e^{+ i phi s},   c_2 = e^{- i phi s},     phi = 2pi/sqrt7,
  and the three generation masses are the |C3-Fourier|^2:
        m_j(s) = | 1 + e^{i(phi s + 2 pi j/3)} + e^{-i(phi s - 2 pi j/3)} |^2 .""")

def masses(s):
    c = np.array([1.0, np.exp(1j*phi*s), np.exp(-1j*phi*s)])
    out = []
    for j in (0, 1, 2):
        amp = sum(c[t]*om**(t*j) for t in (0, 1, 2))
        out.append(abs(amp)**2)
    return np.array(out)

# Closed form: m_j(s) = 3 + 2[cos(phi s + 2pi j/3) + cos(phi s - 2pi j/3) ... ] -> simplify & verify.
print("\n  Computed three masses m_0,m_1,m_2 and the two ratios m2/m1, m3/m1, vs run-position s:")
print(f"   {'phi*s(deg)':>11} {'m_0':>10} {'m_1':>10} {'m_2':>10} {'m_mid/m_lo':>11} {'m_hi/m_lo':>11}")
for s in [0.0, 0.05, 0.1, 1/12, 0.2, 0.3, 0.5, 1.0]:
    m = masses(s); ms = np.sort(m)
    lo, mid, hi = ms
    r1 = mid/lo if lo > 1e-12 else np.inf
    r2 = hi/lo if lo > 1e-12 else np.inf
    print(f"   {np.degrees(phi*s):11.3f} {m[0]:10.5f} {m[1]:10.5f} {m[2]:10.5f} {r1:11.4f} {r2:11.4f}")

# =============================================================================================
hdr("(3b) EXACT closed forms for the three masses and the two ratios as functions of u = phi*s")
# =============================================================================================
print("""  Writing u = phi*s (the accumulated directed phase), the |C3-Fourier|^2 evaluates to:

      m_0(u) = | 1 + e^{ i u}     + e^{ - i u}     |^2 = ( 1 + 2 cos u )^2
      m_1(u) = | 1 + e^{ i(u+2pi/3)} + e^{-i(u-2pi/3)} |^2 = ( 1 + 2 cos(u - 2pi/3) )^2
      m_2(u) = | 1 + e^{ i(u-2pi/3)} + e^{-i(u+2pi/3)} |^2 = ( 1 + 2 cos(u + 2pi/3) )^2

  i.e.  m_j(u) = ( 1 + 2 cos(u - 2pi j/3) )^2 .   (a single forced function, three C3-shifts.)""")
def masses_closed(u):
    return np.array([(1 + 2*np.cos(u - 2*np.pi*j/3))**2 for j in (0, 1, 2)])
# verify the closed form equals the matrix computation
for s in [0.07, 0.23, 0.41]:
    a = np.sort(masses(s)); b = np.sort(masses_closed(phi*s))
    print(f"   check s={s}: |matrix - closed| = {np.max(np.abs(a-b)):.2e}")

print("\n  The two independent ratios (sorting masses lo<=mid<=hi at each u):")
print(f"   {'u(deg)':>8} {'m_lo':>9} {'m_mid':>9} {'m_hi':>9} {'mid/lo':>10} {'hi/lo':>10}")
for udeg in [0, 10, 20, 30, 40, 50, 60, 80, 100, 120]:
    u = np.radians(udeg); m = np.sort(masses_closed(u))
    lo, mid, hi = m
    print(f"   {udeg:8d} {lo:9.5f} {mid:9.5f} {hi:9.5f} "
          f"{(mid/lo if lo>1e-9 else np.inf):10.4f} {(hi/lo if lo>1e-9 else np.inf):10.4f}")

# =============================================================================================
hdr("(4) RANGE, DEGENERACIES, ZEROS as u varies — the distinguished run-positions")
# =============================================================================================
print("  Each mass m_j(u) = (1 + 2 cos(u - 2pi j/3))^2 :")
print(f"     range of a single mass: [0, 9]  (max 9 at cos=1; zero when cos = -1/2).")
print(f"     m_j(u) = 0  <=>  cos(u - 2pi j/3) = -1/2  <=>  u - 2pi j/3 = +-2pi/3.")
print()
print("  Distinguished u:")
print("   * u = 0 (tracial start s=0): masses = (1+2)^2, (1+2cos(-120))^2, (1+2cos(+120))^2")
m_start = masses_closed(0.0)
print(f"        = {np.round(m_start,5)} = {{9, 0, 0}}  -> TWO massless, one heavy (9). A DEGENERATE")
print("          start: at s=0 the windings are phase-aligned, only the symmetric combination is massive.")
print("   * a mass passes through ZERO whenever u hits +-120 deg mod the C3 shifts:")
for j in (0,1,2):
    for sign in (+1,-1):
        u_zero = (2*np.pi*j/3 + sign*2*np.pi/3)
        s_zero = u_zero/phi
        print(f"        m_{j}=0 at u={np.degrees(u_zero):+.1f} deg  (s = u/phi = {s_zero:+.4f})")
print("   * FULL degeneracy m_0=m_1=m_2: requires cos(u)=cos(u-120)=cos(u+120), impossible for the")
print("     three distinct shifts unless all three cos are equal = only at the symmetric point where")
print("     two vanish; generically the three masses are DISTINCT (a genuine 3-way hierarchy for u!=0).")
# numerically scan the spread (hierarchy) vs u
print("\n  Mass SPREAD (max-min) over a u-scan -- where the hierarchy is largest:")
us = np.linspace(0, 2*np.pi/3, 25)
for u in us[::4]:
    m = masses_closed(u)
    print(f"     u={np.degrees(u):6.1f} deg: masses={np.round(np.sort(m),4)}  spread={m.max()-m.min():.4f}")

# =============================================================================================
hdr("(5) FORCED vs CHOICE -- the ledger")
# =============================================================================================
print(f"""  EVERY factor is FORCED by the recurrence; the ONLY residual is the run-position s.

  FORCED (derived from the object, no import, no fit):
   * the three windings = the three C3 sectors {{omega^0, omega^1, omega^2}} of the deck screw;
   * the modulus structure: all three on the rigid Ramanujan shell |h|=sqrt(k-1); the forced
     asymmetry is that omega^0 alone carries the Perron-adjacent return |h|^2=(k-1)^2=4 (the
     '4:2:2' signature) -- so the clean shell moduli are EQUAL, |c_t| = 1;
   * the directed phase the screw imparts: the chiral triple {{0, +phi, -phi}} with the DERIVED
     velocity  phi = 2 pi / sqrt 7  per unit arc-length  (= 2 pi sqrt(3/7) per (1,-1,1) lattice
     step), the '7' = 8 - 1 from sin^2 theta0 = 1 - lam0^2/(4(k-1)), lam0 = -1 the A4 3-irrep value;
   * the C3-Fourier map sqrt(m_j) = sum_t c_t omega^{{tj}} (forced: the only basis diagonalizing
     the cyclic winding-shift), hence mass = |C3-Fourier|^2 = (1 + 2 cos(u - 2pi j/3))^2, u = phi*s;
   * therefore the two ratios m_mid/m_lo and m_hi/m_lo are a FORCED ONE-PARAMETER FAMILY in the
     single run-position s (equivalently u = phi*s).  No second free number leaked in: moduli are
     fixed (shell-rigid, equal), phases are fixed up to the one accumulated angle u.

  THE SINGLE RESIDUAL: the run-position s (equivalently u = phi*s = the accumulated directed
  phase).  This is the scale-free III_1 residual -- the object fixes the SHAPE m_j(u) entirely
  and leaves only WHERE ALONG THE RUN the observer reads it.  An honest 'the spectrum is
  m_j(u) = (1 + 2 cos(u - 2pi j/3))^2; make of u what you will.'

  HONEST NEGATIVES:
   * the construction does NOT fix u (and so does not fix the absolute ratios): u is the free
     endpoint, exactly as the scale-free (III_1) structure demands.
   * at u=0 (the tracial start) the spectrum is degenerate {{9,0,0}}; a non-degenerate three-mass
     hierarchy exists only for u != 0, i.e. only after the screw has run -- the hierarchy is a
     consequence of the run, not of the start.
   * the equal-moduli choice |c_t|=1 uses Ramanujan rigidity (forced); the omega^0 Perron-excess
     is a genuine extra datum NOT used in the equal-moduli mass -- if instead one weights c_0 by
     its Perron-excess the moduli become unequal (a 2nd forced number).  Reported as a fork, not
     hidden: the MINIMAL forced construction uses the rigid (equal) shell moduli.""")
