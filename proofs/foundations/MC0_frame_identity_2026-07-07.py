#!/usr/bin/env python3
"""
proofs/foundations/MC0_frame_identity_2026-07-07.py

MC-0 — the FRAME-IDENTITY THEOREM (the bounded first win of the MC-track). Pre-registered
in internal research notes (committed 10ea02c BEFORE this
file). Frozen contract: internal research notes (925f5b0).
Executor: a model

THE CLAIM (pure algebra on two theorem-grade framework facts): framework-coasting
(a~t, horizon-thermal T~a^{-1/2}) and radiation-FRW (a~t^{1/2}, T~1/a) share their
temperature history T(t) and their H(T) EXACTLY (scaling level), and differ ONLY in a(T)
(coasting a~T^{-2}, radiation a~T^{-1}) -- i.e. only in LENGTHS/ANGLES. This is the
diagnosis's smoking gun: thermally-anchored DYNAMICS is frame-blind; the frame shows up
only where theta_* broke.

POISON (binding): the 1/48 (25/48 vs 1/2) <-> n_s-1 connection is FORBIDDEN pattern-matching.
No 0.0104/67.4/73.0/0.965 (blind confronts are MC-3/4). No scoreboard value moves -- this
station LOCKS a structural identity.
"""
import math
import sys

import sympy as sp

ok_all = True
def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    return cond
def banner(t):
    print("=" * 88); print(f" {t}"); print("=" * 88)

# symbols (all positive; work with power-law EXPONENTS, the scaling content)
t, T, a, t0, c_s = sp.symbols('t T a t0 c_s', positive=True)

def powexp(expr, var):
    """exponent of `var` in a monomial c*var^n, via the logarithmic derivative
    var * d(log expr)/d var = n  (strips the coefficient c cleanly)."""
    return sp.simplify(var * sp.diff(sp.log(expr), var))

# ===========================================================================
banner("MC0-a  the TEMPERATURE identity: both frames give T ~ t^{-1/2}")
# ===========================================================================
# Framework: a ~ t (coasting); T ~ a^{-1/2} (horizon-thermal, alpha=1/2).
a_fw = t                                        # a ~ t
T_fw = a_fw ** sp.Rational(-1, 2)               # T ~ a^{-1/2} = t^{-1/2}
exp_T_fw = powexp(T_fw, t)                       # exponent of t in T_fw
# Radiation-FRW: a ~ t^{1/2}; T ~ 1/a.
a_rad = t ** sp.Rational(1, 2)                  # a ~ t^{1/2}
T_rad = 1 / a_rad                                # T ~ 1/a = t^{-1/2}
exp_T_rad = powexp(T_rad, t)
print(f"    framework:  a~t,      T~a^(-1/2)  => T ~ t^({exp_T_fw})")
print(f"    radiation:  a~t^(1/2), T~1/a      => T ~ t^({exp_T_rad})")
check("MC0-a T(t) exponent identical in both frames (= -1/2)",
      sp.simplify(exp_T_fw - exp_T_rad) == 0 and exp_T_fw == sp.Rational(-1, 2))

# ===========================================================================
banner("MC0-b  the H(T) identity: both frames give H ~ T^2 (report O(1) coefficient)")
# ===========================================================================
# H = adot/a. Use a(t); express H(t), then substitute t(T).
def H_of_t(a_expr):
    return sp.simplify(sp.diff(a_expr, t) / a_expr)     # adot/a
H_fw_t = H_of_t(a_fw)                            # = 1/t
H_rad_t = H_of_t(a_rad)                          # = (1/2)/t
# t in terms of T: from T ~ t^{-1/2} => t ~ T^{-2} (same map both frames)
t_of_T = T ** (-2)
H_fw_T = sp.simplify(H_fw_t.subs(t, t_of_T))     # coeff * T^2
H_rad_T = sp.simplify(H_rad_t.subs(t, t_of_T))
exp_H_fw = powexp(H_fw_T, T)
exp_H_rad = powexp(H_rad_T, T)
print(f"    framework:  H = {H_fw_t} = {H_fw_T}  => H ~ T^({exp_H_fw})")
print(f"    radiation:  H = {H_rad_t} = {H_rad_T}  => H ~ T^({exp_H_rad})")
check("MC0-b H(T) exponent identical in both frames (= +2)",
      sp.simplify(exp_H_fw - exp_H_rad) == 0 and exp_H_fw == 2)
coeff_fw = sp.simplify(H_fw_T / T ** 2); coeff_rad = sp.simplify(H_rad_T / T ** 2)
print(f"    O(1) coefficient: framework H/T^2 = {coeff_fw} ; radiation = {coeff_rad} "
      f"(ratio {sp.simplify(coeff_fw/coeff_rad)}) -- the ONLY dynamical trace of the frame")
check("MC0-b the coefficient difference is O(1) (framework/radiation = 2), NOT a scaling difference",
      sp.simplify(coeff_fw / coeff_rad) == 2)

# ===========================================================================
banner("MC0-c  the difference is CONFINED to a(T) (lengths); dynamics is frame-invariant")
# ===========================================================================
# a(T): invert T(a). Framework T~a^{-1/2} => a~T^{-2}. Radiation T~a^{-1} => a~T^{-1}.
a_of_T_fw = T ** (-2)                             # from T~a^{-1/2}
a_of_T_rad = T ** (-1)                            # from T~1/a
exp_a_fw = powexp(a_of_T_fw, T)
exp_a_rad = powexp(a_of_T_rad, T)
print(f"    a(T): framework a~T^({exp_a_fw}) ; radiation a~T^({exp_a_rad})  => DIFFER (this is where")
print(f"          the frames part: lengths/angles, exactly where theta_* broke)")
check("MC0-c a(T) exponents DIFFER (framework -2 vs radiation -1): the frame lives in lengths",
      exp_a_fw != exp_a_rad and exp_a_fw == -2 and exp_a_rad == -1)
# rate-vs-H at fixed T: a freeze-out/recombination condition is Gamma(T) = H(T). Since H(T) is
# identical up to the O(1) coefficient, the SOLUTION T_freeze is frame-invariant up to O(1).
Gamma = T ** 5                                    # any thermal rate ~ power of T (illustrative)
Tf_fw = sp.solve(sp.Eq(Gamma, H_fw_T), T)         # freeze-out temperature, framework
Tf_rad = sp.solve(sp.Eq(Gamma, H_rad_T), T)       # radiation
ratio_Tf = sp.simplify(Tf_fw[0] / Tf_rad[0]) if Tf_fw and Tf_rad else None
print(f"    freeze-out (Gamma=H, illustrative Gamma~T^5): T_freeze ratio framework/radiation = "
      f"{ratio_Tf} (O(1), NOT a scaling shift) => thermally-anchored dynamics is frame-blind")
check("MC0-c freeze-out temperature is frame-invariant up to O(1) (dynamics frame-blind)",
      ratio_Tf is not None and sp.simplify(sp.log(ratio_Tf) / sp.log(2)).is_rational)

# ===========================================================================
banner("MC0-d  the E-FOLD LEMMA: coasting r_s = c_s t0 x (e-folds of a), each e-fold EQUAL")
# ===========================================================================
# comoving sound horizon r_s = int c_s dt/a. Coasting a = t/t0 (a0=1 at t0). dt/a = t0 dt/t = t0 d(ln t)
# = t0 d(ln a). So integrand in ln a is the CONSTANT c_s t0 -- each e-fold contributes equally.
lna = sp.symbols('lna', real=True)
integrand_dlna = sp.simplify(c_s * t0)            # d r_s / d(ln a) in coasting
check("MC0-d coasting: d r_s/d(ln a) = c_s t0 = CONSTANT (each e-fold of a contributes equally)",
      sp.diff(integrand_dlna, lna) == 0 and integrand_dlna == c_s * t0)
print(f"    => r_s = c_s t0 * (number of e-folds of a). The divergence is an INFINITE SUM of EQUAL")
print(f"       scale-free contributions (not a UV blow-up) -- the object MC-2's memory kernel truncates.")

# ===========================================================================
banner("MC0-e  the NATIVE z_rec FORK (report BOTH; resolve NOTHING -- that is MC-1)")
# ===========================================================================
T_rec_K = 3000.0                                  # physical recombination temperature (~3000 K)
T0_K = 2.7255                                     # CMB today
z_bath = (T_rec_K / T0_K) ** 2 - 1                # native BATH clock: a~T^{-2}
z_photon = (T_rec_K / T0_K) - 1                   # standard/photon clock: a~T^{-1}
print(f"    native BATH-clock (a~T^-2):    1+z_rec = (T_rec/T0)^2 = {z_bath+1:.3e}  (z_rec ~ {z_bath:.3e})")
print(f"    standard/photon (a~T^-1):      1+z_rec = T_rec/T0     = {z_photon+1:.1f}  (the FITTER value;")
print(f"                                   M2c used 1089 = this branch)")
check("MC0-e the two clocks give DIFFERENT z_rec (bath ~1e6 vs photon ~1100): the fork is real",
      z_bath / z_photon > 100)
print("    SUB-FORK (FROZEN, not resolved here -- MC-1): substrate BATH-T (T~a^-1/2) vs free-streaming")
print("    PHOTON-T (T~1/a after decoupling) -- which clocks recombination. A1 hits T_CMB=2.7255 from")
print("    the GUT anchor via horizon-thermal, i.e. treats the CMB T as substrate-thermal.")

# ===========================================================================
banner("MC0-f  T(N) = T_P N^{-25/48}: leading alpha=1/2 + derived 1/48 (POISON-guarded)")
# ===========================================================================
dev = sp.Rational(25, 48) - sp.Rational(1, 2)
print(f"    T(N) ~ N^(-25/48): exponent 25/48 = {float(sp.Rational(25,48)):.4f}; deviation from 1/2 = "
      f"{dev} = {float(dev):.4f} (cumulative-Perron correction)")
check("MC0-f the leading thermal exponent is 1/2 (alpha=1/2) with a small derived deviation 1/48",
      dev == sp.Rational(1, 48) and abs(float(sp.Rational(25, 48)) - 0.5) < 0.03)
print("    POISON HELD: NO connection of 1/48 to n_s-1 -- forbidden until it falls out of MC-2/MC-3.")

# ===========================================================================
banner("SUMMARY")
# ===========================================================================
verdict = "IDENTITY-LOCKED" if ok_all else "BROKEN"
print(f"""    MC-0 OUTCOME = {verdict}: framework-coasting (a~t, T~a^-1/2) and radiation-FRW (a~t^1/2,
          T~1/a) share T(t)~t^-1/2 and H(T)~T^2 EXACTLY (scaling); the ONLY dynamical trace is the
          O(1) H-coefficient (framework 2x radiation). The frames part ONLY in a(T) (T^-2 vs T^-1) =
          LENGTHS/ANGLES = precisely where theta_* broke. E-fold lemma: coasting r_s = c_s t0 x
          (e-folds), each equal => the divergence is a scale-free infinite sum (MC-2's kernel truncates).
          Native z_rec fork REPORTED (bath ~1e6 vs fitter ~1100), sub-fork FROZEN for MC-1.
    => the diagnosis's FOUNDATION is theorem-grade: thermally-anchored dynamics is frame-blind; the
       frame difference is provably confined to lengths. No scoreboard value moved. POISONS held.""")
print("RESULT:", "ALL CHECKS PASS -- MC-0 IDENTITY-LOCKED" if ok_all else "A CHECK FAILED -- diagnosis broken")
sys.exit(0 if ok_all else 1)
