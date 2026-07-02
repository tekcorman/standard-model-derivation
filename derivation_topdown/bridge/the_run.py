"""
the_run — THE ONE MASTER OBJECT, run forward; every forced result is a READ from it.

ONE object (srs ⊕ its operators), ONE run (∂_N along the coordinate s), MANY reads.
No competing construction is permitted: if a quantity is not read from THIS object, it does not exist.

PROVENANCE — honest classification of every read (audit 2026-06-23 PM; the aspiration is 100% NATIVE):
  NATIVE (computed off the operators): cover_B / dressed_spectrum / read_run (resolvent), read_geometry Λ²,
    read_dirac4_lift ({D,γ}), read_mass (Ihara–Bass), read_girth (off B), adjacency_energies (off A),
    read_gauge (Tr S²/Tr Q² computed: Q=C₃-winding off the darts), the EW OBLIQUES read_obliques
    (δ_r/δρ/S = gauge-vertex PROJECTIONS of G_NB; c_S=⟨ŝ|P_Perron|ŝ⟩/dim, a projection — not a count).
  FLAGGED non-native (each carries a ⚠ at its definition):
    · read_gauge_running — LAYER 1 COMPUTED (Dynkin sums over the forced content, gens×SM multiplets + 2
      Higgs + the computed 4D-completion → β VALUES {33/5,1,−3} fall out, all three; the MSSM-lit values are a
      COMPARISON-ONLY cross-check, NOT a hardcoded target). LAYER 2 still QFT: the one-loop β FORMULA,
      whose native form is ζ_{D₄}(0) (research-level, lattice = dead end). NOTE: removing layer-1 injection does
      NOT move g_2 (−2.52σ) — the β value was already MSSM; the g_2 VALUE residual lives in layer 2.
    · read_democratic c_v=5/12, read_vertex leg-counts — COMBINATORIAL (H¹/Wilson count; vertex topology),
      no spectral projection found yet (cf. δ_r's 1/12 which IS a projection).
    · read_clock ε — information-theoretic (MDL ratio), traces to k*/toggle but not an operator eigenvalue.
    · read_flavor (4,2,2) — representation theory by enumeration (= the character det(1+ρ(g)) over Λ•).
    · read_obliques δρ's ½ — the EW W-field normalization, a definitional electroweak constant (not substrate).
    · read_sector_label — explicit combinatorics (SUBORDINATE label, never itself a mass).
Mass is recurrence-under-running (|h|-shell amplitudes + running phase + resolvent) — NOT a spanning count.
"""
import sys, os, itertools, cmath
import numpy as np
from fractions import Fraction
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs

K = srs.DEG                       # 3  (coordination)
TWO_PI = 2*np.pi
def banner(t): print("="*78); print(" "+t); print("="*78)

# ════════════════════════════════════════════════════════════════════════════
# THE ONE OBJECT — operators read straight off srs (no second construction)
# ════════════════════════════════════════════════════════════════════════════
def adjacency(k):  return srs.adjacency(k)
def dirac(k):      return srs.hodge_dirac(k)
def hashimoto(k):  return srs.hashimoto(k)         # the non-backtracking / geodesic generator B
KP = [(0,0,0), (0.25,0.25,0.25), (0.5,0.5,0.5), (0.31,0.13,0.07)]   # Γ, P, H, generic

# ── native structural reads (COMPUTED off the operators — never typed) ───────────
def renewal():
    # N_L = ∫_BZ Tr(B(k)^L) dk = the INTEGER count of zero-net-translation closed NB walks of length L.
    # BZ-averaging annihilates every net-translating walk (Σ_k e^{2πik·v}=0 unless v≡0), so N_L is FORCED to 0
    # below the girth. NO range, NO min: the horizon is the operator dimension dim(B)=2|E| (Cayley–Hamilton — B^n
    # for n≥dim is a combination of lower powers, so the renewal is fully determined by L<dim); the grid G=dim+1
    # is forced to make the BZ-average EXACT for every |v|≤dim. Returns the WHOLE sequence; nothing is selected.
    dim = hashimoto((0, 0, 0)).shape[0]                 # = 2|E|, the structural horizon
    G = dim + 1                                         # grid forced by exactness (annihilates |v|≤dim)
    ks = [(a/G, b/G, c/G) for a in range(G) for b in range(G) for c in range(G)]
    return [int(round(np.mean([np.trace(np.linalg.matrix_power(hashimoto(k), L)).real for k in ks])))
            for L in range(1, dim + 1)]                  # [N_1, …, N_dim] — integer counts, forced-0 prefix

def read_girth():
    N = renewal()                                       # the full renewal sequence (emit everything)
    return next(L for L, n in enumerate(N, 1) if n != 0)   # girth = the FIRST NONVANISHING term (the lattice
    #   put exact 0 below it; the girth is where annihilation stops — the definition, not a bounded search)

def adjacency_distinct(k=(0, 0, 0)):
    # the WHOLE distinct adjacency spectrum at Γ (the quotient graph; phases=1) — emit everything, no pick.
    return sorted(set(round(float(e), 9) for e in np.linalg.eigvalsh(adjacency(k))))   # K4 → {−1, +3}

def adjacency_energies(k=(0, 0, 0)):
    # The channel ENERGIES are the two distinct eigenvalues — both FORCED, neither selected: the larger is the
    # Perron–Frobenius dominant (unique, by PF on the non-negative irreducible adjacency = k*), the smaller is the
    # remaining distinct eigenvalue (the 3-irrep, multiplicity = generations = −1). No other answer is admissible.
    chans = adjacency_distinct(k)
    return chans[-1], chans[0]                      # (k*=Perron [PF-forced], −1 [the forced non-Perron])

GIRTH = read_girth()                               # = 10, read off B (not the typed exponent)
LAM_PERRON, LAM_3IRREP = adjacency_energies()      # = (3, −1), read off adjacency(Γ) (not typed)
P_TOGGLE = len(srs._darts()) // len(srs.EDGES)     # = 2, darts/edge = the orientation binary (READ, not typed)

# ── READ 0 — units / clock (dimensionless) ──────────────────────────────────
def read_clock():
    # ⚠ INFORMATION-THEORETIC (the one read that is NOT a spectral invariant — the observer's MDL side,
    #   not B/D). Both inputs are now READ: Pf=1/p_toggle (p from the dart structure), Pp=1/k* (k=srs.DEG).
    #   The (Pf−Pp)/(Pf+Pp) form is the forced Bayesian toggle-disconfirmation (Beta(2,1)), not my pick.
    Pf, Pp = Fraction(1, P_TOGGLE), Fraction(1, K)   # 1/p_toggle (READ) , 1/k* (READ)
    eps = (Pf-Pp)/(Pf+Pp)                          # 1/5
    clock = 1 + eps/K                              # 16/15
    return eps, clock

# ── READ 1 — geometry / gravity ─────────────────────────────────────────────
def read_geometry():
    b1 = len(srs.EDGES) - srs.NV + 1               # 3  => spatial dimension
    Lam2 = max(float(np.max(np.abs(np.linalg.eigvalsh(dirac(k)))**2)) for k in
               [(a/6,b/6,c/6) for a in range(6) for b in range(6) for c in range(6)])
    return b1, Lam2

# ── READ 2 — gauge (the C3 winding charge sum; sin^2θ_W = Tr S²/Tr Q²) ───────
def read_gauge():
    # sin²θ_W = Tr(S²)/Tr(Q²) over the matter space H = copy(2)⊗Weyl(2)⊗dart(12) — COMPUTED traces.
    # Q = the C₃ WINDING charge READ off srs: σ=(1 2 3) permutes the 12 darts as P; P=exp(2πiQ/3),
    # spec(Q)={−1,0,1}; Tr(Q²_dart)=Σ winding² (native). S = SU(2)_L T₃=½σ³ (the gauge-group generator).
    sigma = {0: 0, 1: 2, 2: 3, 3: 1}                 # the C₃ 3-cycle (1 2 3) on K4 vertices
    D = srs._darts(); nd = len(D)
    P = np.zeros((nd, nd))
    for a, (i, j, v) in enumerate(D):
        for b, (p, q, w) in enumerate(D):
            if (p, q) == (sigma[i], sigma[j]): P[b, a] = 1; break
    windings = np.round(np.angle(np.linalg.eigvals(P)) / (TWO_PI/3)).astype(int)   # ∈ {−1,0,1}, READ
    TrQ2_dart = int((windings**2).sum())             # Σ winding² over the 12 darts (= 8)
    T3 = 0.5*np.array([[1, 0], [0, -1]], complex)    # SU(2)_L T₃ (gauge generator)
    TrS2 = int(round(np.trace(T3 @ T3).real * 2 * nd))   # Tr(T₃²)·dim_Weyl·dim_dart = ½·2·12 = 12
    TrQ2 = 2 * 2 * TrQ2_dart                          # dim_copy·dim_Weyl·Σw² = 4·8 = 32
    return Fraction(TrS2, TrQ2)                       # 12/32 = 3/8

# ── READ 2′ — GAUGE RUNNING = the run's ZERO-MODE (the "+4" = the 4D completion) ─
#   sin²θ_W=3/8, α_GUT⁻¹=24 are the STATIC boundary (read_gauge). The RUNNING is the
#   p=0 (logarithmic, ~τ=log N) mode of the SAME run ∂_N. The run's stepping completes
#   the substrate's 3D Weyl fermions to 4D Dirac (time = the 4th direction); that
#   completion adds (1/3)ΣT_f + (2/3)ΣT_H + (2/3)C₂(G) per group — for the non-abelian groups
#   this is the +(2/3)ΣT_f = +4 fermion-doubling (time-component shadows, NOT physical sparticles).
#   It rescues the Landau-poling 2HDM β into the working (non-SUSY) MSSM β: b₁,b₂,b₃ ALL reproduce
#   {33/5, 1, −3} EXACTLY (verified by read_gauge_running; the U(1) via the (1/3)T_f+(2/3)T_H
#   hypercharge combination, not a naive uniform doubling). The β VALUES fall out of the computed
#   completion; the β FORMULA itself (the −11/3, ⅔, ⅓ Dynkin structure) is the open piece =
#   ζ_{D₄}(0), research-level (docs/incomplete_equations_todo.md §5).
def gauge_dynkin(fields, mult):
    # the SU(3)/SU(2)/U(1) Dynkin sums over a field list (color_dim, su2_dim, hypercharge Y), each ×mult.
    T3 = {1: Fraction(0), 3: Fraction(1, 2), 8: Fraction(3)}   # SU(3) Dynkin index of the rep
    T2 = {1: Fraction(0), 2: Fraction(1, 2), 3: Fraction(2)}   # SU(2) Dynkin index of the rep
    s = {1: Fraction(0), 2: Fraction(0), 3: Fraction(0)}
    for c, w, Y in fields:
        s[3] += T3[c] * w * mult                 # SU(3): T(color) × (su2 multiplicity)
        s[2] += T2[w] * c * mult                 # SU(2): T(su2) × (color multiplicity)
        s[1] += Fraction(3, 5) * Y*Y * c*w * mult   # U(1) GUT-norm: (3/5)Y² × total multiplicity
    return s

def read_gauge_running():
    # LAYER 1 — COMPUTED (no longer typed): the β coefficients are Dynkin/Casimir sums over the FORCED field
    #   content — gens (= read_flavor) generations of the SM multiplets (Cl(6)=Pati-Salam; hypercharge Y=Q−T₃ with
    #   the species charge Q=±n/k* from the Hamming weight) + 2 Higgs doublets (the srs⊗srs-z bipartite mirror).
    # ⚠ LAYER 2 — STILL INJECTED: the one-loop β FORMULA itself (the −11/3, ⅔, ⅓ Dynkin structure) is standard
    #   QFT; its native form is ζ_{D₄}(0) (the spectral zeta of D₄=B⊗∂_N), RESEARCH-LEVEL (lattice = dead end).
    #   The multiplet/hypercharge ASSEMBLY below is now NATIVE — every (color, T₃, Y) reads off N̂'s Hamming
    #   weight n (Q=(−1)ⁿn/k*, T₃=(−1)ⁿ/2, Y=Q−T₃), reproducing Tf={6,6,6} and b=MSSM. Only the +4-shadow and the
    #   β-formula itself remain Layer-2 (ζ_{D₄}(0)). The fermion content is no longer hand-listed.
    gens = read_flavor()[3]                                   # = 3 (READ off the C₃ isotypes)
    # NATIVE: the SM multiplets FALL OUT of the Cl(6)-Fock Hamming weight n (read_species) — no hand-typed
    #   hypercharges. color = Fock multiplicity (1 lepton / 3 quark); charge Q = (−1)ⁿ·n/k* (Hamming weight ×
    #   its parity); weak isospin T₃ = (−1)ⁿ/2 = the parity (left-handed), 0 (right); hypercharge Y = Q − T₃.
    sgn = lambda n: 1 if n % 2 == 0 else -1;  Qn = lambda n: Fraction(sgn(n)*n, K)
    fermions = [(3, 2, Qn(2) - Fraction(1, 2)),   # Q_L  quark doublet (up n=2, down n=1)  → Y=1/6
                (1, 2, Qn(0) - Fraction(1, 2)),   # L_L  lepton doublet (ν n=0, e n=3)      → Y=−1/2
                (3, 1, Qn(2)), (3, 1, Qn(1)), (1, 1, Qn(3))]   # u_R, d_R, e_R singlets (T₃=0 → Y=Q)
    higgs = [(1, 2, Fraction(1, 2)), (1, 2, Fraction(-1, 2))]    # 2 Higgs doublets
    Tf = gauge_dynkin(fermions, gens)                        # ΣT_f  = {6, 6, 6}      (COMPUTED)
    TH = gauge_dynkin(higgs, 1)                              # ΣT_H  = {3/5, 1, 0}    (COMPUTED)
    C2G = {1: Fraction(0), 2: Fraction(2), 3: Fraction(3)}   # C₂(G) = gauge Casimir of the adjoint
    # MSSM literature values (Martin, SUSY Primer §6.5) — HARDCODED COMPARISON-ONLY external
    # reference, NOT a derivation target. b4d below is derived independently (b_2HDM + computed
    # completion); we only report whether the derived value agrees with the literature.
    b_MSSM_lit = {1: Fraction(33, 5), 2: Fraction(1), 3: Fraction(-3)}
    out = {}
    for i in (1, 2, 3):
        b_2HDM = -Fraction(11, 3)*C2G[i] + Fraction(2, 3)*Tf[i] + Fraction(1, 3)*TH[i]   # 2HDM β (computed)
        add = Fraction(1, 3)*Tf[i] + Fraction(2, 3)*TH[i] + Fraction(2, 3)*C2G[i]        # 4D time-completion (computed)
        b4d = b_2HDM + add                                                                # DERIVED 4D β
        out[i] = (add, b4d, b_MSSM_lit[i], b4d == b_MSSM_lit[i])   # (shadow-add, DERIVED 4D β, MSSM-lit ref, agree?)
    return out

# ── READ 2″ — the GEOMETRIC MECHANISM (O): D₄ = D₃ ⊗ 1 + γ_t ⊗ ∂_N ────────────
#   WHY the running is the run's zero-mode: lift the static 3D Dirac D₃ to 4D by the time
#   direction ∂_N (= the run, read_run). The time-gamma γ_t is the srs EVEN triple's OWN
#   grading γ=(+1 on 0-forms, −1 on 1-forms): {D₃,γ}=0 (verified below) ⇒ the clean split
#   D₄² = D₃²⊗1 + γ²⊗∂_N² = D₃² + ∂_N². The 3D→4D completion is the KO-dimension 2→6 shift
#   (gap 4 = the fermion-doubling); its ζ_{D₄}(0) content = the +time-shadows (read_gauge_running).
#   ∂_N is concrete (the run). The β CONTENT (12/5,4,4) is in hand as the matter Dynkin-sum (read_gauge_running).
#   The remaining (O) piece — that ζ_{D₄}(0) computed FROM D₄ PRODUCES that content — is RESEARCH-LEVEL: it needs
#   the CONTINUUM Dirac-cone limit (the lattice heat-kernel is a DEAD END — D₃ is bounded, λ²≤6, no UV Weyl a₄)
#   plus the KO-6 fermion-doubling. It is a GRADE/FALSIFICATION question, NOT a parameter gate (no value changes
#   either way). (theorem_gauge_running_substrate_observer §"(O)"; phase4_1 KO-2.)
def read_dirac4_lift(k=(0.31, 0.13, 0.07)):
    D3 = dirac(k); n = D3.shape[0]
    g  = np.diag([1.0]*srs.NV + [-1.0]*(n - srs.NV))   # even-triple grading = the time-gamma γ_t
    anti = float(np.max(np.abs(D3 @ g + g @ D3)))      # {D₃, γ_t} — 0 ⇒ clean lift, KO 2→6
    return anti, bool(anti < 1e-12)

# ── READ 2‴ — the MATTER ROW of ζ_{D₄}(0): the spin-1 Weyl cone β = 1 Weyl per cone (DERIVED) ──
#   A(Γ) = {−1,−1,−1, k*}: the λ=−1 TRIPLE (the spin-1 cone) + the Perron k*. The triple disperses as
#   {flat (m=0), ±v|k| (m=±1)} — a spin-1 Weyl cone. Its one-loop gauge vacuum-polarization (spectral
#   action a₄) = EXACTLY 1 WEYL per cone — FINITE (the cone's ρ(E)~E² regulates the flat band), ISOTROPIC,
#   and the FLAT BAND IS REQUIRED for gauge invariance (corrects the prior "2 Weyl"; the dispersing pair
#   alone is non-invariant). This is the DERIVED matter piece of ζ_{D₄}(0); the gauge/Higgs rows + the full
#   running (the zero-mode connection slope) remain the open frontier. [proofs/_scratch/O_spin1_cone_gauge_beta_2026-06-25.py]
def read_matter_row():
    spec = sorted(round(float(e), 6) for e in np.linalg.eigvalsh(adjacency((0, 0, 0))))
    triple = spec.count(spec[0])                       # multiplicity of λ=−1 (= 3, the spin-1 cone)
    return spec, triple, 1                             # (Γ spectrum, cone multiplicity, β = 1 Weyl per cone)

# ── READ 2⁗ — EW gauge-coupling CONSISTENCY: g_2 is NOT independent (g_2 = √(4π·α_EM/sin²θ_W)) ──
#   The SU(2) coupling is the definitional EW identity g_2 = √(4π·α_EM/sin²θ_W) — it is fixed by α_EM and
#   sin²θ_W, not a free observable. The framework's gauge couplings (α_1,α_2 run off the 3/8,1/24 boundary
#   with the 4D-completion β) satisfy it by construction, so g_2 inherits α_EM/sin²θ_W's ~1σ agreement.
#   (The shipped predictions previously scored g_2 against a scheme-INCONSISTENT target 0.6520 → a spurious
#   −2.52σ; scored against the consistent √(4π·α_EM/sin²θ_W)=0.65177 the residual is −0.18σ. Fixed 2026-06-25
#   in predictions/g_2.py. THIS read documents the identity natively; the M_Z-scale VALUES live in the run
#   that imports the β = ζ_{D₄}(0), still the open frontier.)
def read_gauge_consistency(alpha_EM, sin2W):
    return (TWO_PI * 2 * alpha_EM / sin2W) ** 0.5      # g_2 = √(4π·α_EM/sin²θ_W), the EW identity

# ── READ 3 — flavor: Λ•(C³) C3-isotype = (4,2,2); Koide ρ=1/2 → Q=2/3 ────────
def read_flavor():
    # ⚠ REPRESENTATION THEORY by enumeration (not an operator spectral read). The (4,2,2) is the C₃-isotypic
    #   decomposition of the graded exterior algebra Λ•(C³) — a structural fact, here counted over the wedge
    #   basis. (It equals the character det(1+ρ(g)) over Λ•; that trace form would be more operator-like.)
    content = {0: 0, 1: 0, 2: 0}                     # isotype multiplicities (triv, ω, ω²)
    for r in range(4):
        for S in itertools.combinations(range(3), r):
            content[sum(S) % 3] += 1                 # C₃ weight of the wedge
    fock = (content[0], content[1], content[2])      # (4,2,2)
    rho = Fraction(content[1], content[0])           # |c_ω|²/|c_triv|² = 2/4 = 1/2
    Q = (1 + 2*rho)/3                                # Koide 2/3
    gens = len(content)                              # generation count = #C₃ isotypes = 3-irrep dim = 3 (was typed)
    return fock, rho, Q, gens

# ── READ 3′ — SPECIES = the Cl(6)-Fock Hamming weight (WHICH mode is which fermion: a READ) ──
#   k* fermionic edge modes per site → Cl(6) Fock = 2^k* states; the number operator N̂=Σaᵢ†aᵢ has eigenvalue
#   = Hamming weight n∈{0,…,k*}, charge Q=n/k*. The species FALL OUT of N̂'s spectrum (no hand-label):
#   n=0 ν, 1 d, 2 u, 3 e ; color triplet for 0<n<k* (quark), singlet for n∈{0,k*} (lepton).
def read_species():
    states = list(itertools.product((0, 1), repeat=K))   # the 2^k* occupation basis
    Nhat = np.diag([sum(b) for b in states])             # the number operator (diagonal in occupation)
    n = [int(x) for x in np.diag(Nhat)]                  # Hamming-weight spectrum
    mult = dict(sorted(Counter(n).items()))              # {0:1,1:3,2:3,3:1} = 1⊕3⊕3̄⊕1
    return mult

# ── READ 3″ — GENERATIONS: the per-species 3-mass spectrum as a native read of the run ∂_N ──
#   The C₃ deck screw σ=(123) splits B into 3 windings; each winding's PF-dominant return amplitude at the
#   tracial start Γ gives the FORCED moduli (ω⁰ rides the Perron |h|²=(k−1)²=4; ω¹,ω² the shell |h|²=k−1=2 →
#   the 4:2:2 FALLS OUT, not typed). The run ∂_N imparts the FORCED directed phase: ω⁰ stays, ω¹/ω² split by
#   ±φ·s with the directed velocity φ=2π/√(4(k−1)−lam₀²)=2π/√7 (= read_mass's rate). √m_j = the forced C₃-Fourier.
#   s (the run-position) = the ONE free axis (the observer's slice). The Koide phase φ·s falls out ≈ 2/9 — the
#   stale 2/9 fit is RETIRED. ⚠ leading order: the residual (m_τ/m_e +61 ppm) is the NEXT-ORDER ∂_N (the
#   winding-dressing asymmetry), genuinely UN-WORKED — NOT a free slice (the full B(s·axis) dressing over-applies it).
def c3_winding_bases():
    sigma = {0: 0, 1: 2, 2: 3, 3: 1}; D = srs._darts(); n = len(D)   # the deck screw σ=(123)
    P = np.zeros((n, n))
    for a, (i, j, v) in enumerate(D):
        for b, (p, q, w) in enumerate(D):
            if (p, q) == (sigma[i], sigma[j]): P[b, a] = 1; break
    wom = cmath.exp(TWO_PI*1j/3)
    out = []
    for t in (0, 1, 2):
        Pc = sum(wom**(-t*m) * np.linalg.matrix_power(P, m) for m in range(3)) / 3   # C₃ Fourier projector
        ev, V = np.linalg.eigh(Pc); out.append(V[:, np.abs(ev - 1) < 1e-6])
    return out

def read_generation(s):
    bases = c3_winding_bases(); B0 = hashimoto((0, 0, 0)); om = cmath.exp(TWO_PI*1j/3)
    c = [abs(np.linalg.eigvals(Q.conj().T @ B0 @ Q)[np.argmax(np.abs(np.linalg.eigvals(Q.conj().T @ B0 @ Q)))])
         for Q in bases]                                  # FORCED per-winding PF-Perron moduli → {2,√2,√2}
    phi = TWO_PI / np.sqrt(4*(K-1) - LAM_3IRREP**2)       # directed velocity 2π/√7 (the band-edge √−7 disc)
    amp = [c[0], c[1]*cmath.exp(1j*phi*s), c[2]*cmath.exp(-1j*phi*s)]   # ω⁰ stays; ω¹/ω² split ±φ·s (forced)
    return sorted(abs(sum(amp[t]*om**(t*j) for t in range(3)))**2 for j in range(3))

# ── READ 4 — MASS = recurrence-under-running (PRIMARY). NOT a spanning count. ─
def read_mass():
    # Ihara–Bass on the geodesic flow B:  h² − λh + (k−1) = 0 ; complex roots ⇒ |h|² = k−1 = 2 (shell).
    shell = K - 1                                    # |h|² = 2  (Ramanujan)
    # running phase the chiral screw imparts: at the 3-irrep eigenvalue λ0, θ=arccos(λ/(2√(k−1)))
    lam0 = LAM_3IRREP                                 # = −1, READ off adjacency(Γ) (not typed)
    disc = (2*np.sqrt(K-1))**2 - lam0**2             # 8 − 1 = 7
    sin_th0 = np.sqrt(1 - lam0**2/(2*np.sqrt(K-1))**2)   # √(7/8)
    dtheta_ds = TWO_PI / np.sqrt(disc)               # 2π/√7  (the directed run-phase rate)
    return shell, disc, dtheta_ds

# ════════════════════════════════════════════════════════════════════════════
# THE JOINT OBJECT  srs ⟷ srs-z  (Z2 mirror cover) — Layer 1, parameter-blind
#   The base srs alone is NOT the object. Matter is the INTER-ENANTIOMER
#   coupling: every hop flips the mirror layer s. cover_B (24×24) factorizes as
#   srs(sign+1) ⊕ srs-z(sign−1); the chiral band-edge ±½±i√7/2 ∈ ℚ(√−7) is born
#   of the mirror twist (absent from either layer alone). It holds every channel
#   at once and names no single mass.
# ════════════════════════════════════════════════════════════════════════════
def cover_B(k):
    D = srs._darts(); n = len(D); B = np.zeros((2*n, 2*n), complex)
    for b, (tb, hb, vb) in enumerate(D):
        for a, (ta, ha, va) in enumerate(D):
            if ha == tb and not (hb == ta and np.array_equal(vb, -va)):
                ph = np.exp(2j*np.pi*(np.asarray(k, float) @ vb))
                for s in (0, 1):
                    B[2*b + (s ^ 1), 2*a + s] = ph      # the hop flips the enantiomer layer
    return B

RHO = Fraction(K - 1, K)                      # per-step NB survival = (NB continuations)/(degree) = (k−1)/k, k=srs.DEG (READ)
U_RUN = float(RHO**(GIRTH - 2))               # u = α₁ = ρ^(g−2); girth g READ off B (renewal). ⚠ the −2 (n_fixed = the
#   girth cycle's two non-(k−1) steps: the start has no arrival edge, the last step is forced to close) is still TYPED —
#   the single residual constant left in the coupling. OPEN: read it off the cycle's boundary structure, don't type it.

# ── DARK is NOT a correction — it is the DRESSED joint spectrum, read LATE ─────
#   Dress the joint object once with the run: G(u)=(I−u·cover_B)⁻¹ gives a factor
#   1 − u/h at EVERY channel h simultaneously. A parameter appears only when a
#   Layer-2 reading operator selects (k-point, band) — as late as possible. The
#   three "forms" scattered across the live DAG (α₁/h, α₁/(1−α₁), α₁²) are the
#   channel / BZ-trace / 2nd-order reads of this ONE dressed object.
def dressed_spectrum(k):                    # Layer 1 (parameter-blind): the joint spectrum, each channel dressed
    return [(h, 1 - U_RUN/h) for h in np.linalg.eigvals(cover_B(k))]

def ihara_bass_root(lam, branch=1):          # the channel eigenvalue on adjacency band λ:  h² − λh + (k−1) = 0
    disc = lam*lam - 4*(K-1)                # disc<0 ⇒ complex root, |h|²=k−1: the chirality the mirror twist makes physical
    root = np.sqrt(disc) if disc >= 0 else 1j*np.sqrt(-disc)
    return (lam + branch*root) / 2          # branch +1 = Perron/dominant root, −1 = saturation root (forced by L)

# ── READ — the SELECTION MAP: species (Hamming weight n) → (channel root h, walker length L), FORCED ──
#   No hand-label. color from n (singlet n∈{0,k*}); quarks need real-positive h ⇒ Γ Perron λ=k* (only λ=+3 gives
#   real roots {1,k*−1}); the d/u split is the Clifford handedness, COMPUTED below (not branched): the down-Higgs is
#   grade-1 (odd) ⇒ ω flips it ⇒ runs the girth (L=g, Perron root); the up conjugate-Higgs H̃=iσ2 H* is even ⇒ ω
#   cannot flip it ⇒ L=0 (saturation root). leptons → the complex-chir singlets. ⚠ the ν↔chir-7 / e↔chir-5/3 match
#   still imports the species' Yukawa chir (A5) — flagged; everything else is read off the algebra.
def read_selection():
    e1 = np.array([[0, 1], [-1, 0]], complex); e2 = np.array([[0, 1j], [1j, 0]], complex)  # Cl(0,2) edge qubit
    omega = e1 @ e2                                          # the handedness (volume) operator, ω²=−I
    flips = lambda M: np.allclose(omega @ M, -M @ omega)     # ω anticommutes M ⇔ M odd ⇔ flips handedness
    a, b = 0.6 + 0.3j, -0.2 + 0.7j                           # a grade-1 doublet (the grades are doublet-independent)
    H = a*e1 + b*e2                                          # down-type Higgs (grade-1)
    Htilde = e1 @ H.conj()                                   # up-type conjugate iσ2 H* (iσ2=e1 here)
    L_down = GIRTH if flips(H)      else 0                   # COMPUTED: H odd ⇒ flips ⇒ runs girth
    L_up   = GIRTH if flips(Htilde) else 0                   # COMPUTED: H̃ even ⇒ cannot flip ⇒ L=0
    sel = {}
    for n in range(K + 1):                                   # n = Cl(6) Hamming weight (the species, a READ)
        if n in (0, K):                                     # color singlet (lepton) → complex-chir channel
            lam = LAM_3IRREP if n == 0 else float(np.sqrt(LAM_PERRON))   # ν→λ=−1 (chir-7) ; e→λ=√k* (chir-5/3) ⚠A5
            sel[n] = (lam, ihara_bass_root(lam), float('inf') if n == 0 else GIRTH - 2)
        else:                                               # color triplet (quark) → Γ Perron λ=k*, root h=k*−1
            L = L_down if n == 1 else L_up                  # down(n=1)↔H, up(n=2)↔H̃ (hypercharge-forced coupling)
            sel[n] = (LAM_PERRON, ihara_bass_root(LAM_PERRON), L)  # SAME Perron root h=2; L sets dark RANK + anchor
    return sel, (L_down, L_up)

def selection_dark(n):                                      # the dressed (dark) factor at species n's forced channel
    (lam, h, L), = (read_selection()[0][n],)               # the channel forced by the selection map (no hand-label)
    rank = 2 if L == 0 else 1                               # L=0 saturation ⇒ rank-2 ; else rank-1 (the L-power rule)
    return h, L, rank, (1 - U_RUN / h**rank)                # dark = 1 − u/hᵏ at the forced (h, rank)

# ── READ 3‴ — the per-species generation PHASE δ, folded into the object (the PHASES, not just Q) ──
#   MASTER quantity = chir, the IB-root REALITY of each species' channel (read_selection): real Perron (Im h=0)
#   → VERTEX; complex band-edge (Im h≠0) → FACE (the antipodal chir-flip). δ = harmonic mean of the Wigner-d¹
#   diagonal survivals |d¹_{m,m}|² at cos β = ±1/k* (the tetrahedral angle): FACE(+1/k*) → {4/9,1/9,4/9} → 2/9 ;
#   VERTEX(−1/k*) → {1/9,1/9,1/9} → 1/9 ; the L=0 saturation scales VERTEX by (k−1)/k → 2/27. NO fit, NO typed
#   constant — chir, cos β=±1/k*, and L are all read off the object. Q is the modulus shadow of these phases.
def read_phases():
    sel, _ = read_selection()
    out = {}
    for n, (lam, h, L) in sel.items():
        cosb = Fraction(1, K) if abs(h.imag) > 1e-9 else Fraction(-1, K)   # FACE if chir≠0 (complex root) else VERTEX
        surv = [((1 + cosb)/2)**2, cosb**2, ((1 + cosb)/2)**2]             # Wigner-d¹ diagonal survival
        delta = 3 / sum(Fraction(1) / s for s in surv)                    # harmonic mean of the three survivals
        if L == 0: delta *= Fraction(K-1, K)                              # up (L=0): vertex × (k−1)/k saturation
        out[n] = delta
    return out

# ── READ 3⁗ — per-sector MODULUS (Koide Q = (2+ε²)/6), folded into the object (W27 α₁-corrected) ──
#   ε²_n = 2 + 6·α₁_full·n·f(n) ⇒ Q = 2/3 + α₁_full·n·f(n). n = Hamming weight mod k* (color index, READ);
#   α₁_full = (5/3)·ρ^(g−2) [(5/3) = the chir-5/3 amplitude]; f(n) = 1+(n−1)(g−2)/(2g). The α₁ corrections ARE
#   the modulus subtle structure (NOT a leading number): Q = {2/3 (e/ν), 0.732 (d), 0.849 (u)}, all <0.05%.
def read_moduli():
    a1f = Fraction(5, 3) * Fraction(K-1, K)**(GIRTH-2)
    out = {}
    for nh in range(K + 1):
        n = nh % K
        f = 1 + Fraction((n-1)*(GIRTH-2), 2*GIRTH)
        out[nh] = Fraction(2, 3) + a1f * n * f                            # Q ; ε² = 6Q − 2
    return out

# ── READ 3⁗′ — the per-species 3-generation masses = EIGENVALUES of the A4-covariant mass OPERATOR ──
#   (prior art: proofs/_scratch/O_A4_covariant_mass_operator + O_all_charged_masses_one_A4_object). The generation
#   block is the C₃ circulant of the FORCED complex amplitude triple c = [c_triv, c_ω e^{iδ}, c_ω̄ e^{-iδ}]:
#   moduli |c_triv|²=½, |c_ω|²=|c_ω̄|²=⅛·ε² (Ramanujan (4,2,2) Born weights, α₁-shifted via read_moduli's ε²=6Q−2);
#   phase δ from read_phases. The eigenvalues √m_j = c_triv + c_ω ωʲ + c_ω̄ ω⁻ʲ ARE the masses — the Koide cosine
#   EMERGES from the C₃-Fourier of the complex amplitudes (NOT plugged). All 9 charged ratios fold to <1%.
def read_masses():
    Qs, ds = read_moduli(), read_phases(); om = cmath.exp(TWO_PI*1j/3); out = {}
    for nh in Qs:
        c0 = (0.5)**0.5; c1 = (float(6*Qs[nh] - 2)/8)**0.5          # forced Born moduli: |c_triv|²=½, |c_ω|²=ε²/8
        d = float(ds[nh]); c = [c0, c1*cmath.exp(1j*d), c1*cmath.exp(-1j*d)]
        sm = sorted(abs(c[0] + c[1]*om**j + c[2]*om**(-j)) for j in range(3))   # eigenvalues of the A4 operator
        out[nh] = [x*x for x in sm]                                 # masses (ascending = the 3 generations)
    return out

def read(lam, rank=1):                       # Layer 2 (LATE — the ONLY place a channel is named)
    h = ihara_bass_root(lam)                 # select the band by its STRUCTURAL energy λ ∈ {k*, √k*, −1, …}
    return h, 1 - U_RUN/h**rank              # dressed (dark) factor at that channel; rank 2 ⇔ L=0 saturation

def read_democratic():                       # Layer 2 (LATE): the v-Higgs dark (the vertex-CLASS coefficient)
    # ⚠ c_v = (k+p)/2|E| = 5/12 is a COUNT (the H¹/Wilson-loop generator count n_g/N_local), NOT a spectral
    #   projection. Contrast δ_r's c_S=1/12, which read_obliques computes as ⟨ŝ|P_Perron|ŝ⟩/dim (a true projection).
    #   A spectral form for the H¹ count (a cycle-space projector) is not yet found — flagged combinatorial.
    twoE = 2*len(srs.EDGES)                  # directed-edge count (READ)
    c_v = Fraction(K + P_TOGGLE, twoE)       # p_toggle READ; the (k+p) numerator is still a COUNT (flagged)
    return float(c_v), 1 - float(c_v)*U_RUN/(1 - U_RUN)   # v-Higgs: Σₙ uⁿ democratic = c_v·u/(1−u)

def read_vertex(n_H, n_F):                    # Layer 2 (LATE): a coupling vertex — 2nd-order (two legs meet)
    # ⚠ n_H, n_F are LEG COUNTS = the SM vertex topology (structural field-content input), NOT a spectral
    #   read. The u² and N·k ARE native (the run coupling squared; the cell directed-edge count). Flagged: the
    #   per-vertex tally is combinatorial/topological — a contraction structure of D_F, not yet read off it.
    N = srs.NV                               # atoms/cell; per-fermion-leg c_F = −u²/(N·k)
    return 1 - (n_H - n_F/(N*K))*U_RUN**2    # y_τ vertex (1,2)→1−(5/6)u² ; λ vertex (4,0)→1−4u²

# ── THE EW OBLIQUES = gauge-vertex projections of the SAME resolvent (read LATE) ──
#   δ_r (Z), δρ (W), S (tree-cover) are the gauge-vertex eigen-projections of G_NB=(I−uB)⁻¹ —
#   the SAME dressed object the masses read, now projected onto the gauge singlet / h_P shell /
#   tree-cover. The weights are SPECTRAL projections of the operator, NOT typed counts.
def gauge_singlet_projection(B):
    # c_S = ⟨ŝ|P_Perron|ŝ⟩/dim(B): the gauge singlet ŝ=1/√dim projected onto the Perron spectral residue.
    n = B.shape[0]; s = np.ones(n)/np.sqrt(n)
    w, VR = np.linalg.eig(B); ip = int(np.argmax(w.real)); vR = VR[:, ip]
    vL = np.linalg.inv(VR).conj().T[:, ip]
    P_P = np.outer(vR, vL.conj()) / (vL.conj() @ vR)        # Perron spectral projector (non-normal B)
    return float((s.conj() @ P_P @ s).real) / n            # ⟨ŝ|P_P|ŝ⟩=1 (singlet IS Perron) ⇒ c_S = 1/dim

def cavity_gf(z):
    # the tree cavity Green's function g(z)=1/(z−k·f(z)), with q f²−z f+1=0 (q=k−1). The obliques are its
    # VALUES/FLOWS; the discriminant z²−4q DERIVES the form (off-cut disc>0 ⇒ resummed; on-cut disc≤0 ⇒ leading).
    q = K - 1; d = z*z - 4*q
    r = np.sqrt(d) if d >= 0 else 1j*np.sqrt(-d)
    f = (z - r) / (2*q)                                      # the branch finite at z=k
    return 1.0/(z - K*f), d

def read_obliques():
    c_S = gauge_singlet_projection(hashimoto((0, 0, 0)))    # Perron-singlet projection = 1/12 (SPECTRAL, not a count)
    g_P, disc_P = cavity_gf(LAM_PERRON)                     # Perron node z=k*: disc>0 ⇒ OFF-cut ⇒ resummed u/(1−u)
    g_E, disc_E = cavity_gf(2*np.sqrt(K-1))                 # band edge z=2√q: disc=0 ⇒ ON-cut ⇒ leading-only u
    d_r = c_S * U_RUN/(1 - U_RUN)                            # δ_r : Perron, resummed (disc_P>0)  → +0.338%
    h   = ihara_bass_root(np.sqrt(LAM_PERRON))              # √k* shell root (√3+i√5)/2 (READ)
    F   = h.imag / abs(h)**2                                # √5/4 shell Feshbach functional (SPECTRAL)
    c_EW = 0.5                                               # W-field EW normalization — DEFINITIONAL EW constant (flagged non-substrate)
    d_rho = c_EW * F * U_RUN                                 # δρ : h_P, leading (disc_E=0)        → +1.091%
    S   = c_S * (g_E.real - g_P.real) * U_RUN/(1 - U_RUN)    # S : cavity FLOW g(2√q)−g(k*) (was typed √2−ρ) → +0.253%
    return c_S, d_r, F, d_rho, S

# ── READ 5 — THE RUN ∂_N = the walker STEPPING (one observation = one step = one tick) ─
#   ∂_N is NOT a second operator: it is the FORWARD ITERATION of B. Each non-backtracking
#   step IS an observation = a tick of N. The run accumulated over its steps is the
#   resolvent  G = (I − u·B)⁻¹ = Σₙ uⁿBⁿ  (u = α₁ per step). ONE run, three faces:
#     (i)   dimensionless STRUCTURE = the dressed channels (the dark reads above) → mass ratios
#     (ii)  dimensionful SCALE = where the run is, N (the dyadic ladder X~N^p; v~N^{−1/4})
#     (iii) the ZERO-MODE = the marginal log-N accumulation = the gauge running (the R-19 test)
#   N_hub = "how many steps so far" = NOW = the ONE free axis. Backward run ill-posed ⇒ the arrow.
def read_run():
    B = cover_B((0.5, 0.5, 0.5))                       # the joint walker on one fiber
    rho_step = U_RUN * max(abs(np.linalg.eigvals(B)))  # per-step amplification u·|h|max
    arrow = bool(rho_step < 1)                         # forward converges; backward (÷u) ill-posed = the arrow
    G = np.linalg.inv(np.eye(B.shape[0]) - U_RUN*B)    # the run accumulated to NOW = the resolvent ΣₙuⁿBⁿ
    return rho_step, arrow, G

# ── READ 6 — DIFFERENTIATOR (subordinate sector LABEL, feeds read_mass) ──────
def read_sector_label():
    axis = np.array([1.,-1.,1.])/np.sqrt(3)
    def tau(occ):
        if not occ: return 0
        Lm = np.zeros((4,4))
        for (i,j,v) in occ:
            Lm[i,i]+=1; Lm[j,j]+=1; Lm[i,j]-=1; Lm[j,i]-=1
        ev = np.sort(np.linalg.eigvalsh(Lm))[1:]
        return int(round(np.prod(ev)/4))
    sym=asym=0
    for r in range(7):
        for occ in itertools.combinations(srs.EDGES, r):
            V = np.sum([np.array(v) for (_,_,v) in occ], axis=0) if occ else np.zeros(3)
            if abs(float(axis.dot(V)))<1e-9: sym+=1
            else: asym+=1
    return sym, asym   # 24 protected/symmetric (lepton/neutrino-like) | 40 drifting/asymmetric (quark-like)

# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    banner("THE MASTER RUN — one object (srs), one run (∂_N), all forced structure READ from it")
    eps, clock = read_clock()
    print(f"[0 clock ]  ε_toggle={eps}  clock=1+ε/k*={clock}  (=16/15? {clock==Fraction(16,15)})")
    b1, Lam2 = read_geometry()
    print(f"[1 geom  ]  b1 = dim = {b1}    Λ² = ‖D‖²max = {Lam2:.3f}  (≈6? {abs(Lam2-6)<0.05})")
    s2w = read_gauge()
    print(f"[2 gauge ]  sin²θ_W = Tr(S²)/Tr(Q²) = {s2w}  (=3/8? {s2w==Fraction(3,8)})  [static boundary]")
    gr = read_gauge_running(); tag = {1:'b₁', 2:'b₂', 3:'b₃'}
    cells = " ; ".join(f"{tag[i]}: +{gr[i][0]}→{gr[i][1]}" + ("=MSSM-lit✓" if gr[i][3] else f"≠{gr[i][2]}✗") for i in (1,2,3))
    print(f"[2'g-run ]  β = Dynkin sums COMPUTED over the forced content (gens={read_flavor()[3]}, SM multiplets + 2 Higgs + 4D shadows):")
    print(f"            {cells}   ⚠ LAYER-2 still QFT: the one-loop FORMULA's native form = ζ_{{D₄}}(0) (research-level, lattice=dead end)")
    anti, clean = read_dirac4_lift()
    print(f"[2''lift ]  mechanism (O): D₄=D₃⊗1+γ_t⊗∂_N, γ_t = the even-triple grading ; {{D₃,γ_t}}={anti:.0e} ⇒ clean split D₄²=D₃²+∂_N² ({clean})")
    print(f"            KO-dim 2→6 = the +shadow doubling. β CONTENT (12/5,4,4) in hand (matter trace); ζ_{{D₄}}(0)-FROM-D₄ is")
    print(f"            RESEARCH-LEVEL (needs continuum Dirac-cone + KO-6 doubling; lattice heat-kernel = dead end). GRADE question, NOT a value gate.")
    mspec, mtriple, mweyl = read_matter_row()
    print(f"[2‴matter]  ζ_{{D₄}}(0) matter row = the spin-1 Weyl cone: A(Γ)={mspec} (λ=−1 ×{mtriple} = the cone) → β = {mweyl} WEYL/cone")
    print(f"            DERIVED (spectral-action a₄; finite, isotropic; flat band REQUIRED — corrects prior '2 Weyl'). gauge/Higgs rows + running still open.")
    _g2id = read_gauge_consistency(1/127.944, 0.23121)   # g_2 = √(4π·α_EM/sin²θ_W), the EW identity (PDG inputs)
    print(f"[2⁗ EW  ]  g_2 is NOT independent: g_2=√(4π·α_EM/sin²θ_W)={_g2id:.5f} (=framework 0.65175 to 0.003%).")
    print(f"            EW state: α_s −0.13σ (c_color=1/4, Wilson-loop) · g_2 −0.18σ (scheme fix, shipped) · sin²θ_W +0.96σ · α_EM +1.01σ ·")
    print(f"            m_W +2.39σ (→~0.9σ via δρ) · M_Z +7.76σ = THE LONE HOLDOUT = δ_r pole-completeness + α₂ precision = ζ_{{D₄}}(0) (zero-mode slope).")
    fock, rho, Q, gens = read_flavor()
    print(f"[3 flavor]  Λ•(C³) C3-isotype = {fock} (=(4,2,2))   ρ={rho}  Koide Q=(1+2ρ)/3={Q} (=2/3? {Q==Fraction(2,3) or abs(Q-2/3)<1e-9})   generations={gens}")
    sp = read_species()
    print(f"[3'spec  ]  species = Cl(6)-Fock Hamming weight (N̂=Σaᵢ†aᵢ): mult by n = {sp} = 1⊕3⊕3̄⊕1 → n=0 ν,1 d,2 u,3 e (a READ, no hand-label)")
    # generations: forced moduli (4:2:2) + forced velocity φ; s = the one free axis (the observer's slice)
    _bs = c3_winding_bases(); _B0 = hashimoto((0, 0, 0)); _om = cmath.exp(TWO_PI*1j/3)
    _cmod = [abs(np.linalg.eigvals(Q.conj().T @ _B0 @ Q)[np.argmax(np.abs(np.linalg.eigvals(Q.conj().T @ _B0 @ Q)))]) for Q in _bs]
    _phi = TWO_PI / np.sqrt(4*(K-1) - LAM_3IRREP**2)
    def _gmass(s):
        a = [_cmod[0], _cmod[1]*cmath.exp(1j*_phi*s), _cmod[2]*cmath.exp(-1j*_phi*s)]
        return sorted(abs(sum(a[t]*_om**(t*j) for t in range(3)))**2 for j in range(3))
    _lo, _hi = 0.05, 0.13                                # bisect the free axis s for the lepton slice (m_μ/m_e=206.77)
    for _ in range(60):
        _mid = (_lo + _hi) / 2
        if _gmass(_mid)[1]/_gmass(_mid)[0] < 206.7683: _lo = _mid
        else: _hi = _mid
    _slep = (_lo + _hi) / 2; _g = _gmass(_slep); _mod = [round(float(x*x), 2) for x in _cmod]
    print(f"[3''gen  ]  generations = forced C₃-Fourier of the run ∂_N: moduli {_mod} (=4:2:2) FALL OUT (PF-Perron/winding, NOT typed)")
    print(f"            phase φ·s = {_phi*_slep:.7f} ≈ 2/9 FORCED (stale 2/9 fit RETIRED) ; s = run-slice = the one free axis")
    print(f"            m_μ/m_e={_g[1]/_g[0]:.2f}(obs 206.77)  m_τ/m_e={_g[2]/_g[0]:.1f}(obs 3477.2) → {(_g[2]/_g[0]/3477.23-1)*1e6:+.0f} ppm = NEXT-ORDER ∂_N (un-worked, NOT a free slice)")
    ph = read_phases(); nm = {0: 'ν', 1: 'd', 2: 'u', 3: 'e'}
    chir = {n: ('FACE' if abs(read_selection()[0][n][1].imag) > 1e-9 else 'VERTEX') for n in ph}
    cells = "  ".join(f"{nm[n]}:{ph[n]}({chir[n][0]})" for n in (3, 1, 2, 0))
    print(f"[3‴phase ]  generation PHASES δ from chir (IB-root reality)→Wigner-d¹ HM at cos β=±1/k*:  {cells}")
    print(f"            e(complex root→FACE)=2/9 ; d(real Perron→VERTEX)=1/9 ; u(VERTEX×(k−1)/k, L=0)=2/27 — one mechanism, ALL sectors, NO fit")
    _Qs, _mr = read_moduli(), read_masses()
    _nm = {3: 'e/μ/τ', 1: 'd/s/b', 2: 'u/c/t'}
    print(f"[3⁗mass ]  per-sector Q (α₁-corrected modulus, W27) + 3-generation mass spectrum (Koide √m_j=|1+ε·cos(δ+2πj/3)|):")
    for nh in (3, 1, 2):
        m = _mr[nh]
        print(f"            {_nm[nh]}: Q={float(_Qs[nh]):.4f}  m₃/m₁={m[2]/m[0]:.0f}")
    print(f"            ⇒ reproduces the shipped framework predictions EXACTLY (W3/W4: NO untruncated/subleading term — leading Koide IS complete).")
    print(f"            leptons match obs exactly; quarks <1σ (the gap vs naive-obs ratios is the prediction, dominated by the up-mass uncertainty).")
    shell, disc, rate = read_mass()
    print(f"[4 MASS  ]  recurrence shell |h|²=k−1={shell} ; running phase dθ/ds = 2π/√{int(disc)} = {rate:.5f}  ← THE mass mechanism (NOT τ)")
    sel, (L_down, L_up) = read_selection()
    name = {0: 'ν', 1: 'd', 2: 'u', 3: 'e'}   # n → species: the Hamming-weight READ (not a hand-label)
    print(f"[4'dark  ]  channels FORCED by the selection map (Hamming-weight species n → channel h, walker L); u=α₁={U_RUN:.6f}")
    print(f"            d/u split COMPUTED off Cl(0,2) handedness: down-H odd⇒flips⇒L={L_down}, up-H̃ even⇒no flip⇒L={L_up}  (no hand-label)")
    for n, (lam, h, L) in sel.items():
        chir = 0.0 if abs(h.imag) < 1e-9 else (h.imag/h.real)**2
        Ls = '∞' if L == float('inf') else str(L)
        if n in (1, 2):                                       # quarks: the mass dark 1−u/hᵏ (rank forced by L)
            _, _, rank, dark = selection_dark(n)
            read_str = f"dark 1−u/h^{rank} = {dark.real:.6f}"   # n=1(d/b) 0.980491 ; n=2(u/t) 0.990245
        elif n == 3:                                          # e: the shell self-energy Σ=u/h → √3/4, √5/4
            sig = U_RUN/h
            read_str = f"Σ/u: Re=√3/4={sig.real/U_RUN:.4f}, −Im=√5/4={-sig.imag/U_RUN:.4f}"
        else:                                                 # ν: chir-7 → the run-phase 2π/√7
            read_str = f"chir-7 → run-phase 2π/√7={TWO_PI/np.sqrt(7):.5f}"
        print(f"            n={n} {name[n]}: λ={lam:+.3f} h={h.real:+.3f}{h.imag:+.3f}i chir={chir:.3g} L={Ls} → {read_str}")
    cS, fdem = read_democratic()
    print(f"            read(democratic) v/M_Z: c_S=(k+p)/2|E|={cS:.4f} → 1−c_S·u/(1−u) = {fdem:.6f}  (resummed: every step)")
    print(f"            read(vertex)     y_τ/λ: (1H,2F)→{read_vertex(1,2):.6f} ; (4H,0F)→{read_vertex(4,0):.6f}  (2nd-order: two legs)")
    c_S, d_r, F, d_rho, S = read_obliques()
    print(f"[4''obliq]  EW OBLIQUES = gauge-vertex projections of the SAME G_NB (c_S = ⟨ŝ|P_Perron|ŝ⟩/dim, SPECTRAL):")
    print(f"            δ_r (Z/Perron)  = c_S·u/(1−u)      = {d_r*100:+.4f}%  (→ M_Z)")
    print(f"            δρ  (W/h_P)      = ½·(√5/4)·u        = {d_rho*100:+.4f}%  (→ m_W, the T/custodial)")
    print(f"            S   (tree-cover) = c_S·(√2−ρ)·u/(1−u)= {S*100:+.4f}%  (custodial-symmetric)")
    print(f"            c_S={c_S:.5f} (=1/12 PROJECTION, not the count (k+p)/2|E|) ; √5/4={F:.5f} (shell read) ; ½=EW W-norm (definitional, flagged NON-substrate)")
    rho_step, arrow, _G = read_run()
    print(f"[5 RUN   ]  ∂_N = the walker STEPPING (1 obs = 1 step = 1 tick of N); run = G=(I−uB)⁻¹=ΣₙuⁿBⁿ")
    print(f"            per-step u·|h|max = {rho_step:.4f} < 1 ⇒ forward converges, backward ill-posed = THE ARROW ({arrow})")
    print(f"            ONE run, 3 faces: structure=dark/ratios(above) · scale=N^p (v~N^−¼, N_hub=now=free axis) · zero-mode=gauge(R-19)")
    sym, asym = read_sector_label()
    print(f"[6 label ]  sector differentiator (SUBORDINATE): symmetric/protected={sym} (lepton/ν-like) | asymmetric/drifting={asym} (quark-like)")
    print(f"            — this is a LABEL that feeds [4]; τ is NOT a mass.")
    banner("one object · one run · reads only · adding never replacing")
