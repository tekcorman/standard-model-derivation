#!/usr/bin/env python3
"""Full wave simulator over the 195-op operator-sweep catalog.

Extends the lean simulator with tag-establishment dynamics:
  - Some ops, when fired, establish assumption tags.
  - Establishing a tag unlocks downstream ops whose extras include that tag.
  - The wave propagates as a cascade of tag-unlocks.

Initial state: {A1, E_FIN, A2W, P1, A5M, E6, K3, ORDER} — the framework's structural
+ definitional tags (A1 axiom, finite alphabet, A2 waterline, P1' storability,
A5-mass labeling, framework-specific |E|=6 / k=3 / ordering choices).

Cascade gating (A2): fire any op whose extras ⊆ tags AND (Φ_marg > 0 OR emits new object).

Halting: no firable op contributes Φ_marg > 0 or new object.
"""
import math, functools
from dataclasses import dataclass, field

E = 6
N_REF = 10

# ---------------- substrate counting (lean refinements) ----------------
def n_raw(n=N_REF):     return E**n
def n_reduced(n=N_REF): return E*(E-1)**(n-1) if n >= 1 else 1

@functools.lru_cache(None)
def n_cyclic(n=N_REF):
    def euler_phi(m):
        r, mm, p = m, m, 2
        while p*p <= mm:
            if mm % p == 0:
                while mm % p == 0: mm //= p
                r -= r // p
            p += 1
        if mm > 1: r -= r // mm
        return r
    a = lambda d: (E-1)**d + (-1)**d * (E-1)
    return sum(euler_phi(n//d) * a(d) for d in range(1, n+1) if n % d == 0) // n

def n_abelian(): return 2**E

# Lean refinement class counts (overlap-aware)
LEAN_REF_COUNT = {
    frozenset():                                       n_raw(),
    frozenset({'reduced'}):                            n_reduced(),
    frozenset({'reduced','cyclic'}):                   n_cyclic(),
    frozenset({'reduced','cyclic','abelian'}):         n_abelian(),
    frozenset({'reduced','abelian'}):                  n_abelian(),
    frozenset({'cyclic','abelian'}):                   n_abelian(),
    frozenset({'abelian'}):                            n_abelian(),
}

# T2.1g: per-op LIVE Φ overrides, computed by the live executor in
# proofs/wave_engine/substrate_state.py. When an op_id is present here, its
# Φ contribution is the LIVE value (computed from actual partition / spectral
# / Clifford arithmetic) rather than the PHI_TEMPLATE lookup.
#
# These values are reproducible by running:
#   python3 proofs/wave_engine/substrate_state.py
# and reading the marginal-Φ contributions from the trace.
#
# Live values disagree with bookkeeping in two systematic ways:
#  (a) Lean refinements (0.4, 1.8, 1.10): the closed-form lean_class_count
#      under-counts partition classes at finite n by ~25/24 (asymptotic).
#  (b) Spectral / Clifford ops (4.21, 2.18, 4.17, 5.9): the BLOCH_SRS and
#      PROJ_RANK2 templates conflate distinct compressions. Live values
#      isolate per-op compression: e.g., 2.18 contributes log₂(k*/(k*-1)) = log₂(3/2),
#      not log₂(8); 5.9 contributes log₂(8/4) = 1, not log₂(8/2).
LIVE_PHI_OVERRIDES = {
    # Word-partition layer (live values at n_max=4; bookkeeping uses n=10 lean closed-form)
    # NOTE: simulator's lean_class_count uses asymptotic E·(E-1)^(n-1) at n=N_REF=10;
    # the live n_max=4 values would NOT directly substitute. We DO NOT override the
    # lean ops here — their existing analytic computation is the framework's reference
    # accounting at n=N_REF.

    # Graph layer
    '4.21': 1.0,                    # K_4 quotient: log₂(24/12) directed-edge collapse
    '2.18': math.log2(3 / 2),       # Hashimoto NB compression: log₂(k*/(k*-1)) ≈ 0.585
    '4.17': math.log2(4 / 3),       # Bloch decomposition: log₂(N_atoms / max_fiber_dim) ≈ 0.415

    # Spinor / Clifford layer
    '5.9':  math.log2(8 / 4),       # Weyl chiral split 8 = 4 + 4: log₂(8/4) = 1.0

    # T2.2 batch (info theory + harmonic analysis + parity)
    '4.5':  0.0,                    # Shannon entropy: information CONTENT, not compression
    '4.6':  0.0,                    # KL divergence: same
    '4.7':  0.0,                    # mutual information: same
    '4.8':  0.0,                    # description length: emit-only
    '4.16': 1.0,                    # isotypic C3: log₂(4/2), max block dim = 2
    '4.19': math.log2(4 / 3),       # S4 protected degeneracy at H: triple = 3-dim
    '5.10': 1.0,                    # fermion parity: log₂(2)
}

# Per-template Φ (independent compressions; each refinement contributes once)
PHI_TEMPLATE = {
    'STRUCT':         0.0,
    'INVOL':          None,  # handled by lean machinery
    'CYCL':           None,
    'QUOT_ABEL':      None,
    'QUOT_K4':        math.log2(4),
    'QUOT_C3':        math.log2(3),
    'QUOT_S4':        math.log2(24),
    'PROJ_RANK2':     math.log2(8/2),
    'PROJ_RANK1':     math.log2(8/1),
    'BLOCH_SRS':      math.log2(8),
    'PT_QUBIT':       1.0,
    'PT_DIRAC':       math.log2(8),
    'ENTROPY_C3':     math.log2(3),
    'ENTROPY_BERN':   1.0,
    'COARSE_STRAUCH': math.log2(N_REF),
    'COARSE_BZ':      math.log2(8),
    'HOMOL_E2':       float(E),
    'THERMAL_SRS':    float(E),
    'K_THEORY':       float(E),
    'MODULAR':        math.log2(n_reduced()),
    'ATIYAH_SINGER':  math.log2(8),
    'TQFT':           math.log2(4),
    'CLASSIFYING':    0.0,
    'RG':             math.log2(N_REF),
}

# ---------------- assumption tags ----------------
# LORENTZ_SIG: Lorentzian signature (-,+,+,+) derivation. CLOSED 2026-04-27 at
#   theorem grade locally via the Γ-cone Minkowski theorem
#   (predictions/lorentz_signature_local.py + commit ec98e6e); structurally
#   complete globally via Iorio elastic + linearised Einstein + G_sub two-route.
#   See docs/theorems/lorentz_sig_ccclose_joint_closure.md.
#   Established by op 6.10 (Lorentzian metric) firing.
# NC_GEOM: substrate's non-commutative-geometric structure (Connes 1994).
#   Replaces CCLOSE per the joint-closure pivot: substrate IS NC-geometric, NOT
#   Riemannian; smooth-manifold continuum-limit not required. Layer 6 ops have
#   NC analogs grounded by Connes' machinery + substrate Lichnerowicz theorem
#   (docs/forward_constructions/forward_construction_substrate_lichnerowicz.md).
#   Established by op 7.1 (spectral triple) firing.
ALL_TAGS = {
    'A1','E_FIN','ORDER','E6','K3','SRS','K4Q','CRYSTAL','C3','S4',
    'A2W','A4','A5M','P1','FF','STRAUCH','NC_GEOM','LORENTZ_SIG',
    'COMPACT','FIN_DIM','LIE','THERM','BZJ','RGFL','N_HUB','C_REP'
}
INITIAL_TAGS = {'A1','E_FIN','A2W','P1','A5M','E6','K3','ORDER'}
PARTIAL_TAGS = set()  # 2026-04-27: LORENTZ_SIG + NC_GEOM closed via joint-closure
                      # (docs/theorems/lorentz_sig_ccclose_joint_closure.md). Both now
                      # established by upstream ops.

# Establishment relations: when op id fires, which tags get added.
ESTABLISHES = {
    '2.22': {'FIN_DIM'},
    '3.1':  {'LIE'},
    '3.5':  {'FF'},
    '3.11': {'STRAUCH'},
    '4.16': {'C3'},
    '4.19': {'S4'},
    '4.21': {'K4Q','SRS','CRYSTAL'},
    '4.30': {'COMPACT'},
    '4.45': {'THERM'},
    '4.51': {'BZJ','N_HUB'},
    '4.52': {'RGFL'},
    '5.1':  {'C_REP'},
    '5.6':  {'A4'},
    # Joint-closure ESTABLISHES (2026-04-27):
    '6.10': {'LORENTZ_SIG'},  # Γ-cone Minkowski theorem
    '7.1':  {'NC_GEOM'},      # substrate IS NC-geometric (Connes 1994)
}

# ---------------- catalog: (id, layer, name, template, L, extras, refinement) ----------------
# `extras` are tags BEYOND INITIAL needed for the op to fire.
# `refinement` is one of the lean refinement names ('reduced'/'cyclic'/'abelian')
# or a per-op refinement key (used to ensure a Φ template fires only once per refinement).
# Self-required tags (e.g., A4 for JW) are removed from extras (the op establishes them).
CATALOG = [
# Layer 0
('0.1',0,'identity element id','STRUCT',1,set(),None),
('0.2',0,'generator T_e','STRUCT',1,set(),None),
('0.3',0,'sequential composition','STRUCT',1,set(),None),
('0.4',0,'involutive cancellation T_e²=id','INVOL',2,set(),'reduced'),
# Layer 1
('1.1',1,'group element g ∈ F_inv(E)','STRUCT',2,set(),None),
('1.2',1,'group multiplication','STRUCT',2,set(),None),
('1.3',1,'group inverse g⁻¹','STRUCT',2,set(),None),
('1.4',1,'group identity ε','STRUCT',2,set(),None),
('1.5',1,'powers g^n','STRUCT',2,set(),None),
('1.6',1,'left action L_h','STRUCT',2,set(),None),
('1.7',1,'right action R_h','STRUCT',2,set(),None),
('1.8',1,'conjugation c_h','CYCL',3,set(),'cyclic'),
('1.9',1,'subgroups, cosets','STRUCT',2,set(),None),
('1.10',1,'quotient F_inv(E)/N (abelianization)','QUOT_ABEL',3,set(),'abelian'),
('1.11',1,'Cayley graph','STRUCT',2,set(),None),
('1.12',1,'word length ℓ(g)','STRUCT',2,set(),None),
('1.13',1,'distance d(g,h)','STRUCT',2,set(),None),
# Layer 2
('2.1',2,'functions f: F_inv(E) → 𝔽','STRUCT',2,set(),None),
('2.2',2,'pointwise +,·,conj','STRUCT',2,set(),None),
('2.3',2,'counting (Haar) measure','STRUCT',2,set(),None),
('2.4',2,'sums Σ_g f(g)','STRUCT',2,set(),None),
('2.5',2,'L²(F_inv(E);𝔽) Hilbert space','STRUCT',3,set(),None),
('2.6',2,'orthonormal basis {δ_g}','STRUCT',3,set(),None),
('2.7',2,'Hilbert-space completeness','STRUCT',2,set(),None),
('2.8',2,'bounded linear operators','STRUCT',3,set(),None),
('2.9',2,'adjoints T*','STRUCT',2,set(),None),
('2.10',2,'unitary/SA/skew classifications','STRUCT',3,set(),None),
('2.11',2,'spectral content of bounded SA','STRUCT',3,set(),None),
('2.13',2,'left regular representation L_h','STRUCT',3,set(),None),
('2.14',2,'right regular representation R_h','STRUCT',3,set(),None),
('2.15',2,'adjacency operator A=Σ L_e','STRUCT',3,set(),None),
('2.16',2,'self-adjointness of A','STRUCT',2,set(),None),
('2.17',2,'spectral decomposition of A','BLOCH_SRS',4,{'SRS','CRYSTAL'},'bloch_decomp'),
('2.18',2,'Hashimoto operator (directed-edge)','BLOCH_SRS',4,{'SRS','CRYSTAL'},'bloch_decomp'),
('2.20',2,'bounded operators ℬ(L²)','STRUCT',2,set(),None),
('2.21',2,'compact operators 𝒦(L²)','STRUCT',3,set(),None),
('2.22',2,'trace-class ℬ_1(L²)','STRUCT',3,set(),None),  # establishes FIN_DIM
('2.23',2,'Hilbert-Schmidt ℬ_2(L²)','STRUCT',3,{'FIN_DIM'},None),
('2.24',2,'self-adjoint ℬ_sa','STRUCT',2,set(),None),
('2.25',2,'closed unbounded operators','STRUCT',3,set(),None),
('2.26',2,'trace Tr(T)','STRUCT',2,{'FIN_DIM'},None),
('2.27',2,'matrix elements ⟨g|T|h⟩','STRUCT',2,set(),None),
('2.28',2,'orthogonal projection P_S','PROJ_RANK2',3,{'FIN_DIM'},'proj_rank2'),
('2.29',2,'HS norm Tr(T*T)','STRUCT',2,{'FIN_DIM'},None),
('2.31',2,'functional calculus p(T)','STRUCT',3,set(),None),
('2.33',2,'resolvent R_λ(T)','STRUCT',3,set(),None),
('2.34',2,'determinant det(T)','STRUCT',2,{'FIN_DIM'},None),
('2.35',2,'algebraic tensor product','STRUCT',3,set(),None),
('2.36',2,'Hilbert tensor product','STRUCT',3,set(),None),
('2.37',2,'tensor product of operators','STRUCT',3,set(),None),
# Layer 3
('3.1',3,'one-parameter unitary group U(t)','STRUCT',3,set(),None),     # establishes LIE
('3.2',3,'strong continuity of U(t)','STRUCT',2,{'LIE'},None),
('3.3',3,'continuous-time quantum walks','STRUCT',3,set(),None),
('3.4',3,'Stone (complex form)','STRUCT',3,{'FF','LIE'},None),
('3.5',3,'Stone (real form)','STRUCT',3,{'LIE'},None),                  # establishes FF
('3.6',3,'self-adjoint H on ℂ-L²','STRUCT',2,{'FF'},None),
('3.7',3,'skew-symmetric B on ℝ-L²','STRUCT',2,set(),None),
('3.8',3,'spectrum σ(H)⊂ℝ vs σ(B)⊂iℝ','STRUCT',3,{'FF'},None),
('3.9',3,'Cayley transform','STRUCT',3,{'FF'},None),
('3.10',3,'discrete-time quantum walk U^n','STRUCT',2,set(),None),
('3.11',3,'discrete→continuum walk limit','COARSE_STRAUCH',4,set(),'coarse_strauch'),  # establishes STRAUCH
('3.12',3,'continuum-limit Hamiltonian H','STRUCT',3,{'STRAUCH','FF'},None),
('3.13',3,"framework's specific continuum H",'STRUCT',3,{'STRAUCH','SRS','FF'},None),
# Layer 4
('4.1',4,'probability measure P','STRUCT',2,set(),None),
('4.2',4,'expectation E_P[f]','STRUCT',2,set(),None),
('4.3',4,'joint and marginal distributions','STRUCT',3,set(),None),
('4.4',4,'conditional probability / Bayes','STRUCT',3,set(),None),
('4.5',4,'Shannon entropy','ENTROPY_BERN',3,set(),'entropy_shannon'),
('4.6',4,'KL divergence','ENTROPY_BERN',3,set(),'entropy_kl'),
('4.7',4,'mutual information I(X;Y)','ENTROPY_BERN',3,set(),'entropy_mi'),
('4.8',4,'description length L(M)','ENTROPY_BERN',3,set(),'entropy_mdl'),
('4.9',4,'source coding (entropy)','ENTROPY_BERN',3,{'FIN_DIM'},'entropy_source'),
('4.10',4,'rate-distortion bound','ENTROPY_BERN',4,{'FIN_DIM'},'entropy_rd'),
('4.11',4,'discrete-time Markov chain','STRUCT',3,set(),None),
('4.12',4,'stationary distribution','ENTROPY_BERN',3,set(),'entropy_stat'),
('4.13',4,'continuous-time Markov process','STRUCT',3,{'STRAUCH'},None),
('4.14',4,'correlation function C_n(s)','STRUCT',3,set(),None),
('4.15',4,'decay rate / correlation length','STRUCT',3,{'SRS'},None),
('4.16',4,'isotypic decomposition','QUOT_C3',3,set(),'isotypic_C3'),    # establishes C3
('4.17',4,'Bloch decomposition','BLOCH_SRS',4,{'SRS','CRYSTAL'},'bloch_decomp'),
('4.18',4,'per-Brillouin-point fibers T(k)','BLOCH_SRS',3,{'SRS','CRYSTAL'},'bloch_decomp'),
('4.19',4,'symmetry-protected degeneracies','PROJ_RANK2',3,{'SRS','C3'},'proj_rank2'),  # establishes S4
('4.20',4,'Alon-Boppana / Ramanujan bound','STRUCT',3,set(),None),
('4.21',4,'group quotient F_inv(E)/N (K_4)','QUOT_K4',3,set(),'quot_K4'),  # establishes K4Q,SRS,CRYSTAL
('4.22',4,'quotient under equivalence','COARSE_BZ',3,{'SRS','CRYSTAL'},'coarse_BZ'),
('4.23',4,'coarse-graining (lossy projection)','COARSE_BZ',3,{'SRS'},'coarse_BZ'),
('4.24',4,'partial trace over subfactor','PT_DIRAC',4,{'FIN_DIM'},'pt_dirac'),
('4.25',4,'conditional expectation','STRUCT',4,set(),None),
('4.30',4,'group representation ρ:G→𝒰(V)','STRUCT',3,set(),None),       # establishes COMPACT
('4.31',4,'character χ_ρ(g)','STRUCT',2,{'COMPACT'},None),
('4.32',4,'representation matrix elements','STRUCT',2,{'COMPACT'},None),
('4.33',4,'Schur orthogonality','STRUCT',3,{'COMPACT'},None),
('4.34',4,'Peter-Weyl decomposition','QUOT_S4',3,{'COMPACT','S4'},'quot_S4'),
('4.35',4,"Wigner d-matrices",'STRUCT',4,{'LIE','COMPACT'},None),
('4.36',4,'Clebsch-Gordan decomposition','QUOT_C3',3,{'COMPACT','C3'},'cg_C3'),
('4.37',4,'Clebsch-Gordan coefficients','STRUCT',3,{'COMPACT'},None),
('4.38',4,'trace identities under reps','STRUCT',2,{'COMPACT','LIE'},None),
('4.39',4,'matrix Lie group','STRUCT',3,{'LIE','C_REP'},None),
('4.40',4,'Lie algebra','STRUCT',3,{'LIE'},None),
('4.41',4,'exponential map exp(X)','STRUCT',2,{'LIE'},None),
('4.42',4,'structure constants','STRUCT',3,{'LIE'},None),
('4.43',4,'Killing form K(X,Y)','STRUCT',3,{'LIE','COMPACT'},None),
('4.44',4,'one-parameter subgroup t↦exp(tX)','STRUCT',2,{'LIE'},None),
('4.45',4,'partition function Z(β)','THERMAL_SRS',3,{'FIN_DIM'},'thermal_Z'),  # establishes THERM
('4.46',4,'free energy F(β)','THERMAL_SRS',2,{'THERM','FIN_DIM'},'thermal_F'),
('4.47',4,'Boltzmann distribution','THERMAL_SRS',2,{'THERM','FIN_DIM'},'thermal_B'),
('4.48',4,'order parameter / phase diagram','STRUCT',2,{'BZJ'},None),
('4.49',4,'critical exponents','STRUCT',3,{'BZJ','RGFL'},None),
('4.50',4,'mean-field approximation','COARSE_BZ',3,{'SRS'},'coarse_BZ'),
('4.51',4,'BZJ scaling v∝N^{-1/4}','RG',3,{'SRS'},'rg_BZJ'),  # establishes BZJ, N_HUB
('4.52',4,'renormalization group flow','RG',4,set(),'rg_flow'),  # establishes RGFL
('4.53',4,'Curie-Weiss mean-field model','COARSE_BZ',3,{'SRS','BZJ'},'coarse_BZ'),
# Layer 5
('5.1',5,'imaginary unit i in op algebra','STRUCT',2,{'FF'},None),  # establishes C_REP
('5.2',5,'Pauli σ^x,σ^y,σ^z','STRUCT',3,{'FF'},None),
('5.3',5,'Hermitian (complex) operators','STRUCT',2,{'FF'},None),
('5.4',5,'anti-Hermitian operators','STRUCT',2,{'FF','LIE'},None),
('5.5',5,'spectral decomp (real eig, complex evec)','BLOCH_SRS',3,{'SRS','CRYSTAL','FF'},'bloch_decomp'),
('5.6',5,'Jordan-Wigner construction','STRUCT',5,{'FF'},None),  # establishes A4
('5.7',5,'CAR {c_i,c_j†}=δ_ij','STRUCT',3,{'A4'},None),
('5.8',5,'complex Clifford Cl(n;ℂ)','STRUCT',3,{'FF','C_REP'},None),
('5.9',5,'spinor reps of Cl(n;ℂ)','PROJ_RANK2',3,{'FF','C_REP'},'spinor_chiral'),
('5.10',5,'ℤ/2-grading by (-1)^F','PT_QUBIT',2,{'A4'},'pt_qubit'),
('5.11',5,'Majorana operators γ','STRUCT',3,{'A4','FF'},None),
('5.12',5,'density matrix ρ','PT_DIRAC',3,{'FIN_DIM','FF'},'pt_dirac'),
('5.13',5,'pure vs mixed state','PT_DIRAC',2,{'FF'},'pt_dirac'),
('5.14',5,'partial trace ρ_A=Tr_B(ρ_AB)','PT_DIRAC',3,{'FIN_DIM','FF'},'pt_dirac'),
('5.15',5,'purification of ρ_A','PT_DIRAC',3,{'FF'},'pt_dirac'),
('5.16',5,'Schmidt decomposition','PT_DIRAC',3,{'FIN_DIM','FF'},'pt_dirac'),
('5.17',5,'quantum tensor products w/ ent.','STRUCT',3,{'FF'},None),
('5.18',5,'complex conjugation K (anti-linear)','STRUCT',2,{'FF'},None),
('5.19',5,'anti-unitary V','STRUCT',2,{'FF'},None),
('5.20',5,'time-reversal symmetry','STRUCT',3,{'FF','SRS'},None),
('5.21',5,'Schrödinger evolution e^{-iHt}','STRUCT',3,{'FF','LIE','THERM'},None),
('5.22',5,'Heisenberg picture','STRUCT',3,{'FF','LIE','THERM'},None),
('5.23',5,'interaction picture','STRUCT',4,{'FF','LIE','THERM'},None),
('5.24',5,'time-dependent perturbation','STRUCT',4,{'FF','LIE','THERM'},None),
('5.25',5,'non-real algebraic eigenvalues','BLOCH_SRS',3,{'SRS','FF'},'bloch_decomp'),
('5.26',5,'eigenvectors w/ complex phases','BLOCH_SRS',3,{'SRS','CRYSTAL','FF'},'bloch_decomp'),
('5.27',5,'Berry / geometric phases','BLOCH_SRS',3,{'SRS','CRYSTAL','FF','LIE'},'berry_phase'),
('5.28',5,'complex Lie groups','STRUCT',3,{'LIE','C_REP','FF'},None),
('5.29',5,'spin reps of Spin(n) on Cl spinors','PROJ_RANK2',3,{'FF','C_REP','LIE'},'spinor_chiral'),
('5.30',5,'Pati-Salam embedding in Spin(6)','QUOT_K4',3,{'SRS','S4','C_REP','FF','LIE'},'pati_salam'),
('5.31',5,'complex characters χ_ρ ∈ ℂ','QUOT_C3',2,{'C3','C_REP','FF'},'isotypic_C3'),
('5.32',5,'complex Clebsch-Gordan SU(n)','QUOT_C3',3,{'C_REP','FF','LIE'},'cg_C3'),
('5.33',5,'Wick rotation t→-iτ','STRUCT',3,{'FF','LIE','THERM'},None),
('5.34',5,'quantum partition Z(β)=Tr e^{-βH}','THERMAL_SRS',3,{'THERM','FIN_DIM','FF'},'thermal_Z'),
('5.35',5,'thermal density ρ(β)','THERMAL_SRS',3,{'THERM','FIN_DIM','FF'},'thermal_F'),
('5.36',5,'von Neumann entropy S(ρ)','ENTROPY_BERN',3,{'FIN_DIM','FF'},'entropy_vN'),
('5.37',5,'Schmidt rank of bipartite pure','PT_DIRAC',3,{'FIN_DIM','FF'},'pt_dirac'),
('5.38',5,'entanglement entropy','ENTROPY_BERN',3,{'FIN_DIM','FF'},'entropy_ent'),
# Layer 5.I — Anomaly machinery (T1.3 added 2026-04-27)
('5.39',5,'Adler-Bell-Jackiw chiral anomaly','STRUCT',4,{'FF','C_REP','LIE'},None),
('5.40',5,'Wess-Zumino consistency condition','STRUCT',3,{'FF','LIE'},None),
('5.41',5,'anomaly inflow (bulk → boundary)','STRUCT',4,{'FF','LIE'},None),
('5.42',5,'anomaly cancellation on chiral content','STRUCT',4,{'FF','C_REP','LIE','S4'},None),
('5.43',5,"'t Hooft anomaly matching",'STRUCT',4,{'FF','LIE','RGFL'},None),
# Layer 5.J — S-matrix / asymptotic states / LSZ (T1.3 added 2026-04-27)
('5.44',5,'asymptotic in/out states','STRUCT',3,{'FF','STRAUCH'},None),
('5.45',5,'S-matrix S = ⟨β;out|α;in⟩','STRUCT',3,{'FF','STRAUCH'},None),
('5.46',5,'LSZ reduction formula','STRUCT',5,{'FF','STRAUCH','FIN_DIM'},None),
('5.47',5,'S-matrix unitarity S†S = I','STRUCT',2,{'FF','STRAUCH'},None),
('5.48',5,'cluster decomposition principle','STRUCT',3,{'FF','STRAUCH','SRS'},None),
('5.49',5,'cross-section dσ/dΩ','STRUCT',3,{'FF','STRAUCH'},None),
# Layer 6 — formerly CCLOSE-blocked, now NC_GEOM-grounded after joint closure
# (docs/theorems/lorentz_sig_ccclose_joint_closure.md). NC_GEOM established by op 7.1
# (spectral triple); LORENTZ_SIG established by op 6.10 (Γ-cone Minkowski theorem).
('6.1',6,'smooth manifold M','STRUCT',3,{'NC_GEOM'},None),
('6.2',6,'tangent space T_p M','STRUCT',3,{'NC_GEOM'},None),
('6.3',6,'tangent / cotangent bundle','STRUCT',3,{'NC_GEOM'},None),
('6.4',6,'tensor fields T^(p,q)(M)','STRUCT',3,{'NC_GEOM','SRS'},None),
('6.5',6,'differential forms Ω^k(M)','STRUCT',3,{'NC_GEOM','SRS'},None),
('6.6',6,'exterior derivative d','STRUCT',2,{'SRS'},None),
('6.7',6,'Lie derivative ℒ_X','STRUCT',3,{'NC_GEOM','LIE'},None),
('6.8',6,'de Rham cohomology H^k_dR','HOMOL_E2',3,{'NC_GEOM'},'cohomology_dR'),
('6.9',6,'Riemannian metric g','STRUCT',3,{'NC_GEOM','SRS'},None),
('6.10',6,'Lorentzian metric (-,+,+,+)','STRUCT',3,{'STRAUCH','SRS'},None),
('6.11',6,'Levi-Civita connection ∇','STRUCT',3,{'NC_GEOM'},None),
('6.12',6,'Christoffel symbols Γ','STRUCT',3,{'NC_GEOM'},None),
('6.13',6,'Riemann curvature R^a_{bcd}','STRUCT',3,{'NC_GEOM'},None),
('6.14',6,'Ricci R_{ab}, scalar R','STRUCT',3,{'NC_GEOM'},None),
('6.15',6,'geodesics','STRUCT',3,{'SRS'},None),
('6.16',6,'parallel transport','STRUCT',3,{'SRS','CRYSTAL'},None),
('6.17',6,'Killing vector fields','STRUCT',3,{'NC_GEOM','LIE'},None),
('6.18',6,'FLRW metric','STRUCT',3,{'NC_GEOM','LORENTZ_SIG','N_HUB'},None),
('6.19',6,'Einstein equations','STRUCT',4,{'NC_GEOM','LORENTZ_SIG'},None),
('6.20',6,'Friedmann equations','STRUCT',3,{'NC_GEOM','LORENTZ_SIG','N_HUB'},None),
('6.21',6,'Hubble parameter H(t)','STRUCT',2,{'NC_GEOM','LORENTZ_SIG','N_HUB','BZJ'},None),
('6.22',6,'cosmological scale factor a(t)','STRUCT',2,{'NC_GEOM','LORENTZ_SIG','N_HUB'},None),
('6.23',6,'stress-energy tensor T_{ab}','STRUCT',3,{'NC_GEOM','LORENTZ_SIG'},None),
('6.24',6,'causal structure / horizons','STRUCT',3,set(),None),
# Layer 7 — Non-commutative geometry / Connes spectral triples (T1.3 added 2026-04-27)
('7.1',7,'spectral triple (A, H, D)','STRUCT',4,{'FF','C_REP'},None),
('7.2',7,'bounded commutator [D, a] (Lipschitz)','STRUCT',3,{'FF'},None),
('7.3',7,'finite spectral dimension via Tr(e^{-tD²})','STRUCT',4,{'FF','FIN_DIM'},None),
('7.4',7,'p-summability D⁻¹ ∈ ℬ_p','STRUCT',3,{'FF','FIN_DIM'},None),
('7.5',7,'real structure J (anti-unitary, J²=±I)','STRUCT',3,{'FF','C_REP'},None),
('7.6',7,'Connes distance d_D(φ, ψ)','STRUCT',3,{'FF','FIN_DIM'},None),
('7.7',7,'Dixmier trace Tr_ω','STRUCT',4,{'FF','FIN_DIM'},None),
('7.8',7,'inner fluctuation D → D + A','STRUCT',4,{'FF','LIE'},None),
('7.9',7,'Ω¹_D(A) one-forms via a[D,b]','STRUCT',3,{'FF'},None),
('7.10',7,'Aut(A) inner-automorphism gauge action','STRUCT',3,{'FF','LIE'},None),
('7.11',7,'Connes-Chamseddine spectral action Tr f(D²/Λ²)','STRUCT',5,{'FF','FIN_DIM','LIE'},None),
('7.12',7,'heat-kernel expansion Σ a_{2k}(D²/Λ²)','STRUCT',4,{'FF','FIN_DIM'},None),
('7.13',7,'KK-theory class [D] ∈ KK(A,ℂ)','K_THEORY',5,{'FF'},'kk_class'),
# Appendix
('A.1',7,'group cohomology H^n(F_inv;ℤ)','HOMOL_E2',5,set(),'cohomology_groupF'),
('A.2',7,'classifying space BF_inv(E)','CLASSIFYING',5,set(),None),
('A.3',7,'K-theory K_*(C*_red(F_inv))','K_THEORY',6,set(),'K_theory'),
('A.4',7,'Atiyah-Singer / graph Dirac index','ATIYAH_SINGER',6,{'SRS','ORDER','A4','FF','C_REP'},'atiyah_singer'),
('A.5',7,'reduced group C*-algebra','STRUCT',5,set(),None),
('A.6',7,'group von Neumann algebra L(F_inv)','STRUCT',5,set(),None),
('A.7',7,'KMS states on C*_red','THERMAL_SRS',5,{'THERM','STRAUCH'},'kms_thermal'),
('A.8',7,'free convolution of measures','STRUCT',5,set(),None),
('A.9',7,'free entropy / free Fisher info','ENTROPY_BERN',5,set(),'entropy_free'),
('A.10',7,'F_inv(E) as monoidal category','STRUCT',4,set(),None),
('A.11',7,'ZX-calculus diagrammatic reasoning','STRUCT',5,{'FF','A4'},None),
('A.12',7,'monoidal functors','STRUCT',5,set(),None),
('A.13',7,'Brownian motion as continuum limit','COARSE_STRAUCH',5,{'STRAUCH'},'brownian'),
('A.14',7,'SDEs on L²','STRUCT',5,{'STRAUCH'},None),
('A.15',7,'martingales, multiway filtration','STRUCT',5,set(),None),
('A.16',7,'modular forms (spectral)','MODULAR',6,{'SRS','CRYSTAL'},'modular_form'),
('A.17',7,'automorphic L-functions','MODULAR',7,{'SRS','CRYSTAL','C_REP'},'L_function'),
('A.18',7,'Selberg zeta function','MODULAR',6,{'SRS','CRYSTAL'},'selberg_zeta'),
('A.19',7,'quantum gravity operations','STRUCT',6,{'NC_GEOM'},None),
('A.20',7,'TQFT operations','TQFT',6,{'COMPACT','LIE'},'tqft'),
('A.21',7,'CFT operators (OPE, Virasoro)','STRUCT',6,{'NC_GEOM','LIE','C_REP'},None),
]
assert len(CATALOG) == 219, f"CATALOG has {len(CATALOG)} entries (expected 219 = 195 + T1.3 additions)"

# ---------------- wave state + step ----------------
@dataclass
class WaveState:
    refinements: frozenset
    tags: set
    fired: list                # list of op tuples in order
    fired_ids: set
    refinements_used: set      # refinement keys already counted (legacy, kept for trace)
    templates_used: set        # T1.1: Φ-templates already counted (template-level dedupe)
    Phi_total: float
    L_total: int
    objects: list

    @property
    def Net(self): return self.Phi_total - self.L_total

def lean_class_count(refs):
    """Class count for substrate-counting (lean) refinements only."""
    lean = refs & {'reduced','cyclic','abelian'}
    return LEAN_REF_COUNT.get(frozenset(lean), n_raw())

def marginal_Phi(state, op, use_live: bool = False):
    """Marginal Φ contribution given current state.

    T1.1 update (2026-04-27): non-lean templates dedupe at the *template* level,
    not the refinement-key level. Multiple refinement keys sharing a Φ-template
    correspond to different views of the same underlying substrate compression
    (e.g., MODULAR_FORM / L_FUNCTION / SELBERG_ZETA all express the substrate's
    Hecke-eigenvalue structure; THERMAL_Z / THERMAL_F / THERMAL_B all express
    the thermal partition). The first op in a template fires at full Φ;
    subsequent ops in the same template fire at 0.

    T2.1g update (2026-04-27): when `use_live=True`, ops in `LIVE_PHI_OVERRIDES`
    use their live-executor value (computed in substrate_state.py from actual
    partition/spectral/Clifford arithmetic) instead of the template lookup.
    Live overrides are NOT subject to template dedupe — each live-overridden op
    contributes its own per-op compression independently.
    """
    op_id, layer, name, tmpl, L, extras, ref = op
    if use_live and op_id in LIVE_PHI_OVERRIDES:
        # Live override: per-op value, no template dedupe
        return LIVE_PHI_OVERRIDES[op_id]
    if tmpl == 'STRUCT' or tmpl == 'CLASSIFYING':
        return 0.0
    if tmpl in ('INVOL','CYCL','QUOT_ABEL'):
        # Lean refinements: marginal Φ from exact substrate-counting (overlap-aware)
        if ref in state.refinements:
            return 0.0
        new_refs = state.refinements | {ref}
        before = lean_class_count(state.refinements)
        after  = lean_class_count(new_refs)
        return math.log2(before / after) if after < before else 0.0
    # Non-lean: dedupe by TEMPLATE (T1.1)
    if tmpl in state.templates_used:
        return 0.0
    return PHI_TEMPLATE.get(tmpl, 0.0) or 0.0

def can_fire(state, op):
    op_id, layer, name, tmpl, L, extras, ref = op
    if op_id in state.fired_ids: return False
    return extras.issubset(state.tags)

# Load-bearing op set: ops that appear in some chain.op_ids in audit_pilot.
# Imported lazily to avoid circular import; populated on first use.
_LOAD_BEARING_OPS: set[str] = set()

def _load_bearing_ops() -> set[str]:
    """Return the set of op_ids that load-bear for some prediction chain.
    Imports CHAINS from audit_pilot lazily."""
    global _LOAD_BEARING_OPS
    if _LOAD_BEARING_OPS:
        return _LOAD_BEARING_OPS
    try:
        import os as _os, sys as _sys
        _here = _os.path.dirname(_os.path.abspath(__file__))
        if _here not in _sys.path:
            _sys.path.insert(0, _here)
        from audit_pilot import CHAINS  # noqa: WPS433
        ids = set()
        for c in CHAINS.values():
            ids.update(c.get('op_ids', []))
        _LOAD_BEARING_OPS = ids
    except Exception:
        _LOAD_BEARING_OPS = set()
    return _LOAD_BEARING_OPS

def passes_a2(state, op, Phi: float, L: int) -> bool:
    """A2-strict gate: an op contributes to the substrate's halt state iff
    it pays for itself in the integrated bit budget (substrate + prediction).

    An op fires under A2-strict iff ANY of:
      - It's a closure op (in ESTABLISHES) — structurally required to unlock tags.
      - Its substrate Φ_marg ≥ L_marg — substrate-side compression pays directly.
      - It's load-bearing for at least one prediction chain (in some chain.op_ids)
        — its L is amortized into the prediction-side compression Σ B_pred.

    Pure-description ops (no substrate Φ, not used by any prediction, not a
    closure) don't pass — they're catalog padding, not framework content.
    """
    op_id = op[0]
    if op_id in ESTABLISHES:
        return True
    if Phi >= L:
        return True
    if op_id in _load_bearing_ops():
        return True
    return False

def formal_L(op) -> int:
    """T1.2 formal L encoding (closure-amortized, 2026-04-27).

    Per-op L = bits to describe THIS op alone given its preconditions satisfied.

    The hand-rated L scheme double-counts: every downstream op pays its own
    L even when the closure theorem upstream already paid the descriptive
    cost. E.g., op 6.13 (Riemann curvature) inherits Layer 6's closure from
    op 6.10 (Lorentzian metric) which inherits from the Γ-cone Minkowski
    theorem. Charging 6.13 its hand-rated 3 bits would re-pay for content
    already in the upstream closure.

    Closure-amortized rule:
      - Closure ops (in ESTABLISHES): pay full hand-rated L (theorem cost).
      - Refinement-producing ops (`ref` field set, non-STRUCT template):
        pay full hand-rated L (introduce new compression measure).
      - Lean-refinement ops (INVOL/CYCL/QUOT_ABEL): pay hand-rated L (their
        compression IS the substrate's foundational refinement chain).
      - All other ops (downstream consumers; STRUCT with `extras` non-empty
        or pure-structural definitions): 1 bit (amortized into upstream).

    Conservative: still charges 1 bit per downstream op for "use the closure"
    overhead. A more aggressive scheme would set this to 0 for STRUCT-only
    downstream consumers; 1 bit is the conservative middle ground.
    """
    op_id, _layer, _name, tmpl, L_hand, extras, ref = op
    if op_id in ESTABLISHES:
        return L_hand                      # closure step: pay theorem cost
    if tmpl in ('INVOL', 'CYCL', 'QUOT_ABEL'):
        return L_hand                      # lean refinement: foundational
    if ref and tmpl not in ('STRUCT', 'CLASSIFYING'):
        return L_hand                      # introduces new refinement / compression
    return 1                                # downstream consumer: amortized to 1 bit

def fire(state, op, Phi, use_formal_L: bool = False):
    op_id, layer, name, tmpl, L, extras, ref = op
    new_refs = state.refinements
    new_refs_used = state.refinements_used
    new_tmpls_used = state.templates_used
    if ref:
        if ref in {'reduced','cyclic','abelian'}:
            new_refs = state.refinements | {ref}
        else:
            new_refs_used = state.refinements_used | {ref}
    # T1.1: track template-level dedupe for non-lean Φ templates
    if tmpl not in ('STRUCT','CLASSIFYING','INVOL','CYCL','QUOT_ABEL'):
        new_tmpls_used = state.templates_used | {tmpl}
    new_obj = f"[{op_id}] {name}"
    L_actual = formal_L(op) if use_formal_L else L
    return WaveState(
        refinements = frozenset(new_refs),
        tags = state.tags | ESTABLISHES.get(op_id, set()),
        fired = state.fired + [op],
        fired_ids = state.fired_ids | {op_id},
        refinements_used = new_refs_used,
        templates_used = new_tmpls_used,
        Phi_total = state.Phi_total + Phi,
        L_total = state.L_total + L_actual,
        objects = state.objects + [new_obj],
    )

def step_cascade(state, use_live: bool = False, use_formal_L: bool = False,
                  strict_a2: bool = False):
    """Process catalog in order; fire each firable op exactly once.

    Default mode (strict_a2=False): catalog enumeration — fire every op whose
    tag preconditions are met. This is for catalog-completeness audits.

    A2-strict mode (strict_a2=True): only fire ops where compression pays for
    itself (Φ_marg ≥ L_marg), with closure ops in ESTABLISHES exempted.
    This is the framework's actual A2-T construction: the substrate is the
    SUBSET of catalog ops that pass the MDL waterline gate."""
    for op in CATALOG:
        if not can_fire(state, op): continue
        Phi = marginal_Phi(state, op, use_live=use_live)
        L = formal_L(op) if use_formal_L else op[4]
        if strict_a2 and not passes_a2(state, op, Phi, L):
            continue
        return fire(state, op, Phi, use_formal_L=use_formal_L)
    return None

def run_full(use_live: bool = False, use_formal_L: bool = False,
              strict_a2: bool = False):
    state = WaveState(
        refinements = frozenset(),
        tags = set(INITIAL_TAGS),
        fired = [], fired_ids = set(),
        refinements_used = set(),
        templates_used = set(),
        Phi_total = 0.0, L_total = 0,
        objects = [],
    )
    history = [state]
    while True:
        nxt = step_cascade(state, use_live=use_live, use_formal_L=use_formal_L,
                           strict_a2=strict_a2)
        if nxt is None: break
        state = nxt
        history.append(state)
    return state, history

# ---------------- run + report ----------------
if __name__ == '__main__':
    import sys
    use_live = '--live' in sys.argv
    use_formal_L = '--formal-L' in sys.argv
    aggressive = '--aggressive-L' in sys.argv
    strict_a2 = '--strict-a2' in sys.argv

    # Aggressive variant: downstream consumers pay 0 (not 1) — pure inheritance
    if aggressive:
        _formal_L_orig = formal_L
        def formal_L(op):  # noqa: F811
            v = _formal_L_orig(op)
            return 0 if v == 1 else v
        # Rebind in fire's scope; simpler to just patch the global
        globals()['formal_L'] = formal_L
        use_formal_L = True

    print("="*100)
    print(f"FULL WAVE SIMULATOR | {len(CATALOG)} ops | reference (n={N_REF}, |E|={E})")
    print(f"Initial tags: {sorted(INITIAL_TAGS)}")
    print(f"Open frontier (cannot establish): {sorted(PARTIAL_TAGS)}")
    if use_live:
        print(f"Live Φ overrides ENABLED for ops: {sorted(LIVE_PHI_OVERRIDES.keys())}")
    if use_formal_L:
        print(f"Formal L (closure-amortized) ENABLED")
    if strict_a2:
        print(f"Strict A2 gating ENABLED (Φ_marg ≥ L_marg required, closure ops exempt)")
    print("="*100)

    final, hist = run_full(use_live=use_live, use_formal_L=use_formal_L,
                            strict_a2=strict_a2)

    # Trace: print only ops that actually fire, in cascade order, with milestone markers
    print(f"\n{'tick':>4} {'op':>5} L{'lvl':<1} {'name':<46} {'tags-est':<28} {'Φ':>6} {'L':>3} {'Net':>7}")
    print("-"*108)
    prev_tags = set(INITIAL_TAGS)
    for i, st in enumerate(hist[1:], 1):
        op = st.fired[-1]
        op_id, layer, name, tmpl, L, extras, ref = op
        Phi = st.Phi_total - hist[i-1].Phi_total
        new_tags = st.tags - prev_tags
        tag_str = ",".join(sorted(new_tags)) if new_tags else ""
        prev_tags = set(st.tags)
        marker = "★" if new_tags else " "
        print(f"{i:>4} {op_id:>5} L{layer} {marker} {name:<44} {tag_str:<28} {Phi:>6.2f} {L:>3} {Phi-L:>+7.2f}")

    print("-"*108)
    print(f"\n{'='*100}\nHALT after {len(final.fired)} firings (of {len(CATALOG)} catalog ops)")
    print(f"{'='*100}")
    fired_ids = final.fired_ids
    not_fired = [op for op in CATALOG if op[0] not in fired_ids]
    print(f"\nFinal tags established: {sorted(final.tags)}")
    print(f"Tags NEVER established: {sorted((ALL_TAGS - final.tags))}")
    print(f"\nTotals: Φ = {final.Phi_total:.2f} bits | L = {final.L_total} bits | Net = {final.Net:+.2f} bits")
    print(f"Refinements used: lean={sorted(final.refinements)} | other={len(final.refinements_used)} keys")

    print(f"\nFired by layer:")
    from collections import Counter
    fired_by_layer = Counter(op[1] for op in final.fired)
    not_fired_by_layer = Counter(op[1] for op in not_fired)
    for layer in range(8):
        f = fired_by_layer.get(layer, 0)
        n_layer = sum(1 for op in CATALOG if op[1] == layer)
        label = f"L{layer}" if layer < 7 else "App"
        print(f"  {label}: {f}/{n_layer} fired ({f/n_layer*100:.0f}%)")

    print(f"\n{len(not_fired)} ops did NOT fire.  Reasons (top blockers):")
    blocker_count = Counter()
    for op in not_fired:
        op_id, layer, name, tmpl, L, extras, ref = op
        missing = extras - final.tags
        if missing:
            blocker_count[frozenset(missing)] += 1
        else:
            blocker_count[frozenset({'NO-Φ-NO-CONSTRUCT'})] += 1
    for missing, count in blocker_count.most_common(10):
        if missing == frozenset({'NO-Φ-NO-CONSTRUCT'}):
            label = "fired into already-saturated refinement (no marginal Φ, no new object)"
        else:
            label = "missing tag(s): " + ",".join(sorted(missing))
        print(f"  {count:3d} ops — {label}")

    # T2.1g: bookkeeping-vs-live comparison
    if not use_live:
        print(f"\n{'='*100}")
        print(f"T2.1g Comparison: bookkeeping (this run) vs live overrides")
        print(f"{'='*100}")
        live_final, _ = run_full(use_live=True)
        delta = live_final.Phi_total - final.Phi_total
        print(f"  Φ_total (bookkeeping):  {final.Phi_total:>9.4f} bits  Net = {final.Net:>+8.4f}")
        print(f"  Φ_total (live override): {live_final.Phi_total:>9.4f} bits  Net = {live_final.Net:>+8.4f}")
        print(f"  Δ Φ_total: {delta:+.4f} bits  ({len(LIVE_PHI_OVERRIDES)} live ops)")
        print(f"  Per-op deltas (live − bookkeeping):")
        for op_id in sorted(LIVE_PHI_OVERRIDES.keys()):
            live_v = LIVE_PHI_OVERRIDES[op_id]
            # Find this op in CATALOG to look up its template
            op_entry = next((o for o in CATALOG if o[0] == op_id), None)
            if op_entry is None:
                continue
            _id, _layer, name, tmpl, _L, _ex, _ref = op_entry
            template_v = PHI_TEMPLATE.get(tmpl, 0.0) or 0.0
            # Note: bookkeeping value depends on dedupe state; this prints the
            # template value, not the actual contribution after dedupe.
            print(f"    {op_id:>5}  {name:<46}  template({tmpl})={template_v:>6.3f}  live={live_v:>6.3f}  Δ={live_v-template_v:+.3f}")
    else:
        print(f"\nLive-override mode: this run already incorporates live values for "
              f"{len(LIVE_PHI_OVERRIDES)} ops.")

    # T2.4: PHI_TEMPLATE coverage report — show which templates have live-overridden ops
    print(f"\n{'='*100}")
    print(f"T2.4 PHI_TEMPLATE coverage: live-override breakdown by template")
    print(f"{'='*100}")
    from collections import defaultdict
    template_to_ops = defaultdict(list)
    template_live_count = defaultdict(int)
    for op in CATALOG:
        op_id, _layer, _name, tmpl, *_ = op
        template_to_ops[tmpl].append(op_id)
        if op_id in LIVE_PHI_OVERRIDES:
            template_live_count[tmpl] += 1
    print(f"  {'template':<18} {'ops':>5} {'live':>5} {'%cov':>6}  ops_live")
    print(f"  {'-'*70}")
    for tmpl in sorted(template_to_ops, key=lambda t: (-template_live_count[t], t)):
        n_ops = len(template_to_ops[tmpl])
        n_live = template_live_count[tmpl]
        pct = 100 * n_live / n_ops if n_ops > 0 else 0.0
        live_ids = sorted(o for o in template_to_ops[tmpl] if o in LIVE_PHI_OVERRIDES)
        live_str = ', '.join(live_ids) if live_ids else '—'
        print(f"  {tmpl:<18} {n_ops:>5} {n_live:>5} {pct:>5.1f}%  {live_str}")
    total_live = sum(template_live_count.values())
    total_ops = len(CATALOG)
    print(f"  {'-'*70}")
    print(f"  Total: {total_live} / {total_ops} catalog ops have live overrides "
          f"({100*total_live/total_ops:.1f}% coverage).")
    print(f"  Note: lean ops (INVOL, CYCL, QUOT_ABEL) use simulator's analytic "
          f"closed-form (asymptotic at n=N_REF=10); not in LIVE_PHI_OVERRIDES "
          f"but are 'live-equivalent' at the closed-form level.")
