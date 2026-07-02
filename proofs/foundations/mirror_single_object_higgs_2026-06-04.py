"""
The single mirror, correctly identified: the odd-grade Higgs does BOTH masses.

This CORRECTS the strawman in `mirror_bridge_gamma7_hodge_2026-06-04.py`, which
tested gamma7 (the grade-6 pseudoscalar = the chirality GRADING, even & central)
and wrongly concluded "fermion-mass and boson-mass mirrors are different operators."

The framework's Higgs is a GRADE-1 (odd) element (theorem_updown_split_conjugate_higgs:
"a Higgs-doublet component is a grade-1 odd element ... an odd-grade element flips
handedness"). Odd grade does both jobs with ONE object:

  (a) it ANTICOMMUTES with gamma7      -> flips fermion L<->R chirality (Yukawa mass)
  (b) it is NON-central on the gauge   -> breaks the gauge symmetry (W/Z mass)
      bivectors (anticommutes with the bivectors sharing its index)

gamma7 (grade-6, even) cannot do (b) (it is adjoint-central). That is the chirality
grading, NOT the mirror/Higgs. The single-object locking the framework already
builds (Higgs = odd-grade edge qubit) is therefore real; the earlier 'REFUTED'
verdict was a mis-identification of the object.

Gates:
  G1  gamma7 is the GRADING: even-central on bivectors (gaps nothing), anticommutes
      with all grade-1 vectors. Not the Higgs.
  G2  a grade-1 (odd) Higgs f1 anticommutes with gamma7  -> chirality flip (fermion mass).
  G3  the same f1 is NON-central on the gauge bivectors  -> gaps gauge bosons (boson mass).
  G4  contrast: no even-grade central element can do (b); odd grade is required for both.
"""
import numpy as np

PASS = []
def check(name, cond):
    PASS.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

I2 = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], complex); Yk = np.array([[0,-1j],[1j,0]]); Zk = np.array([[1,0],[0,-1]], complex)
def kron(*o):
    r = np.array([[1]], complex)
    for m in o: r = np.kron(r, m)
    return r
c = [kron(*([Zk]*k + [p] + [I2]*(2-k))) for k in range(3) for p in (X, Yk)]
g7 = c[0]@c[1]@c[2]@c[3]@c[4]@c[5]
for ph in (1, 1j, -1, -1j):
    if np.allclose((ph*g7)@(ph*g7), np.eye(8)): g7 = ph*g7; break
bivs = [c[i]@c[j] for i in range(6) for j in range(i+1, 6)]   # 15 gauge bivectors
def anti(A, B): return np.allclose(A@B + B@A, 0)
def comm(A, B): return np.allclose(A@B - B@A, 0)

# G1 — gamma7 is the chirality grading, not the Higgs
print("G1  gamma7 (grade-6, even) = chirality grading: central on gauge, odd on vectors")
check("gamma7 commutes with all 15 bivectors (central; gaps nothing)",
      sum(comm(g7, B) for B in bivs) == 15)
check("gamma7 anticommutes with all 6 grade-1 vectors (it is the grading)",
      sum(anti(g7, ci) for ci in c) == 6)

# G2 — odd-grade Higgs flips chirality (fermion mass)
print("G2  grade-1 Higgs f1 anticommutes with gamma7 -> flips L<->R (fermion mass)")
f1 = c[0]
check("f1 anticommutes with gamma7", anti(f1, g7))

# G3 — same Higgs gaps gauge bosons (boson mass)
print("G3  same f1 is NON-central on gauge bivectors -> gaps gauge bosons (boson mass)")
ngap = sum(not comm(f1, B) for B in bivs)
check("f1 has non-trivial adjoint on >0 bivectors (not central)", ngap > 0)
check("f1 anticommutes with exactly the 5 bivectors sharing its index",
      sum(anti(f1, B) for B in bivs) == 5)

# G4 — the point: ONE odd object does both; gamma7 (even) could not do G3
print("G4  one odd-grade object does both; the even grading gamma7 cannot gap gauge")
check("f1 does (a) chirality-flip AND (b) gauge-gap; gamma7 does only (a)",
      anti(f1, g7) and ngap > 0 and sum(comm(g7, B) for B in bivs) == 15)

print()
print("VERDICT: the single 'mirror' that gives BOTH masses is the ODD-GRADE Higgs (f1),")
print("         which the framework already builds (theorem_updown_split, grade-1).")
print("         gamma7 is the chirality GRADING (and = Hodge-* on forms), not the Higgs.")
print("         The earlier 'single mirror REFUTED' was a mis-identification -> REVERSED.")
print()
print(f"{sum(PASS)}/{len(PASS)} gates PASS")
if not all(PASS):
    raise SystemExit("single-object Higgs probe: FAIL")
