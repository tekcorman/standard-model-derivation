import numpy as np
from fractions import Fraction
np.set_printoptions(precision=4, suppress=True, linewidth=160)

I2=np.eye(2); sz=np.diag([1.,-1.]); Pp=np.diag([1.,0.])
g4=np.kron(sz,I2); h4=np.kron(I2,sz); hand4=np.kron(sz,I2)
I3=np.eye(3); Vw=np.diag([-1.,0.,1.]); I4=np.eye(4)
G=np.kron(g4,I3); H=np.kron(h4,I3); V=np.kron(I4,Vw)
HXH=np.kron(hand4@h4,I3)
W=2*V+HXH
T4=np.kron(Pp,0.5*sz); T=np.kron(T4,I3)

def diagvec(M): return np.real(np.diag(M))

# 1) Is the product g*h = handedness*helicity in the span? It equals HXH (since hand4=g4).
print("hand4 == g4 ?", np.allclose(hand4,g4))
print("HXH == G@H ?", np.allclose(HXH, G@H))   # so handedness x helicity = g*h
# So W = 2V + g*h exactly. Is g*h independent of {I,g,h,V}?
ops={"I":np.eye(12),"g":G,"h":H,"V":V,"gh":G@H}
M=np.array([diagvec(o) for o in ops.values()])
print("rank{I,g,h,V,gh} =", np.linalg.matrix_rank(M,tol=1e-9), " (5 => gh independent)")

# So the linear span {I,g,h,V,W} has dim 5, and W = 2V + gh where gh is a NEW independent generator.
# Thus W introduces the product gh into the linear span. Good: W's content beyond {V} is exactly g*h.

# 2) The T-relation: T = (1/4)h - (1/2)V + (1/4)W. Substitute W=2V+gh:
#    (1/4)W = (1/2)V + (1/4)gh ; so RHS = (1/4)h -(1/2)V +(1/2)V +(1/4)gh = (1/4)h + (1/4)gh = (1/4)h(I+g).
# Check:
RHS = 0.25*H + 0.25*(G@H)
print("\nT == (1/4) h (I+g) ?", np.allclose(T, RHS))
print("i.e. T = (1/4)*h*(I+g) = (1/2)*h * P_{chi=+}  where P_{chi=+}=(I+g)/2")
Pchi=(np.eye(12)+G)/2
print("T == (1/2) h P_{+} ?", np.allclose(T, 0.5*H@Pchi))

# 3) Now the honesty check: T's overall scale.
# T was built as Cartan (1/2 sz) on helicity restricted to chi=+ half. The '1/2' is the SU(2) Cartan
# normalization (Cartan of su(2) has eigenvalues +-1/2 in the fundamental). The 'restrict to chi=+ half'
# is the 'acts on one chirality half' instruction. So T = (1/2) * (helicity projected to one half).
# In terms of the OTHER charges: helicity h, chirality projector (I+g)/2. So
#    T = (1/2) h (I+g)/2 = (1/4)(h + gh).
# Whether the relation's coefficients are 'forced':
#   - The combination h + gh = h(I+g) = 2 h P_+ is FORCED structurally (h restricted to the chi=+ block).
#   - The overall 1/4 = (1/2 Cartan)*(1/2 projector) carries TWO normalization conventions:
#        * the SU(2) Cartan 1/2 (a CHOICE of su(2) normalization; could be 1 for sigma_z),
#        * the projector 1/2 from (I+g)/2 (FORCED: a projector is idempotent, coefficient fixed).
#   So the *shape* T proportional to h(I+g) is FORCED; the scalar in front (1/2 vs 1) is a free su(2)
#   normalization. Show both:
for c,lab in [(0.5,"Cartan=1/2 (fundamental su(2))"),(1.0,"Cartan=1 (sigma_z)")]:
    Tc=np.kron(np.kron(Pp,c*sz),I3)
    # express Tc = ? * h(I+g)
    coeff = Tc[0,0]/ (H@(np.eye(12)+G))[0,0]
    print(f"  with {lab}: T = {coeff:.4f} * h(I+g)  -> in W,V,h form: coefficients scale by {2*c:.2f}")

# 4) Re-express the relation with W kept (the asked form) and state forced vs choice per coefficient.
# 0 = -(1/4)h + (1/2)V - (1/4)W + T   with T at fundamental norm.
# Equivalent: T = (1/4)(W - 2V) + (1/4)h = (1/4)(gh) + (1/4)h  [using W-2V = gh]  ✓ matches.
print("\nFINAL: T = (1/4)(W - 2V + h).  Check:", np.allclose(T, 0.25*(W-2*V+H)))

print("\n=== span structure recap ===")
# independent generators of the linear span:
ops={"I":np.eye(12),"g":G,"h":H,"V":V,"gh":G@H}
M=np.array([diagvec(o) for o in ops.values()])
print("dim span{I,g,h,V,gh} =", np.linalg.matrix_rank(M,tol=1e-9))
# W and T expressed in this basis:
def express(target):
    A=M.T  # 12 x 5
    coef,res,rk,sv=np.linalg.lstsq(A,diagvec(target),rcond=None)
    rec=A@coef
    ok=np.allclose(rec,diagvec(target))
    return [Fraction(c).limit_denominator(8) for c in coef], ok
cW,okW=express(W); cT,okT=express(T)
print("W in {I,g,h,V,gh}:", dict(zip(ops,cW)), "exact?",okW)
print("T in {I,g,h,V,gh}:", dict(zip(ops,cT)), "exact?",okT)
