#!/usr/bin/env python3
"""Validated interval Gaussian elimination for 29-point PHS r^7 + P4 weights.
Uses mpmath interval arithmetic with outward-rounded MPFI-style endpoints.
No floating point linear algebra is used to certify the inequalities.
"""
from mpmath import iv
iv.dps=70
S=sorted([(i,j) for i in range(-3,4) for j in range(-3,4) if i*i+j*j<=9],key=lambda p:(p[0]*p[0]+p[1]*p[1],p[0],p[1]))
P=[(a,d-a) for d in range(5) for a in range(d+1)]
n=len(S);m=len(P);D=n+m
Z=iv.mpf(0); O=iv.mpf(1)
def v(k):return iv.mpf(k)
def r7(q):return v(q)**3*iv.sqrt(v(q)) if q else Z
def r5(q):return v(q)**2*iv.sqrt(v(q)) if q else Z
A=[[Z for j in range(D)] for i in range(D)]
b=[Z for i in range(D)]
for i,(x,y) in enumerate(S):
 for j,(xx,yy) in enumerate(S):A[i][j]=r7((x-xx)**2+(y-yy)**2)
 for j,(a,c) in enumerate(P):
  z=v(x**a*y**c);A[i][n+j]=z;A[n+j][i]=z
 b[i]=v(49)*r5(x*x+y*y)
for j,(a,c) in enumerate(P):b[n+j]=v(2 if (a,c) in [(2,0),(0,2)] else 0)
# Interval elimination with fixed pivoting; a pivot is accepted only when its interval excludes zero.
def excludes_zero(x):return x.a>0 or x.b<0
for k in range(D):
 if not excludes_zero(A[k][k]):
  swap=next((r for r in range(k+1,D) if excludes_zero(A[r][k])),None)
  if swap is None:raise RuntimeError('No certified nonzero pivot at '+str(k))
  A[k],A[swap]=A[swap],A[k];b[k],b[swap]=b[swap],b[k]
 pivot=A[k][k]
 for i in range(k+1,D):
  if A[i][k]==Z:continue
  q=A[i][k]/pivot
  A[i][k]=Z
  for j in range(k+1,D):A[i][j]-=q*A[k][j]
  b[i]-=q*b[k]
x=[Z]*D
for k in range(D-1,-1,-1):
 x[k]=(b[k]-sum((A[k][j]*x[j] for j in range(k+1,D)),Z))/A[k][k]
# Chebyshev T_a(1-2t) coefficients in ascending order, exact integers.
T=[[1], [1,-2]]
for k in range(2,4):
 z=[0]*(k+1)
 for j,c in enumerate(T[-1]):z[j]+=2*c;z[j+1]-=4*c
 for j,c in enumerate(T[-2]):z[j]-=c
 T.append(z)
C=[[Z for j in range(4)] for i in range(4)]
for (a,beta),w in zip(S,x):
 for i,ci in enumerate(T[abs(a)]):
  for j,cj in enumerate(T[abs(beta)]):C[i][j]+=w*v(ci*cj)
a=-(C[3][0]+C[0][3])/v(2)
bb=(C[2][1]+C[1][2])/v(2)
c=-C[2][2]
margin=v(4)/v(3)-(bb-a)
for label,val in [('a',a),('b',bb),('c',c),('b-a',bb-a),('margin',margin)]:
 print(label,iv.nstr(val,25) if hasattr(iv,'nstr') else str(val))
print('certified:',a.a>0,bb.a>a.b,c.a>0,margin.a>0)
# NOTE: some entries are mathematically zero but intervals contain 0; inspect for correct structural form.
for i in range(4):
 for j in range(4):
  if (i,j) not in [(0,0),(1,0),(0,1),(2,0),(0,2),(3,0),(0,3),(2,1),(1,2),(2,2)]:
   if not (C[i][j].a<=0<=C[i][j].b):print('Unexpected nonzero coefficient',i,j,C[i][j])
print('C00:',C[0][0]);print('C10:',C[1][0]);print('C20:',C[2][0]);print('C11:',C[1][1]);print('C30:',C[3][0]);print('C21:',C[2][1]);print('C22:',C[2][2]);print('C33:',C[3][3]);
