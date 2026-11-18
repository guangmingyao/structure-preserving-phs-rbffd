"""Independent spatial-spectrum, RK4 time-step and midpoint stability audit.
Uses C++-generated r11+P5 matrices. Run from project root.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import splu

p=argparse.ArgumentParser()
p.add_argument('--data',default='data')
p.add_argument('--N',type=int,default=21)
p.add_argument('--T',type=float,default=20.)
p.add_argument('--rk4-steps',default='0.001,0.0005,0.00025')
p.add_argument('--midpoint-steps',default='0.01,0.005')
a=p.parse_args(); D=Path(a.data)
N=a.N; h=np.pi/(N-1); interior=N-2
x=np.arange(1,N-1)*h
u0=np.outer(np.sin(2*x),np.sin(3*x)).reshape(-1)
results=[]
for method in ('KNN','Reflected'):
 name=f'{method}_r11_P5_N{N}'
 f=D/f'matrix_{name}.csv'
 if not f.exists():
  print('SKIP: missing',f);continue
 z=pd.read_csv(f);n=int(max(z.row.max(),z.col.max())+1)
 L=coo_matrix((z.value,(z.row,z.col)),shape=(n,n)).tocsr()
 if n!=u0.size:raise ValueError(f'Grid/matrix dimension mismatch {n} vs {u0.size}')
 ev=np.linalg.eigvals(L.toarray());growth=float(np.max(np.abs(ev.imag)))
 print(f'{name}: maximum |Im eigenvalue(L)|={growth:.10g}; continuous-time generator maximum Re eigenvalue={growth:.10g}')
 print('  A complex eigenvalue of L implies an exponentially growing semidiscrete mode; eigenvalue alone does not prove the selected initial condition excites it.')
 def record(kind,dt,z):
  t=a.T
  exact=u0*np.exp(-13j*t)
  err=np.linalg.norm(z-exact)/np.linalg.norm(exact)
  mass=abs(np.vdot(z,z).real/np.vdot(u0,u0).real-1)
  E0=-np.vdot(u0,L@u0).real
  E=-np.vdot(z,L@z).real
  edrift=abs(E/E0-1)
  results.append(dict(method=method,N=N,T=t,integrator=kind,dt=dt,relative_l2_error=err,mass_relative_drift=mass,energy_relative_drift=edrift,max_abs_imag_eig_L=growth))
  print(f'  {kind:8s} dt={dt:.8g} error={err:.8g} mass_drift={mass:.8g} energy_drift={edrift:.8g}')
 for dt_request in [float(v) for v in a.rk4_steps.split(',') if v.strip()]:
  steps=int(np.ceil(a.T/dt_request));dt=a.T/steps
  z0=u0.astype(complex)
  def rhs(y):return 1j*(L@y)
  for it in range(steps):
   k1=rhs(z0);k2=rhs(z0+dt*k1/2);k3=rhs(z0+dt*k2/2);k4=rhs(z0+dt*k3)
   z0=z0+(dt/6)*(k1+2*k2+2*k3+k4)
  record('RK4',dt,z0)
 for dt_request in [float(v) for v in a.midpoint_steps.split(',') if v.strip()]:
  steps=int(np.ceil(a.T/dt_request));dt=a.T/steps
  B=eye(n,format='csc')-1j*dt/2*L.tocsc()
  C=eye(n,format='csc')+1j*dt/2*L.tocsc()
  lu=splu(B);z0=u0.astype(complex)
  for it in range(steps):z0=lu.solve(C@z0)
  record('Midpoint',dt,z0)
output=D/f'stability_audit_N{N}_T{a.T:g}.csv';pd.DataFrame(results).to_csv(output,index=False)
print('saved',output)
