from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

p=argparse.ArgumentParser();p.add_argument('--data',default='data');p.add_argument('--figures',default='figures');p.add_argument('--spectra',action='store_true',help='Compute dense N=21,31 eigenvalues (slow)');a=p.parse_args()
D=Path(a.data);F=Path(a.figures);F.mkdir(parents=True,exist_ok=True)
def save(name):
 plt.tight_layout();plt.savefig(F/name,dpi=200,bbox_inches='tight');plt.close();print('saved',name)
modal=pd.read_csv(D/'modal_rk4.csv')
for case in ['mixed','favorable']:
 for T in [1,20]:
  fig,ax=plt.subplots(figsize=(7,4.5))
  for method in ['r11_P5','r7_P4','FD4']:
   z=modal[(modal["case"]==case)&(modal["T"]==T)&(modal["method"]==method)].sort_values('N')
   if z.empty: raise ValueError(f'No modal rows for {case=}, {T=}, {method=}')
   ax.loglog(z['N'],z['relative_l2_error'],'o-',label=method)
  ax.set(xlabel='Grid nodes per side N',ylabel='Relative discrete L2 error',title=f'{case.capitalize()} modes, T={T} (modal RK4)');ax.grid(True,which='both',alpha=.2);ax.legend();save(f'convergence_{case}_T{T}.png')
s=pd.read_csv(D/'basis_screening.csv');fig,ax=plt.subplots(figsize=(9,4.5));ax.bar(range(len(s)),s.relative_l2_error);ax.set_yscale('log');ax.set_xticks(range(len(s)));ax.set_xticklabels([f'{r.kernel}+P{r.poly_degree}' for r in s.itertuples()],rotation=40,ha='right');ax.set(ylabel='Relative L2 error',title='PHS–polynomial screening, N=31, T=1');save('basis_screening.png')
mom=pd.read_csv(D/'moments.csv');ratio=np.linspace(.02,1,400);k=1/np.sqrt(1+ratio**2);l=ratio*k;fig,ax=plt.subplots(figsize=(7,4.5))
for r in mom.itertuples():
 coeff=(r.M60*(k**6+l**6)+15*r.M42*(k**4*l**2+k**2*l**4))/720
 ax.plot(ratio,abs(coeff),label=r.kernel)
ax.plot(ratio,(k**6+l**6)/90,label='FD4');ax.set(xlabel='Wavenumber ratio',ylabel='Absolute leading h⁴ coefficient',title='Directional dispersion');ax.legend();ax.grid(alpha=.2);save('directional_dispersion.png')
for N in [21,31]:
 names=[f'KNN_r11_P5_N{N}',f'Reflected_r11_P5_N{N}']
 fig,ax=plt.subplots(figsize=(7,4.5))
 for name in names:
  z=pd.read_csv(D/f'boundary_{name}.csv');ax.semilogy(z.layer,np.maximum(z.max_residual,1e-16),'o-',label=name)
 ax.set(xlabel='Distance from boundary (layers)',ylabel='Maximum |Lφ+13φ|',title=f'Matched r11+P5 boundary residuals, N={N}');ax.legend();ax.grid(alpha=.2);save(f'boundary_residual_N{N}.png')
 for integrator,T in [('RK4',2),('Midpoint',2)]+([('RK4',20)] if N==21 else []):
  for field,label in [('combined_rel_l2_error','Combined relative L2 error'),('mass_relative_drift','Relative mass drift'),('energy_relative_drift','Relative Hamiltonian drift')]:
   fig,ax=plt.subplots(figsize=(7,4.5))
   for name in names:
    z=pd.read_csv(D/f'trajectory_{name}_{integrator}_T{T}.csv');ax.plot(z.time,z[field],label=name)
   ax.set(xlabel='Time',ylabel=label,title=f'r11+P5 {integrator}, N={N}, T={T}');ax.legend();ax.grid(alpha=.2);save(f'{field}_{integrator}_N{N}_T{T}.png')
 if a.spectra:
  for name in names:
   z=pd.read_csv(D/f'matrix_{name}.csv');n=int(max(z.row.max(),z.col.max())+1)
   A=coo_matrix((z.value,(z.row,z.col)),shape=(n,n)).toarray();ev=np.linalg.eigvals(A)
   pd.DataFrame({'real':ev.real,'imag':ev.imag}).to_csv(D/f'eigenvalues_{name}.csv',index=False)
   fig,ax=plt.subplots(figsize=(7,4.5));ax.scatter(ev.real,ev.imag,s=10,alpha=.6);ax.set(xlabel='Real eigenvalue',ylabel='Imaginary eigenvalue',title=f'r11+P5 spectrum: {name}');ax.grid(alpha=.2);save(f'spectrum_{name}.png')
# Independent long-time stability audit, if available.
for auditfile in sorted(D.glob('stability_audit_N*_T*.csv')):
 audit=pd.read_csv(auditfile)
 for integrator in audit.integrator.unique():
  fig,ax=plt.subplots(figsize=(7,4.5))
  for method in ('KNN','Reflected'):
   q=audit[(audit['method']==method)&(audit['integrator']==integrator)].sort_values('dt')
   if not q.empty:ax.loglog(q['dt'],q['relative_l2_error'],'o-',label=method)
  ax.set(xlabel='Time step Δt',ylabel='Relative L2 error at final time',title=f'Stability audit {integrator}, N={int(audit.N.iloc[0])}, T={audit["T"].iloc[0]:g}')
  ax.grid(True,which='both',alpha=.2);ax.legend()
  save(f'{auditfile.stem}_{integrator}_error.png')
