from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--data',default='data');p.add_argument('--figures',default='figures');a=p.parse_args()
d=pd.read_csv(Path(a.data)/'higher_grid_knn_sensitivity.csv');f=Path(a.figures);f.mkdir(exist_ok=True,parents=True)
base=d[(d.method=='Reflected')|((d.method=='KNN')&(d.stencil_points==29)&(d.tie_break=='xy_ascending'))]
for field,ylabel,name in [('frequency_error_k2_l3','Frequency error |ω_h−13|','frequency'),('shape_residual_rel_l2','Relative eigenmode shape residual','shape'),('boundary_layer1_inf','First-layer max residual','boundary'),('symmetry_defect','Relative matrix symmetry defect','symmetry'),('rk4_T1_rel_l2_error','Sparse RK4 relative error at T=1','rk4')]:
 fig,ax=plt.subplots(figsize=(7,4.5))
 for method in ['KNN','Reflected']:
  q=base[(base.method==method)&np.isfinite(base[field])].sort_values('N')
  if field=='symmetry_defect':ax.plot(q.N,q[field],'o-',label=method)
  else:ax.loglog(q.N,q[field],'o-',label=method)
 ax.set(xlabel='Grid nodes per side N',ylabel=ylabel,title='Matched r11+P5: '+ylabel);ax.grid(True,which='both',alpha=.25);ax.legend();fig.tight_layout();fig.savefig(f/f'higher_grid_{name}.png',dpi=180);plt.close(fig)
for field,ylabel,name in [('frequency_error_k2_l3','Frequency error','frequency'),('boundary_layer1_inf','First-layer maximum residual','boundary'),('symmetry_defect','Relative symmetry defect','symmetry')]:
 fig,ax=plt.subplots(figsize=(7,4.5))
 for ns in [29,35,41]:
  q=d[(d.method=='KNN')&(d.stencil_points==ns)]
  lo=q.groupby('N')[field].min();hi=q.groupby('N')[field].max();mid=q.groupby('N')[field].median()
  ax.plot(mid.index,mid.values,'o-',label=f'KNN {ns} points');ax.fill_between(mid.index,lo.values,hi.values,alpha=.16)
 ax.set(xlabel='Grid nodes per side N',ylabel=ylabel,title='KNN tie-break sensitivity (bands: min–max)');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(f/f'knn_sensitivity_{name}.png',dpi=180);plt.close(fig)
print('Higher-grid plots generated')
