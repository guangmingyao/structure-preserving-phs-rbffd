from pathlib import Path
import pandas as pd
import numpy as np
D=Path(__file__).resolve().parents[1]/'data'
s=pd.read_csv(D/'basis_screening.csv');m=pd.read_csv(D/'modal_rk4.csv');mom=pd.read_csv(D/'moments.csv')
assert len(s)==10 and len(m)==72
assert abs(s.loc[s.kernel=='r11','relative_l2_error'].iloc[0]-.0011518)<1e-5
assert abs(mom.loc[mom.kernel=='r11_P5','M60'].iloc[0]-1.813544100277468)<1e-8
for N in (21,31):
 for method in ('KNN','Reflected'):
  name=f'{method}_r11_P5_N{N}'
  d=pd.read_csv(D/f'diagnostics_{name}.csv');assert np.isfinite(d.to_numpy()).all(),name
  if method=='Reflected':assert d.symmetry_defect.iloc[0]<1e-9
  for integ in ('RK4','Midpoint'):
   t=pd.read_csv(D/f'trajectory_{name}_{integ}_T2.csv');assert len(t)>10 and np.isfinite(t.to_numpy()).all(),name
print('PASS: all new r11+P5 matched diagnostics, trajectories, and reference screening')
